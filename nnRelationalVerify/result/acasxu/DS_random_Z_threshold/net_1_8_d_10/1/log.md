## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 141.076127489203


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367)
1: (-348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983)
2: (-187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022)
3: (-321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637)
4: (-236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.99 + 2.11 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -141.0803599, upper bound: 141.0803599

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803591, upper bound: 141.0803599
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803591, upper bound: 141.0803591
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -141.0803591, upper bound: 141.0803599
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 0, lower bound: -141.0803591, upper bound: 141.0803591

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802608, upper bound: 141.0802773
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802608, upper bound: 141.0802644
time: 0.85 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803467, upper bound: 141.0803462
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803462, upper bound: 141.0803486
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.57 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -141.0802608, upper bound: 141.0802773
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -141.0802608, upper bound: 141.0802644
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -141.0803467, upper bound: 141.0803462
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -141.0803462, upper bound: 141.0803486

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801606, upper bound: 141.0801490
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801492, upper bound: 141.0801651
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799360
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799317
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802864, upper bound: 141.0802588
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802666, upper bound: 141.0802854
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803216, upper bound: 141.0803355
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0803216, upper bound: 141.0803244
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0801606, upper bound: 141.0801490
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0801492, upper bound: 141.0801651
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799360
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799317
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0802864, upper bound: 141.0802588
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0802666, upper bound: 141.0802854
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0803216, upper bound: 141.0803355
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0803216, upper bound: 141.0803244

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800779, upper bound: 141.0800779
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800779, upper bound: 141.0800779
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801270, upper bound: 141.0801435
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801270, upper bound: 141.0801467
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799233
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799315
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799121
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799317
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802610, upper bound: 141.0802586
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802343, upper bound: 141.0802606
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802343, upper bound: 141.0802666
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800326, upper bound: 141.0800326
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800326, upper bound: 141.0800566
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802588, upper bound: 141.0802609
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802588, upper bound: 141.0802590
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0800779, upper bound: 141.0800779
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0800779, upper bound: 141.0800779
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0801270, upper bound: 141.0801435
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0801270, upper bound: 141.0801467
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799233
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799315
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799121
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0799121, upper bound: 141.0799317
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802610, upper bound: 141.0802586
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802343, upper bound: 141.0802606
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802343, upper bound: 141.0802666
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0800326, upper bound: 141.0800326
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0800326, upper bound: 141.0800566
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802588, upper bound: 141.0802609
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.47
Output dim: 0, lower bound: -141.0802588, upper bound: 141.0802590

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801432
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801216
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800628
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800645
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799079
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799233
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799048
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799050
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797314, upper bound: 141.0797226
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797226, upper bound: 141.0797226
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799189
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799272
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802341, upper bound: 141.0802341
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802341, upper bound: 141.0802341
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795204, upper bound: 141.0795580
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795204, upper bound: 141.0795204
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786708, upper bound: 141.0786711
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786708, upper bound: 141.0786711
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801432
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801216
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800628
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800645
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799079
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799233
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799048
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799050
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0797314, upper bound: 141.0797226
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0797226, upper bound: 141.0797226
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799189
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0799079, upper bound: 141.0799272
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802341, upper bound: 141.0802341
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802341, upper bound: 141.0802341
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0795204, upper bound: 141.0795580
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0795204, upper bound: 141.0795204
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0786708, upper bound: 141.0786711
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0786708, upper bound: 141.0786711
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799706, upper bound: 141.0799706
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799706, upper bound: 141.0799706
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799840, upper bound: 141.0799840
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799840, upper bound: 141.0799840
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801216
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801432
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800542, upper bound: 141.0800542
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800542, upper bound: 141.0800542
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800628
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800563
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0797012
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796859
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0796128
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0795368
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0798838
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799048
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796180, upper bound: 141.0796180
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796180, upper bound: 141.0796345
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793487, upper bound: 141.0793430
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793487, upper bound: 141.0793430
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797156, upper bound: 141.0797156
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797156, upper bound: 141.0797156
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798056, upper bound: 141.0798056
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798056, upper bound: 141.0798166
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797846, upper bound: 141.0797846
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797846, upper bound: 141.0797846
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801138, upper bound: 141.0801138
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801214, upper bound: 141.0801138
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792351
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792327
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791232, upper bound: 141.0791232
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791232, upper bound: 141.0791232
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0715607, upper bound: 141.0715607
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -141.0715607, upper bound: 141.0715607
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0785595, upper bound: 141.0785595
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0785595, upper bound: 141.0785595
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786516
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786501
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802633, upper bound: 141.0802586
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800305
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.77 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.58 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800749, upper bound: 141.0800749
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799706, upper bound: 141.0799706
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799706, upper bound: 141.0799706
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799921, upper bound: 141.0799921
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799840, upper bound: 141.0799840
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799840, upper bound: 141.0799840
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801216
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801216, upper bound: 141.0801432
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800542, upper bound: 141.0800542
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800542, upper bound: 141.0800542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800628
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800495, upper bound: 141.0800563
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0797012
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796859
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0796128
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0795368
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0798838
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798838, upper bound: 141.0799048
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0796180, upper bound: 141.0796180
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0796180, upper bound: 141.0796345
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0793487, upper bound: 141.0793430
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0793487, upper bound: 141.0793430
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797156, upper bound: 141.0797156
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797156, upper bound: 141.0797156
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798056, upper bound: 141.0798056
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798056, upper bound: 141.0798166
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797846, upper bound: 141.0797846
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797846, upper bound: 141.0797846
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801138, upper bound: 141.0801138
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801214, upper bound: 141.0801138
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792351
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0791232, upper bound: 141.0791232
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0791232, upper bound: 141.0791232
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0715607, upper bound: 141.0715607
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0715607, upper bound: 141.0715607
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0785595, upper bound: 141.0785595
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0785595, upper bound: 141.0785595
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786516
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786501
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802633, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800305
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800673, upper bound: 141.0800673
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800673, upper bound: 141.0800673
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798692, upper bound: 141.0798692
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798692, upper bound: 141.0798692
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799656, upper bound: 141.0799656
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799656, upper bound: 141.0799656
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799873, upper bound: 141.0799873
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799873, upper bound: 141.0799873
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795386, upper bound: 141.0795386
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795386, upper bound: 141.0795386
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799660, upper bound: 141.0799660
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799660, upper bound: 141.0799674
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796972
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796859
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800405
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800511
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0796128
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0795368
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0796042
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0795814
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793422, upper bound: 141.0793355
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793355, upper bound: 141.0793355
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792689, upper bound: 141.0792625
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792625, upper bound: 141.0792625
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797081
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797111
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797905
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797822
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801071, upper bound: 141.0801071
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801086, upper bound: 141.0801071
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791349
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791262
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792215
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792327
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790256
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790255
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786516
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786501
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801455, upper bound: 141.0801394
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799550
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799429
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
time: 0.69 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800673, upper bound: 141.0800673
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800673, upper bound: 141.0800673
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798692, upper bound: 141.0798692
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798692, upper bound: 141.0798692
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799656, upper bound: 141.0799656
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799656, upper bound: 141.0799656
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799873, upper bound: 141.0799873
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799873, upper bound: 141.0799873
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795386, upper bound: 141.0795386
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795386, upper bound: 141.0795386
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799660, upper bound: 141.0799660
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799660, upper bound: 141.0799674
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796972
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0796859, upper bound: 141.0796859
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800405
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800511
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0796128
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0795368
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0796042
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0795814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793422, upper bound: 141.0793355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0793355, upper bound: 141.0793355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0792689, upper bound: 141.0792625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0792625, upper bound: 141.0792625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797111
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797905
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797822
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801071, upper bound: 141.0801071
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801086, upper bound: 141.0801071
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791349
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791262
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792215
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790256
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786516
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786501
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801455, upper bound: 141.0801394
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799550
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799429
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.70
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795277, upper bound: 141.0795277
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795277, upper bound: 141.0795277
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799765, upper bound: 141.0799765
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799765, upper bound: 141.0799765
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795184, upper bound: 141.0795184
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795184, upper bound: 141.0795184
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798611, upper bound: 141.0798611
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0798611, upper bound: 141.0798611
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799544, upper bound: 141.0799544
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799544, upper bound: 141.0799544
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795743, upper bound: 141.0795743
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795743, upper bound: 141.0795743
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794456, upper bound: 141.0794456
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794456, upper bound: 141.0794456
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797674, upper bound: 141.0797674
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797674, upper bound: 141.0797674
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794265, upper bound: 141.0794265
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794265, upper bound: 141.0794265
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799791, upper bound: 141.0799791
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799791, upper bound: 141.0799791
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0794972
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0794972
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794184, upper bound: 141.0794184
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794184, upper bound: 141.0794184
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794890, upper bound: 141.0794890
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794890, upper bound: 141.0794890
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795354, upper bound: 141.0795354
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795354, upper bound: 141.0795354
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799574, upper bound: 141.0799574
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0799574, upper bound: 141.0799574
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796753, upper bound: 141.0796796
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796753, upper bound: 141.0796753
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797460, upper bound: 141.0797460
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0797460, upper bound: 141.0797460
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795295, upper bound: 141.0795322
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795295, upper bound: 141.0795437
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796793, upper bound: 141.0796793
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0796793, upper bound: 141.0796793
time: 0.78 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795277, upper bound: 141.0795277
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795277, upper bound: 141.0795277
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799765, upper bound: 141.0799765
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799765, upper bound: 141.0799765
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795184, upper bound: 141.0795184
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795184, upper bound: 141.0795184
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0798611, upper bound: 141.0798611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0798611, upper bound: 141.0798611
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799544, upper bound: 141.0799544
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799544, upper bound: 141.0799544
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795743, upper bound: 141.0795743
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795743, upper bound: 141.0795743
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794456, upper bound: 141.0794456
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794456, upper bound: 141.0794456
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796248, upper bound: 141.0796248
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797745, upper bound: 141.0797745
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797674, upper bound: 141.0797674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797674, upper bound: 141.0797674
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794265, upper bound: 141.0794265
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794265, upper bound: 141.0794265
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799791, upper bound: 141.0799791
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799791, upper bound: 141.0799791
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0794972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794972, upper bound: 141.0794972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796950, upper bound: 141.0796950
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794184, upper bound: 141.0794184
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794184, upper bound: 141.0794184
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794320, upper bound: 141.0794320
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794890, upper bound: 141.0794890
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0794890, upper bound: 141.0794890
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795354, upper bound: 141.0795354
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795354, upper bound: 141.0795354
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799574, upper bound: 141.0799574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0799574, upper bound: 141.0799574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796753, upper bound: 141.0796796
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796753, upper bound: 141.0796753
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797460, upper bound: 141.0797460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0797460, upper bound: 141.0797460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0800420, upper bound: 141.0800420
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0793981, upper bound: 141.0793981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795295, upper bound: 141.0795322
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0795295, upper bound: 141.0795437
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796793, upper bound: 141.0796793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 0, lower bound: -141.0796793, upper bound: 141.0796793
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800405
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800405, upper bound: 141.0800511
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794366
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0794211, upper bound: 141.0794211
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798012, upper bound: 141.0798012
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798921, upper bound: 141.0798921
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0796128
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795368, upper bound: 141.0795368
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0793227, upper bound: 141.0793227
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797970, upper bound: 141.0797970
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795260, upper bound: 141.0795260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0796042
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0795814, upper bound: 141.0795814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0793422, upper bound: 141.0793355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0793355, upper bound: 141.0793355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0792689, upper bound: 141.0792625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0792625, upper bound: 141.0792625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788841, upper bound: 141.0788841
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797033, upper bound: 141.0797033
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797964, upper bound: 141.0797964
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797081
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797081, upper bound: 141.0797111
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797583, upper bound: 141.0797583
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797905
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797822, upper bound: 141.0797822
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801280, upper bound: 141.0801280
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801770, upper bound: 141.0801770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802574, upper bound: 141.0802574
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800213, upper bound: 141.0800213
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801071, upper bound: 141.0801071
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801086, upper bound: 141.0801071
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797361, upper bound: 141.0797361
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799210, upper bound: 141.0799210
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799992, upper bound: 141.0799992
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791349
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0791262, upper bound: 141.0791262
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792215
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0792215, upper bound: 141.0792327
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790256
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0790256, upper bound: 141.0790255
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0788824, upper bound: 141.0788824
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801552
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801552, upper bound: 141.0801888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798790, upper bound: 141.0798790
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0798508, upper bound: 141.0798508
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797513, upper bound: 141.0797513
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0797440, upper bound: 141.0797440
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0770237, upper bound: 141.0770237
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0785466, upper bound: 141.0785466
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786516
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0786501, upper bound: 141.0786501
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0786500, upper bound: 141.0786500
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802599
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801455, upper bound: 141.0801394
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0801393, upper bound: 141.0801393
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802136, upper bound: 141.0802136
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799460, upper bound: 141.0799460
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799550
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0799429, upper bound: 141.0799429
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0802586, upper bound: 141.0802586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.74
Output dim: 0, lower bound: -141.0800240, upper bound: 141.0800240

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.10 + 418.04 = 421.14 seconds
