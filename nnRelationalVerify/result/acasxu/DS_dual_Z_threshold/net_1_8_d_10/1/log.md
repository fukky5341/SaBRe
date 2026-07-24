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
execution time: IAR + RelationalAnalysis = 1.00 + 2.09 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -141.0803599, upper bound: 141.0803599

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.74 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -141.0795734, upper bound: 141.0795734

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795702
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795545
time: 0.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795702
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795545
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795702
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795545
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795702
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -141.0795545, upper bound: 141.0795545

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795417, upper bound: 141.0795523
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795588
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795417, upper bound: 141.0795413
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795413
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795439
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795588
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795413
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795417
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795417, upper bound: 141.0795523
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795588
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795417, upper bound: 141.0795413
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795413
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795439
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795588
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795413
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -141.0795413, upper bound: 141.0795417

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794653
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794671
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794643
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794650
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794649
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794653
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794671
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794643
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794650
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794649
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794647
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794653
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794651
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794671
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
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794643
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794642
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794642
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794650
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794649
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794654
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794654
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
time: 0.77 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794651
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794671
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794642
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794642
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794650
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794649
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794654
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794654
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794809
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.70
Output dim: 0, lower bound: -141.0794641, upper bound: 141.0794641

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792961
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792982
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792945
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792943
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792976
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793013
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792952
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792950
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792949
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792974
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792948
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792969
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792986
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793109
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792986
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793109
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
time: 0.71 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.77 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792961
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792982
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792945
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792943
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792976
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792950
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792974
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792948
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792969
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792986
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793109
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792986
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0793109
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -141.0792915, upper bound: 141.0792915

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792677
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792695
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792662
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792660
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792693
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792759
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792670
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792669
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792889, upper bound: 141.0792608
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792665
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792688
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792664
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792684
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792669, upper bound: 141.0792608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792677, upper bound: 141.0792608
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
time: 0.85 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792677
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792695
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792662
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792660
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792693
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792759
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792670
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792889, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792665
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792688
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792684
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792669, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792677, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792166
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792189
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792159
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792156
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792200
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792243
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -91.7711868, 77.5051651, -91.7711868, 77.5051651, -169.2763367, 169.2763367
1: -348.8091125, 289.5091858, -348.8091125, 289.5091858, -638.3182983, 638.3182983
2: -187.6257019, 293.7341309, -187.6257019, 293.7341309, -481.3598022, 481.3598022
3: -321.2740479, 264.3711853, -321.2740479, 264.3711853, -585.6452637, 585.6452637
4: -236.9158630, 295.7434998, -236.9158630, 295.7434998, -532.6593628, 532.6593018

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
time: 0.73 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792166
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792189
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792159
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792156
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792200
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792243
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.81
Output dim: 0, lower bound: -141.0792078, upper bound: 141.0792078
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792670
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792669
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792889, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792665
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792688
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792684
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792706
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792889
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792669, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792677, upper bound: 141.0792608
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.81
Output dim: 0, lower bound: -141.0792608, upper bound: 141.0792608

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.10 + 417.02 = 420.11 seconds
