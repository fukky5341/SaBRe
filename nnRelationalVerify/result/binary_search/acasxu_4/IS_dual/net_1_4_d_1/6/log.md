## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 2204.5111029827913


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-780.0952148, 1252.1962891, -780.0952148, 1252.1962891, -2032.2915039, 2032.2915039)
1: (-876.9357910, 1279.8708496, -876.9357910, 1279.8708496, -2156.8066406, 2156.8066406)
2: (-884.4795532, 1279.2316895, -884.4795532, 1279.2316895, -2163.7106934, 2163.7106934)
3: (-1073.1099854, 1476.0378418, -1073.1099854, 1476.0378418, -2549.1479492, 2549.1479492)
4: (-971.7084351, 1472.0739746, -971.7084351, 1472.0739746, -2443.7822266, 2443.7824707)

## BASE Result
execution time: IAR + LP analysis = 2.26 + 2.64 = 4.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -2204.6365348, upper bound: 2204.6365348


# Binary Search by BASE starts (time budget: 1195.10 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2549.14794921875
rel_dist={3: [-2204.6289847979, 2204.6289847979006]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2549.14794921875
rel_dist={3: [-2204.604555886248, 2204.604555886248]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2549.14794921875
rel_dist={3: [-2204.582210724847, 2204.5822107248478]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=2549.14794921875
rel_dist={3: [-2204.5694564795213, 2204.5694564795213]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=2549.14794921875
rel_dist={3: [-2204.5628129689744, 2204.562812968975]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=2549.14794921875
rel_dist={3: [-2204.5594493278445, 2204.559449327844]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=2549.14794921875
rel_dist={3: [-2204.5577393459516, 2204.5577393459516]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=2549.14794921875
rel_dist={3: [-2204.556871540629, 2204.55687154063]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=2549.14794921875
rel_dist={3: [-2204.5564302488237, 2204.5564302488237]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=2549.14794921875
rel_dist={3: [-2204.5562075285707, 2204.5562075285707]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=2549.14794921875
rel_dist={3: [-2204.55609592738, 2204.5560959273807]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=2549.14794921875
rel_dist={3: [-2204.5560401268694, 2204.5560401467037]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=2549.14794921875
rel_dist={3: [-2204.5560122267834, 2204.5560122267834]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=2549.14794921875
rel_dist={3: [-2204.555998277074, 2204.5559982770747]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=2549.14794921875
rel_dist={3: [-2204.5559912968474, 2204.555991306161]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=2549.14794921875
rel_dist={3: [-2204.5559878170498, 2204.5559878092868]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=2549.14794921875
rel_dist={3: [-2204.555986068337, 2204.5559860765234]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=2549.14794921875
rel_dist={3: [-2204.555985201443, 2204.555985210228]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=2549.14794921875
rel_dist={3: [-2204.5559848080475, 2204.555984802268]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=2549.14794921875
rel_dist={3: [-2204.555984696635, 2204.5559846294236]}

## Binary Search Result
Binary search time: 100.66 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1094.44 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6175475, upper bound: 2204.5555431
time: 0.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5562721, upper bound: 2204.5562720
time: 1.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -2204.6175475, upper bound: 2204.5555431
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -2204.5562721, upper bound: 2204.5562720

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -780.0952148, 1252.1962891, -2017.9642334, 2007.3833008
1: -860.6093140, 1254.4846191, -876.9357910, 1279.8708496, -2140.4802246, 2131.4201660
2: -868.1311646, 1254.1179199, -884.4795532, 1279.2316895, -2147.3625488, 2138.5971680
3: -1052.7419434, 1447.0034180, -1073.1099854, 1476.0378418, -2528.7797852, 2520.1132812
4: -953.8444824, 1442.9805908, -971.7084351, 1472.0739746, -2425.9177246, 2414.6889648

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6141625, upper bound: 2204.5329061
time: 1.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6173165, upper bound: 2204.5543266
time: 1.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -895.4413452, 1430.4816895, -779.7164917, 1251.5360107, -2146.9772949, 2210.1979980
1: -1006.0681763, 1462.4388428, -876.5065918, 1279.1898193, -2285.2578125, 2338.9453125
2: -1015.1561890, 1461.2092285, -884.0462646, 1278.5438232, -2293.6999512, 2345.2553711
3: -1227.3483887, 1688.1374512, -1072.5823975, 1475.2720947, -2702.6206055, 2760.7197266
4: -1118.4635010, 1682.0313721, -971.2271729, 1471.3106689, -2589.7739258, 2653.2585449

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5335792, upper bound: 2204.5547700
time: 1.08 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5335792, upper bound: 2204.5549838
time: 1.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -2204.6141625, upper bound: 2204.5329061
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -2204.6173165, upper bound: 2204.5543266
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -2204.5335792, upper bound: 2204.5547700
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 3, lower bound: -2204.5335792, upper bound: 2204.5549838

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -765.4657593, 1226.8005371, -746.0270996, 1195.2812500, -1960.7465820, 1972.8273926
1: -860.2699585, 1253.9866943, -838.7745361, 1221.9118652, -2082.1816406, 2092.7609863
2: -867.7856445, 1253.6174316, -845.3336792, 1221.3171387, -2089.1025391, 2098.9504395
3: -1052.3360596, 1446.4301758, -1026.6439209, 1409.3378906, -2461.6735840, 2473.0734863
4: -953.4553833, 1442.4057617, -927.8871460, 1405.6160889, -2359.0712891, 2370.2929688

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6107514, upper bound: 2204.5278435
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
time: 1.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -780.0045166, 1252.0537109, -2017.8215332, 2007.2924805
1: -860.6093140, 1254.4846191, -876.8342896, 1279.7246094, -2140.3334961, 2131.3186035
2: -868.1311646, 1254.1179199, -884.3770752, 1279.0852051, -2147.2163086, 2138.4951172
3: -1052.7419434, 1447.0034180, -1072.9860840, 1475.8693848, -2528.6110840, 2519.9892578
4: -953.8444824, 1442.9805908, -971.5972900, 1471.9053955, -2425.7497559, 2414.5778809

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6138632, upper bound: 2204.5493363
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
time: 0.92 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -855.3549805, 1367.5964355, -779.4248047, 1251.0574951, -2106.4123535, 2147.0212402
1: -961.0993652, 1397.7777100, -876.1824951, 1278.7015381, -2239.8007812, 2273.9599609
2: -969.0943604, 1396.9897461, -883.7112427, 1278.0531006, -2247.1469727, 2280.7004395
3: -1174.3461914, 1613.4357910, -1072.1895752, 1474.7115479, -2649.0576172, 2685.6254883
4: -1065.9420166, 1607.8955078, -970.8472900, 1470.7481689, -2536.6889648, 2578.7426758

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5298035, upper bound: 2204.5500977
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
time: 1.21 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -895.3894043, 1430.3947754, -779.7164917, 1251.5360107, -2146.9248047, 2210.1110840
1: -1006.0085449, 1462.3494873, -876.5065918, 1279.1898193, -2285.1979980, 2338.8557129
2: -1015.0972290, 1461.1177979, -884.0462646, 1278.5438232, -2293.6411133, 2345.1640625
3: -1227.2722168, 1688.0335693, -1072.5823975, 1475.2720947, -2702.5444336, 2760.6159668
4: -1118.3990479, 1681.9255371, -971.2271729, 1471.3106689, -2589.7094727, 2653.1518555

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5511104, upper bound: 2204.5500162
time: 1.25 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
time: 1.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.72 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.6107514, upper bound: 2204.5278435
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.6138632, upper bound: 2204.5493363
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.5298035, upper bound: 2204.5500977
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.5511104, upper bound: 2204.5500162
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.72
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -765.4657593, 1226.8005371, -730.0594482, 1170.0349121, -1935.5003662, 1956.8598633
1: -860.2699585, 1253.9866943, -820.8773804, 1196.0142822, -2056.2841797, 2074.8640137
2: -867.7856445, 1253.6174316, -827.3017578, 1195.5252686, -2063.3107910, 2080.9182129
3: -1052.3360596, 1446.4301758, -1004.7631836, 1379.4556885, -2431.7917480, 2451.1931152
4: -953.4553833, 1442.4057617, -908.2239380, 1375.9803467, -2329.4357910, 2350.6296387

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
time: 1.12 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -762.6414185, 1222.2523193, -928.8084717, 1488.8227539, -2251.4638672, 2151.0607910
1: -857.1156006, 1249.3442383, -1043.7491455, 1522.0285645, -2379.1440430, 2293.0932617
2: -864.5966797, 1248.9622803, -1051.8295898, 1520.4223633, -2385.0183105, 2300.7912598
3: -1048.4257812, 1441.0684814, -1276.7451172, 1755.2833252, -2803.7089844, 2717.8125000
4: -949.9830933, 1437.0302734, -1151.7391357, 1749.9390869, -2699.9221191, 2588.7695312

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -763.5306396, 1225.9390869, -1991.7069092, 1990.8186035
1: -860.6093140, 1254.4846191, -858.3320312, 1252.9741211, -2113.5832520, 2112.8161621
2: -868.1311646, 1254.1179199, -865.7702637, 1252.4456787, -2120.5766602, 2119.8881836
3: -1052.7419434, 1447.0034180, -1050.3448486, 1445.0013428, -2497.7431641, 2497.3481445
4: -953.8444824, 1442.9805908, -951.3379517, 1441.2866211, -2395.1308594, 2394.3183594

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
time: 0.95 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -762.9428101, 1222.7377930, -962.1281128, 1545.0631104, -2308.0051270, 2184.8659668
1: -857.4537354, 1249.8403320, -1081.1672363, 1579.0612793, -2436.5151367, 2331.0068359
2: -864.9411621, 1249.4605713, -1090.0550537, 1577.4730225, -2442.4138184, 2339.5156250
3: -1048.8302002, 1441.6395264, -1322.3431396, 1820.9400635, -2869.7702637, 2763.9819336
4: -950.3709106, 1437.6024170, -1194.2437744, 1815.4454346, -2765.8156738, 2631.8461914

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
time: 1.16 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -855.3549805, 1367.5964355, -762.9709473, 1224.9735107, -2080.3286133, 2130.5666504
1: -961.0993652, 1397.7777100, -857.7024536, 1251.9821777, -2213.0812988, 2255.4799805
2: -969.0943604, 1396.9897461, -865.1264038, 1251.4456787, -2220.5397949, 2262.1162109
3: -1174.3461914, 1613.4357910, -1049.5766602, 1443.8779297, -2618.2241211, 2663.0124512
4: -1065.9420166, 1607.8955078, -950.6104126, 1440.1671143, -2506.1086426, 2558.5058594

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
time: 1.11 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -852.4576416, 1362.8616943, -961.5333252, 1544.0474854, -2396.5051270, 2324.3950195
1: -957.8701172, 1392.9641113, -1080.4964600, 1578.0167236, -2535.8867188, 2473.4599609
2: -965.8312378, 1392.1412354, -1089.3729248, 1576.4199219, -2542.2509766, 2481.5139160
3: -1170.3087158, 1607.8902588, -1321.5236816, 1819.7723389, -2990.0810547, 2929.4133301
4: -1062.4040527, 1602.2945557, -1193.4772949, 1814.2619629, -2876.6655273, 2795.7717285

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
time: 1.26 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -895.3894043, 1430.3947754, -763.2592773, 1225.4467773, -2120.8356934, 2193.6538086
1: -1006.0085449, 1462.3494873, -858.0225220, 1252.4648438, -2258.4733887, 2320.3720703
2: -1015.0972290, 1461.1177979, -865.4575806, 1251.9306641, -2267.0278320, 2326.5749512
3: -1227.2722168, 1688.0335693, -1049.9645996, 1444.4327393, -2671.7050781, 2737.9980469
4: -1118.3990479, 1681.9255371, -950.9865723, 1440.7230225, -2559.1215820, 2632.9116211

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5462269
time: 1.36 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -892.4617310, 1425.5600586, -961.8248291, 1544.5335693, -2436.9953613, 2387.3847656
1: -1002.7413940, 1457.4615479, -1080.8210449, 1578.5106201, -2581.2519531, 2538.2827148
2: -1011.7980347, 1456.1823730, -1089.7071533, 1576.9149170, -2588.7128906, 2545.8891602
3: -1223.1864014, 1682.3981934, -1321.9185791, 1820.3394775, -3043.5253906, 3004.3166504
4: -1114.8439941, 1676.2183838, -1193.8543701, 1814.8297119, -2929.1752930, 2870.0722656

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
time: 1.06 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.55 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6066380, upper bound: 2204.5241031
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.6089398, upper bound: 2204.5455435
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5463066
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5247957, upper bound: 2204.5462269
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 3, lower bound: -2204.5462269, upper bound: 2204.5462269

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -749.0150146, 1200.6947021, -730.0594482, 1170.0349121, -1919.0496826, 1930.7540283
1: -841.8372192, 1227.2375488, -820.8773804, 1196.0142822, -2037.8513184, 2048.1149902
2: -849.1994629, 1226.9832764, -827.3017578, 1195.5252686, -2044.7244873, 2054.2849121
3: -1029.7426758, 1415.5695801, -1004.7631836, 1379.4556885, -2409.1982422, 2420.3327637
4: -933.2001343, 1411.8001709, -908.2239380, 1375.9803467, -2309.1801758, 2320.0241699

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6070938, upper bound: 2204.5258676
time: 1.14 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6103457, upper bound: 2204.5262155
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6080404, upper bound: 2204.5232228
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -947.4668579, 1519.7716064, -730.0594482, 1170.0349121, -2117.5017090, 2249.8305664
1: -1064.4677734, 1553.2272949, -820.8773804, 1196.0142822, -2260.4819336, 2374.1044922
2: -1073.3414307, 1551.9005127, -827.3017578, 1195.5252686, -2268.8664551, 2379.2021484
3: -1301.5438232, 1791.3920898, -1004.7631836, 1379.4556885, -2680.9995117, 2796.1552734
4: -1175.9427490, 1785.8458252, -908.2239380, 1375.9803467, -2551.9230957, 2694.0698242

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6070938, upper bound: 2204.5258676
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6103457, upper bound: 2204.5262155
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6080404, upper bound: 2204.5232228
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -749.0150146, 1200.6947021, -928.8084717, 1488.8227539, -2237.8378906, 2129.5031738
1: -841.8372192, 1227.2375488, -1043.7491455, 1522.0285645, -2363.8657227, 2270.9868164
2: -849.1994629, 1226.9832764, -1051.8295898, 1520.4223633, -2369.6215820, 2278.8122559
3: -1029.7426758, 1415.5695801, -1276.7451172, 1755.2833252, -2785.0258789, 2692.3142090
4: -933.2001343, 1411.8001709, -1151.7391357, 1749.9390869, -2683.1386719, 2563.5393066

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6062417, upper bound: 2204.5221555
time: 1.29 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6055767, upper bound: 2204.5202737
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -947.4668579, 1519.7716064, -928.8084717, 1488.8227539, -2436.2895508, 2448.5795898
1: -1064.4677734, 1553.2272949, -1043.7491455, 1522.0285645, -2586.4960938, 2596.9765625
2: -1073.3414307, 1551.9005127, -1051.8295898, 1520.4223633, -2593.7631836, 2603.7299805
3: -1301.5438232, 1791.3920898, -1276.7451172, 1755.2833252, -3056.8266602, 3068.1367188
4: -1175.9427490, 1785.8458252, -1151.7391357, 1749.9390869, -2925.8818359, 2937.5849609

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6062417, upper bound: 2204.5221555
time: 1.10 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6055767, upper bound: 2204.5202737
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -763.5306396, 1225.9390869, -1975.2553711, 1964.7105713
1: -842.1754150, 1227.7322998, -858.3320312, 1252.9741211, -2095.1494141, 2086.0642090
2: -849.5438843, 1227.4807129, -865.7702637, 1252.4456787, -2101.9892578, 2093.2509766
3: -1030.1467285, 1416.1391602, -1050.3448486, 1445.0013428, -2475.1479492, 2466.4838867
4: -933.5875244, 1412.3710938, -951.3379517, 1441.2866211, -2374.8740234, 2363.7087402

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6108486, upper bound: 2204.5446453
time: 1.26 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6108069, upper bound: 2204.5310043
time: 1.69 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -763.5306396, 1225.9390869, -2173.7060547, 2283.7956543
1: -1064.8012695, 1553.7296143, -858.3320312, 1252.9741211, -2317.7753906, 2412.0612793
2: -1073.6849365, 1552.4042969, -865.7702637, 1252.4456787, -2326.1298828, 2418.1745605
3: -1301.9499512, 1791.9694824, -1050.3448486, 1445.0013428, -2746.9511719, 2842.3144531
4: -1176.3287354, 1786.4251709, -951.3379517, 1441.2866211, -2617.6152344, 2737.7626953

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6108069, upper bound: 2204.5310043
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
time: 2.36 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -962.1281128, 1545.0631104, -2294.3791504, 2163.3081055
1: -842.1754150, 1227.7322998, -1081.1672363, 1579.0612793, -2421.2365723, 2308.8994141
2: -849.5438843, 1227.4807129, -1090.0550537, 1577.4730225, -2427.0168457, 2317.5354004
3: -1030.1467285, 1416.1391602, -1322.3431396, 1820.9400635, -2851.0861816, 2738.4821777
4: -933.5875244, 1412.3710938, -1194.2437744, 1815.4454346, -2749.0329590, 2606.6145020

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6064406, upper bound: 2204.5260728
time: 1.28 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
time: 1.48 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -962.1281128, 1545.0631104, -2492.8295898, 2482.3930664
1: -1064.8012695, 1553.7296143, -1081.1672363, 1579.0612793, -2643.8623047, 2634.8964844
2: -1073.6849365, 1552.4042969, -1090.0550537, 1577.4730225, -2651.1574707, 2642.4594727
3: -1301.9499512, 1791.9694824, -1322.3431396, 1820.9400635, -3122.8898926, 3114.3125000
4: -1176.3287354, 1786.4251709, -1194.2437744, 1815.4454346, -2991.7736816, 2980.6684570

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6064406, upper bound: 2204.5260728
time: 1.64 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
time: 1.27 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -762.9709473, 1224.9735107, -2063.8330078, 2104.3142090
1: -942.6451416, 1370.9106445, -857.7024536, 1251.9821777, -2194.6262207, 2228.6130371
2: -950.4666748, 1370.2813721, -865.1264038, 1251.4456787, -2201.9121094, 2235.4077148
3: -1151.6676025, 1582.3807373, -1049.5766602, 1443.8779297, -2595.5451660, 2631.9572754
4: -1045.6632080, 1577.2132568, -950.6104126, 1440.1671143, -2485.8303223, 2527.8237305

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254884, upper bound: 2204.5312992
time: 1.88 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5256403, upper bound: 2204.5483050
time: 1.86 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -762.9709473, 1224.9735107, -2261.0117188, 2421.7260742
1: -1163.6542969, 1695.2358398, -857.7024536, 1251.9821777, -2415.6359863, 2552.9379883
2: -1173.1274414, 1693.5303955, -865.1264038, 1251.4456787, -2424.5729980, 2558.6567383
3: -1421.6109619, 1955.9685059, -1049.5766602, 1443.8779297, -2865.4887695, 3005.5451660
4: -1286.8170166, 1948.8736572, -950.6104126, 1440.1671143, -2726.9841309, 2899.4836426

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254884, upper bound: 2204.5312992
time: 1.40 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5256403, upper bound: 2204.5483050
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -852.0816650, 1362.2653809, -949.5307007, 1523.9667969, -2376.0483398, 2311.7961426
1: -957.4457397, 1392.3555908, -1066.9807129, 1557.7336426, -2515.1794434, 2459.3361816
2: -965.4024048, 1391.5332031, -1075.6669922, 1556.0726318, -2521.4750977, 2467.2001953
3: -1169.8012695, 1607.1876221, -1304.9840088, 1796.3824463, -2966.1835938, 2912.1713867
4: -1061.9270020, 1601.5933838, -1178.5600586, 1790.8259277, -2852.7526855, 2780.1533203

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
time: 1.12 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
time: 1.15 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -852.4576416, 1362.8616943, -961.2992554, 1543.6788330, -2396.1359863, 2324.1608887
1: -957.8701172, 1392.9641113, -1080.2346191, 1577.6384277, -2535.5083008, 2473.1977539
2: -965.8312378, 1392.1412354, -1089.1076660, 1576.0421143, -2541.8728027, 2481.2487793
3: -1170.3087158, 1607.8902588, -1321.2048340, 1819.3353271, -2989.6440430, 2929.0944824
4: -1062.4040527, 1602.2945557, -1193.1864014, 1813.8272705, -2876.2312012, 2795.4802246

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
time: 1.31 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -763.2592773, 1225.4467773, -2103.8400879, 2166.4516602
1: -986.9750977, 1434.5900879, -858.0225220, 1252.4648438, -2239.4399414, 2292.6125488
2: -995.9152832, 1433.4410400, -865.4575806, 1251.9306641, -2247.8459473, 2298.8979492
3: -1203.8254395, 1656.0020752, -1049.9645996, 1444.4327393, -2648.2583008, 2705.9667969
4: -1097.6263428, 1650.1330566, -950.9865723, 1440.7230225, -2538.3493652, 2601.1196289

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5316614, upper bound: 2204.5484018
time: 1.38 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5494030, upper bound: 2204.5483965
time: 1.09 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -763.2592773, 1225.4467773, -2300.9477539, 2484.4973145
1: -1207.9224854, 1759.2728271, -858.0225220, 1252.4648438, -2460.3872070, 2617.2954102
2: -1218.3870850, 1757.4587402, -865.4575806, 1251.9306641, -2470.3178711, 2622.9160156
3: -1473.9526367, 2029.8526611, -1049.9645996, 1444.4327393, -2918.3852539, 3079.8171387
4: -1338.0777588, 2022.5037842, -950.9865723, 1440.7230225, -2778.8007812, 2973.4902344

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5316614, upper bound: 2204.5484018
time: 1.61 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5494030, upper bound: 2204.5483965
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -961.8248291, 1544.5335693, -2422.9270020, 2365.0173340
1: -986.9750977, 1434.5900879, -1080.8210449, 1578.5106201, -2565.4858398, 2515.4111328
2: -995.9152832, 1433.4410400, -1089.7071533, 1576.9149170, -2572.8298340, 2523.1477051
3: -1203.8254395, 1656.0020752, -1321.9185791, 1820.3394775, -3024.1650391, 2977.9201660
4: -1097.6263428, 1650.1330566, -1193.8543701, 1814.8297119, -2911.8779297, 2843.9873047

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5267148, upper bound: 2204.5445661
time: 1.36 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445582, upper bound: 2204.5445582
time: 1.41 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -961.8248291, 1544.5335693, -2620.0346680, 2683.0634766
1: -1207.9224854, 1759.2728271, -1080.8210449, 1578.5106201, -2786.4331055, 2840.0937500
2: -1218.3870850, 1757.4587402, -1089.7071533, 1576.9149170, -2795.3015137, 2847.1660156
3: -1473.9526367, 2029.8526611, -1321.9185791, 1820.3394775, -3294.2917480, 3351.7705078
4: -1338.0777588, 2022.5037842, -1193.8543701, 1814.8297119, -3151.0290527, 3216.3581543

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5267148, upper bound: 2204.5445661
time: 1.09 seconds

## Relational analysis of IS_A2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5445582, upper bound: 2204.5445582
time: 1.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.19 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6103457, upper bound: 2204.5262155
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6080404, upper bound: 2204.5232228
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6103457, upper bound: 2204.5262155
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6080404, upper bound: 2204.5232228
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6062417, upper bound: 2204.5221555
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6055767, upper bound: 2204.5202737
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6062417, upper bound: 2204.5221555
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6055767, upper bound: 2204.5202737
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6108069, upper bound: 2204.5310043
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6108069, upper bound: 2204.5310043
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6064406, upper bound: 2204.5260728
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6064406, upper bound: 2204.5260728
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5254884, upper bound: 2204.5312992
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5256403, upper bound: 2204.5483050
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5254884, upper bound: 2204.5312992
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5256403, upper bound: 2204.5483050
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5208896, upper bound: 2204.5262344
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5208816, upper bound: 2204.5444492
IS_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5316614, upper bound: 2204.5484018
IS_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5494030, upper bound: 2204.5483965
IS_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5316614, upper bound: 2204.5484018
IS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5494030, upper bound: 2204.5483965
IS_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5267148, upper bound: 2204.5445661
IS_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5445582, upper bound: 2204.5445582
IS_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5267148, upper bound: 2204.5445661
IS_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 3, lower bound: -2204.5445582, upper bound: 2204.5445582

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -748.6428833, 1200.0980225, -720.9932861, 1154.6287842, -1903.2714844, 1921.0910645
1: -841.4171753, 1226.6296387, -810.5822754, 1180.3763428, -2021.7934570, 2037.2119141
2: -848.7762451, 1226.3736572, -817.0637817, 1180.1094971, -2028.8854980, 2043.4375000
3: -1029.2375488, 1414.8686523, -992.1181030, 1361.4935303, -2390.7304688, 2406.9868164
4: -932.7316284, 1411.0986328, -897.1439819, 1357.9986572, -2290.7302246, 2308.2426758

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6105866, upper bound: 2204.5295802
time: 1.22 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -749.0150146, 1200.6947021, -729.7404785, 1169.5179443, -1918.5328369, 1930.4350586
1: -841.8372192, 1227.2375488, -820.5187378, 1195.4871826, -2037.3243408, 2047.7563477
2: -849.1994629, 1226.9832764, -826.9398804, 1194.9981689, -2044.1975098, 2053.9230957
3: -1029.7426758, 1415.5695801, -1004.3226318, 1378.8468018, -2408.5888672, 2419.8920898
4: -933.2001343, 1411.8001709, -907.8271484, 1375.3747559, -2308.5739746, 2319.6274414

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6073708, upper bound: 2204.5202562
time: 1.00 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6032259, upper bound: 2204.5174109
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -947.1087646, 1519.1866455, -720.9932861, 1154.6287842, -2101.7370605, 2240.1799316
1: -1064.0676270, 1552.6326904, -810.5822754, 1180.3763428, -2244.4438477, 2363.2148438
2: -1072.9339600, 1551.3051758, -817.0637817, 1180.1094971, -2253.0429688, 2368.3688965
3: -1301.0545654, 1790.7080078, -992.1181030, 1361.4935303, -2662.5476074, 2782.8261719
4: -1175.4904785, 1785.1596680, -897.1439819, 1357.9986572, -2533.4892578, 2682.3034668

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066207, upper bound: 2204.5244931
time: 1.48 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6102780, upper bound: 2204.5259651
time: 1.33 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6066328, upper bound: 2204.5244055
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -947.4668579, 1519.7716064, -729.7404785, 1169.5179443, -2116.9848633, 2249.5119629
1: -1064.4677734, 1553.2272949, -820.5187378, 1195.4871826, -2259.9548340, 2373.7456055
2: -1073.3414307, 1551.9005127, -826.9398804, 1194.9981689, -2268.3395996, 2378.8403320
3: -1301.5438232, 1791.3920898, -1004.3226318, 1378.8468018, -2680.3901367, 2795.7145996
4: -1175.9427490, 1785.8458252, -907.8271484, 1375.3747559, -2551.3173828, 2693.6728516

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6031333, upper bound: 2204.5151440
time: 1.51 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6076095, upper bound: 2204.5229607
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6059322, upper bound: 2204.5229340
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -748.6428833, 1200.0980225, -919.7720947, 1473.4671631, -2222.1096191, 2119.8698730
1: -841.4171753, 1226.6296387, -1033.4727783, 1506.4949951, -2347.9118652, 2260.1025391
2: -848.7762451, 1226.3736572, -1041.6309814, 1504.9711914, -2353.7475586, 2268.0046387
3: -1029.2375488, 1414.8686523, -1264.1079102, 1737.4920654, -2766.7294922, 2678.9758301
4: -932.7316284, 1411.0986328, -1140.7159424, 1731.9567871, -2664.6884766, 2551.8144531

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6067217, upper bound: 2204.5228176
time: 1.60 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6094698, upper bound: 2204.5269754
time: 1.35 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6084711, upper bound: 2204.5225191
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -749.0150146, 1200.6947021, -928.4917603, 1488.3095703, -2237.3247070, 2129.1865234
1: -841.8372192, 1227.2375488, -1043.3935547, 1521.5067139, -2363.3437500, 2270.6311035
2: -849.1994629, 1226.9832764, -1051.4704590, 1519.8994141, -2369.0988770, 2278.4531250
3: -1029.7426758, 1415.5695801, -1276.3092041, 1754.6802979, -2784.4226074, 2691.8786621
4: -933.2001343, 1411.8001709, -1151.3472900, 1749.3391113, -2682.5388184, 2563.1472168

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6057603, upper bound: 2204.5195815
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6092845, upper bound: 2204.5247660
time: 1.32 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6081190, upper bound: 2204.5206204
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -947.1087646, 1519.1866455, -919.7720947, 1473.4671631, -2420.5754395, 2438.9587402
1: -1064.0676270, 1552.6326904, -1033.4727783, 1506.4949951, -2570.5620117, 2586.1054688
2: -1072.9339600, 1551.3051758, -1041.6309814, 1504.9711914, -2577.9050293, 2592.9360352
3: -1301.0545654, 1790.7080078, -1264.1079102, 1737.4920654, -3038.5466309, 3054.8149414
4: -1175.4904785, 1785.1596680, -1140.7159424, 1731.9567871, -2907.4472656, 2925.8750000

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5731681, upper bound: 2204.5134482
time: 1.85 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6035740, upper bound: 2204.5139385
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -947.4668579, 1519.7716064, -928.4917603, 1488.3095703, -2435.7763672, 2448.2631836
1: -1064.4677734, 1553.2272949, -1043.3935547, 1521.5067139, -2585.9741211, 2596.6208496
2: -1073.3414307, 1551.9005127, -1051.4704590, 1519.8994141, -2593.2407227, 2603.3710938
3: -1301.5438232, 1791.3920898, -1276.3092041, 1754.6802979, -3056.2236328, 3067.7009277
4: -1175.9427490, 1785.8458252, -1151.3472900, 1749.3391113, -2925.2817383, 2937.1926270

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5693484, upper bound: 2204.5118185
time: 1.47 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6029858, upper bound: 2204.5124047
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -748.9434204, 1200.5820312, -751.0092163, 1205.1932373, -1954.1362305, 1951.5911865
1: -841.7546997, 1227.1235352, -844.2378540, 1231.9139404, -2073.6684570, 2071.3608398
2: -849.1198120, 1226.8696289, -851.4539185, 1231.3310547, -2080.4504395, 2078.3232422
3: -1029.6407471, 1415.4370117, -1033.1020508, 1420.7476807, -2450.3884277, 2448.5388184
4: -933.1181641, 1411.6680908, -935.6453247, 1416.9969482, -2350.1152344, 2347.3134766

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6113083, upper bound: 2204.5349617
time: 1.32 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6086050, upper bound: 2204.5322407
time: 1.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -763.2968750, 1225.5712891, -1974.8875732, 1964.4768066
1: -842.1754150, 1227.7322998, -858.0709839, 1252.5972900, -2094.7727051, 2085.8027344
2: -849.5438843, 1227.4807129, -865.5057983, 1252.0693359, -2101.6127930, 2092.9858398
3: -1030.1467285, 1416.1391602, -1050.0274658, 1444.5662842, -2474.7128906, 2466.1665039
4: -933.5875244, 1412.3710938, -951.0480957, 1440.8536377, -2374.4411621, 2363.4189453

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6145272, upper bound: 2204.5483363
time: 1.52 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6120599, upper bound: 2204.5455274
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -947.4082031, 1519.6790771, -751.0092163, 1205.1932373, -2152.6005859, 2270.6882324
1: -1064.4000244, 1553.1339111, -844.2378540, 1231.9139404, -2296.3134766, 2397.3713379
2: -1073.2762451, 1551.8077393, -851.4539185, 1231.3310547, -2304.6074219, 2403.2604980
3: -1301.4593506, 1791.2839355, -1033.1020508, 1420.7476807, -2722.2070312, 2824.3857422
4: -1175.8752441, 1785.7375488, -935.6453247, 1416.9969482, -2592.8720703, 2721.3825684

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6072130, upper bound: 2204.5296851
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6107267, upper bound: 2204.5307852
time: 1.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6067999, upper bound: 2204.5277825
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -763.2968750, 1225.5712891, -2173.3383789, 2283.5615234
1: -1064.8012695, 1553.7296143, -858.0709839, 1252.5972900, -2317.3984375, 2411.8002930
2: -1073.6849365, 1552.4042969, -865.5057983, 1252.0693359, -2325.7534180, 2417.9094238
3: -1301.9499512, 1791.9694824, -1050.0274658, 1444.5662842, -2746.5161133, 2841.9970703
4: -1176.3287354, 1786.4251709, -951.0480957, 1440.8536377, -2617.1823730, 2737.4729004

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
time: 1.25 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -748.9434204, 1200.5820312, -949.9262695, 1524.6152344, -2273.5585938, 2150.5083008
1: -841.7546997, 1227.1235352, -1067.4194336, 1558.3970947, -2400.1518555, 2294.5429688
2: -849.1198120, 1226.8696289, -1076.1213379, 1556.7495117, -2405.8693848, 2302.9907227
3: -1029.6407471, 1415.4370117, -1305.5079346, 1797.1195068, -2826.7602539, 2720.9448242
4: -933.1181641, 1411.6680908, -1179.0668945, 1791.5882568, -2724.7065430, 2590.7346191

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6068949, upper bound: 2204.5281372
time: 1.23 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6095737, upper bound: 2204.5309517
time: 1.15 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6086150, upper bound: 2204.5264471
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -961.9064941, 1544.7144775, -2294.0305176, 2163.0864258
1: -842.1754150, 1227.7322998, -1080.9193115, 1578.7032471, -2420.8786621, 2308.6516113
2: -849.5438843, 1227.4807129, -1089.8040771, 1577.1158447, -2426.6596680, 2317.2846680
3: -1030.1467285, 1416.1391602, -1322.0413818, 1820.5269775, -2850.6733398, 2738.1799316
4: -933.5875244, 1412.3710938, -1193.9682617, 1815.0343018, -2748.6215820, 2606.3391113

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6097908, upper bound: 2204.5441708
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6122054, upper bound: 2204.5489252
time: 1.51 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6122054, upper bound: 2204.5489252
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -947.4082031, 1519.6790771, -949.9262695, 1524.6152344, -2472.0227051, 2469.6054688
1: -1064.4000244, 1553.1339111, -1067.4194336, 1558.3970947, -2622.7971191, 2620.5532227
2: -1073.2762451, 1551.8077393, -1076.1213379, 1556.7495117, -2630.0258789, 2627.9287109
3: -1301.4593506, 1791.2839355, -1305.5079346, 1797.1195068, -3098.5788574, 3096.7919922
4: -1175.8752441, 1785.7375488, -1179.0668945, 1791.5882568, -2967.4633789, 2964.8037109

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5731980, upper bound: 2204.5175735
time: 1.36 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5731980, upper bound: 2204.5179164
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -961.9064941, 1544.7144775, -2492.4807129, 2482.1713867
1: -1064.8012695, 1553.7296143, -1080.9193115, 1578.7032471, -2643.5043945, 2634.6489258
2: -1073.6849365, 1552.4042969, -1089.8040771, 1577.1158447, -2650.8005371, 2642.2084961
3: -1301.9499512, 1791.9694824, -1322.0413818, 1820.5269775, -3122.4768066, 3114.0102539
4: -1176.3287354, 1786.4251709, -1193.9682617, 1815.0343018, -2991.3627930, 2980.3930664

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
time: 1.27 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -838.4881592, 1340.7558594, -750.4823608, 1204.3085938, -2042.7967529, 2091.2377930
1: -942.2261963, 1370.3107910, -843.6499634, 1231.0010986, -2173.2268066, 2213.9606934
2: -950.0430298, 1369.6817627, -850.8475952, 1230.4072266, -2180.4501953, 2220.5288086
3: -1151.1674805, 1581.6879883, -1032.3881836, 1419.7283936, -2570.8955078, 2614.0761719
4: -1045.1920166, 1576.5213623, -934.9654541, 1415.9511719, -2461.1430664, 2511.4863281

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5205745, upper bound: 2204.5298803
time: 1.41 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5173937, upper bound: 2204.5282823
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -762.7373047, 1224.6062012, -2063.4658203, 2104.0808105
1: -942.6451416, 1370.9106445, -857.4414673, 1251.6049805, -2194.2497559, 2228.3520508
2: -950.4666748, 1370.2813721, -864.8619995, 1251.0694580, -2201.5361328, 2235.1433105
3: -1151.6676025, 1582.3807373, -1049.2593994, 1443.4429932, -2595.1101074, 2631.6401367
4: -1045.6632080, 1577.2132568, -950.3207397, 1439.7341309, -2485.3972168, 2527.5339355

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5207500, upper bound: 2204.5468335
time: 1.39 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5178499, upper bound: 2204.5421897
time: 1.21 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1035.6640625, 1658.1582031, -750.4823608, 1204.3085938, -2239.9726562, 2408.6401367
1: -1163.2320557, 1694.6240234, -843.6499634, 1231.0010986, -2394.2331543, 2538.2739258
2: -1172.7014160, 1692.9180908, -850.8475952, 1230.4072266, -2403.1083984, 2543.7656250
3: -1421.1052246, 1955.2650146, -1032.3881836, 1419.7283936, -2840.8334961, 2987.6533203
4: -1286.3428955, 1948.1697998, -934.9654541, 1415.9511719, -2702.2939453, 2883.1342773

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5190887, upper bound: 2204.5297974
time: 1.28 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5253829, upper bound: 2204.5312033
time: 1.76 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5215032, upper bound: 2204.5281973
time: 1.32 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -762.7373047, 1224.6062012, -2260.6445312, 2421.4929199
1: -1163.6542969, 1695.2358398, -857.4414673, 1251.6049805, -2415.2592773, 2552.6772461
2: -1173.1274414, 1693.5303955, -864.8619995, 1251.0694580, -2424.1967773, 2558.3918457
3: -1421.6109619, 1955.9685059, -1049.2593994, 1443.4429932, -2865.0537109, 3005.2277832
4: -1286.8170166, 1948.8736572, -950.3207397, 1439.7341309, -2726.5510254, 2899.1938477

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5194587, upper bound: 2204.5435114
time: 1.36 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5255312, upper bound: 2204.5482444
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5214838, upper bound: 2204.5481697
time: 1.27 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -838.4881592, 1340.7558594, -949.5307007, 1523.9667969, -2362.4550781, 2290.2863770
1: -942.2261963, 1370.3107910, -1066.9807129, 1557.7336426, -2499.9599609, 2437.2915039
2: -950.0430298, 1369.6817627, -1075.6669922, 1556.0726318, -2506.1157227, 2445.3486328
3: -1151.1674805, 1581.6879883, -1304.9840088, 1796.3824463, -2947.5493164, 2886.6718750
4: -1045.1920166, 1576.5213623, -1178.5600586, 1790.8259277, -2836.0180664, 2755.0812988

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130267, upper bound: 2204.5230258
time: 1.10 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130276, upper bound: 2204.5183239
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1035.6640625, 1658.1582031, -949.5307007, 1523.9667969, -2559.6308594, 2607.6884766
1: -1163.2320557, 1694.6240234, -1066.9807129, 1557.7336426, -2720.9658203, 2761.6047363
2: -1172.7014160, 1692.9180908, -1075.6669922, 1556.0726318, -2728.7736816, 2768.5849609
3: -1421.1052246, 1955.2650146, -1304.9840088, 1796.3824463, -3217.4873047, 3260.2490234
4: -1286.3428955, 1948.1697998, -1178.5600586, 1790.8259277, -3076.3398438, 3126.7292480

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130267, upper bound: 2204.5230258
time: 1.28 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130276, upper bound: 2204.5183239
time: 1.54 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -961.2992554, 1543.6788330, -2382.5380859, 2302.6423340
1: -942.6451416, 1370.9106445, -1080.2346191, 1577.6384277, -2520.2834473, 2451.1452637
2: -950.4666748, 1370.2813721, -1089.1076660, 1576.0421143, -2526.5080566, 2459.3891602
3: -1151.6676025, 1582.3807373, -1321.2048340, 1819.3353271, -2971.0029297, 2903.5854492
4: -1045.6632080, 1577.2132568, -1193.1864014, 1813.8272705, -2859.4904785, 2770.3989258

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130120, upper bound: 2204.5386723
time: 1.10 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130163, upper bound: 2204.5403134
time: 1.23 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -961.2992554, 1543.6788330, -2579.7165527, 2620.0549316
1: -1163.6542969, 1695.2358398, -1080.2346191, 1577.6384277, -2741.2927246, 2775.4697266
2: -1173.1274414, 1693.5303955, -1089.1076660, 1576.0421143, -2749.1689453, 2782.6379395
3: -1421.6109619, 1955.9685059, -1321.2048340, 1819.3353271, -3240.9462891, 3277.1733398
4: -1286.8170166, 1948.8736572, -1193.1864014, 1813.8272705, -3099.6774902, 3142.0588379

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130120, upper bound: 2204.5386723
time: 1.17 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5130163, upper bound: 2204.5403134
time: 1.22 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -869.1335449, 1388.8323975, -762.9126587, 1224.8819580, -2094.0156250, 2151.7446289
1: -976.6564941, 1419.9968262, -857.6343994, 1251.8887939, -2228.5451660, 2277.6313477
2: -985.3414307, 1418.8310547, -865.0612793, 1251.3529053, -2236.6943359, 2283.8923340
3: -1191.5489502, 1639.1267090, -1049.4923096, 1443.7698975, -2635.3186035, 2688.6188965
4: -1086.0450439, 1633.1595459, -950.5423584, 1440.0598145, -2526.1042480, 2583.7016602

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5354543, upper bound: 2204.5498179
time: 1.43 seconds

## Relational analysis of IS_A2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5326644, upper bound: 2204.5455147
time: 1.27 seconds

## BFS IS instance: IS_A2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -878.1973267, 1402.8771973, -763.2592773, 1225.4467773, -2103.6440430, 2166.1364746
1: -986.7549438, 1434.2666016, -858.0225220, 1252.4648438, -2239.2197266, 2292.2890625
2: -995.6931763, 1433.1166992, -865.4575806, 1251.9306641, -2247.6237793, 2298.5739746
3: -1203.5535889, 1655.6300049, -1049.9645996, 1444.4327393, -2647.9863281, 2705.5947266
4: -1097.3848877, 1649.7606201, -950.9865723, 1440.7230225, -2538.1079102, 2600.7470703

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5488217, upper bound: 2204.5502061
time: 1.12 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5459639, upper bound: 2204.5459639
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1066.4095459, 1706.7022705, -762.9126587, 1224.8819580, -2291.2915039, 2469.6145020
1: -1197.7752686, 1744.5541992, -857.6343994, 1251.8887939, -2449.6635742, 2602.1884766
2: -1207.9581299, 1742.6806641, -865.0612793, 1251.3529053, -2459.3110352, 2607.7419434
3: -1461.8750000, 2012.9472656, -1049.4923096, 1443.7698975, -2905.6450195, 3062.4394531
4: -1326.6899414, 2005.4429932, -950.5423584, 1440.0598145, -2766.7487793, 2955.9851074

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5310525, upper bound: 2204.5484018
time: 1.49 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5282766, upper bound: 2204.5437516
time: 1.18 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5315935, upper bound: 2204.5483717
time: 1.18 seconds

## Relational analysis of IS_A2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5272051, upper bound: 2204.5483132
time: 1.64 seconds

## BFS IS instance: IS_A2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1075.2875977, 1720.9014893, -763.2592773, 1225.4467773, -2300.7343750, 2484.1601562
1: -1207.6826172, 1758.9262695, -858.0225220, 1252.4648438, -2460.1474609, 2616.9487305
2: -1218.1456299, 1757.1115723, -865.4575806, 1251.9306641, -2470.0761719, 2622.5688477
3: -1473.6567383, 2029.4526367, -1049.9645996, 1444.4327393, -2918.0893555, 3079.4172363
4: -1337.8161621, 2022.1049805, -950.9865723, 1440.7230225, -2778.5390625, 2973.0913086

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5487235, upper bound: 2204.5483965
time: 1.32 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5487235, upper bound: 2204.5483965
time: 1.57 seconds

## BFS IS instance: IS_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -869.1335449, 1388.8323975, -961.4785156, 1543.9632568, -2413.0966797, 2350.3110352
1: -976.6564941, 1419.9968262, -1080.4333496, 1577.9312744, -2554.5876465, 2500.4301758
2: -985.3414307, 1418.8310547, -1089.3114014, 1576.3348389, -2561.6757812, 2508.1425781
3: -1191.5489502, 1639.1267090, -1321.4456787, 1819.6732178, -3011.2219238, 2960.5722656
4: -1086.0450439, 1633.1595459, -1193.4128418, 1814.1629639, -2899.8537598, 2826.5720215

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5303000, upper bound: 2204.5436623
time: 1.03 seconds

## Relational analysis of IS_A2_A2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5308136, upper bound: 2204.5492193
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5315566, upper bound: 2204.5491108
time: 1.15 seconds

## Relational analysis of IS_A2_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5286149, upper bound: 2204.5451582
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -878.1973267, 1402.8771973, -961.8248291, 1544.5335693, -2422.7307129, 2364.7021484
1: -986.7549438, 1434.2666016, -1080.8210449, 1578.5106201, -2565.2656250, 2515.0876465
2: -995.6931763, 1433.1166992, -1089.7071533, 1576.9149170, -2572.6081543, 2522.8237305
3: -1203.5535889, 1655.6300049, -1321.9185791, 1820.3394775, -3023.8928223, 2977.5480957
4: -1097.3848877, 1649.7606201, -1193.8543701, 1814.8297119, -2911.6354980, 2843.6149902

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5437464, upper bound: 2204.5440989
time: 1.12 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5475730, upper bound: 2204.5494030
time: 1.41 seconds

## Relational analysis of IS_A2_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5475730, upper bound: 2204.5494030
time: 1.13 seconds

## BFS IS instance: IS_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -1066.4095459, 1706.7022705, -961.4785156, 1543.9632568, -2610.3728027, 2668.1806641
1: -1197.7752686, 1744.5541992, -1080.4333496, 1577.9312744, -2775.7062988, 2824.9875488
2: -1207.9581299, 1742.6806641, -1089.3114014, 1576.3348389, -2784.2922363, 2831.9921875
3: -1461.8750000, 2012.9472656, -1321.4456787, 1819.6732178, -3281.5483398, 3334.3930664
4: -1326.6899414, 2005.4429932, -1193.4128418, 1814.1629639, -3139.1506348, 3198.8554688

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5259863, upper bound: 2204.5445661
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5242728, upper bound: 2204.5405406
time: 1.21 seconds

## Relational analysis of IS_A2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5184762, upper bound: 2204.5405514
time: 10.38 seconds

## BFS IS instance: IS_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -1075.2875977, 1720.9014893, -961.8248291, 1544.5335693, -2619.8212891, 2682.7263184
1: -1207.6826172, 1758.9262695, -1080.8210449, 1578.5106201, -2786.1933594, 2839.7473145
2: -1218.1456299, 1757.1115723, -1089.7071533, 1576.9149170, -2795.0600586, 2846.8188477
3: -1473.6567383, 2029.4526367, -1321.9185791, 1820.3394775, -3293.9960938, 3351.3706055
4: -1337.8161621, 2022.1049805, -1193.8543701, 1814.8297119, -3150.7653809, 3215.9587402

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5437387, upper bound: 2204.5445582
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5437387, upper bound: 2204.5445582
time: 1.23 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.94 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6105866, upper bound: 2204.5295802
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6073708, upper bound: 2204.5202562
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6032259, upper bound: 2204.5174109
IS_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6102780, upper bound: 2204.5259651
IS_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6066328, upper bound: 2204.5244055
IS_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6076095, upper bound: 2204.5229607
IS_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6059322, upper bound: 2204.5229340
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6094698, upper bound: 2204.5269754
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6084711, upper bound: 2204.5225191
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6092845, upper bound: 2204.5247660
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6081190, upper bound: 2204.5206204
IS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5731681, upper bound: 2204.5134482
IS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6035740, upper bound: 2204.5139385
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5693484, upper bound: 2204.5118185
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6029858, upper bound: 2204.5124047
IS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6113083, upper bound: 2204.5349617
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6086050, upper bound: 2204.5322407
IS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6145272, upper bound: 2204.5483363
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6120599, upper bound: 2204.5455274
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6107267, upper bound: 2204.5307852
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6067999, upper bound: 2204.5277825
IS_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
IS_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6136072, upper bound: 2204.5477986
IS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6095737, upper bound: 2204.5309517
IS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6086150, upper bound: 2204.5264471
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6122054, upper bound: 2204.5489252
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6122054, upper bound: 2204.5489252
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5731980, upper bound: 2204.5175735
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5731980, upper bound: 2204.5179164
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.6087042, upper bound: 2204.5439735
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5205745, upper bound: 2204.5298803
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5173937, upper bound: 2204.5282823
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5207500, upper bound: 2204.5468335
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5178499, upper bound: 2204.5421897
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5253829, upper bound: 2204.5312033
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5215032, upper bound: 2204.5281973
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5255312, upper bound: 2204.5482444
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5214838, upper bound: 2204.5481697
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130267, upper bound: 2204.5230258
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130276, upper bound: 2204.5183239
IS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130267, upper bound: 2204.5230258
IS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130276, upper bound: 2204.5183239
IS_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130120, upper bound: 2204.5386723
IS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130163, upper bound: 2204.5403134
IS_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130120, upper bound: 2204.5386723
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5130163, upper bound: 2204.5403134
IS_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5354543, upper bound: 2204.5498179
IS_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5326644, upper bound: 2204.5455147
IS_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5488217, upper bound: 2204.5502061
IS_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5459639, upper bound: 2204.5459639
IS_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5315935, upper bound: 2204.5483717
IS_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5272051, upper bound: 2204.5483132
IS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5487235, upper bound: 2204.5483965
IS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5487235, upper bound: 2204.5483965
IS_A2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5315566, upper bound: 2204.5491108
IS_A2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5286149, upper bound: 2204.5451582
IS_A2_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5475730, upper bound: 2204.5494030
IS_A2_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5475730, upper bound: 2204.5494030
IS_A2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5242728, upper bound: 2204.5405406
IS_A2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5184762, upper bound: 2204.5405514
IS_A2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5437387, upper bound: 2204.5445582
IS_A2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.94
Output dim: 3, lower bound: -2204.5437387, upper bound: 2204.5445582

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -711.4460449, 1139.1976318, -720.9932861, 1154.6287842, -1866.0748291, 1860.1907959
1: -799.4622192, 1164.4405518, -810.5822754, 1180.3763428, -1979.8385010, 1975.0228271
2: -806.5594482, 1164.2080078, -817.0637817, 1180.1094971, -1986.6689453, 1981.2717285
3: -977.3726196, 1342.8676758, -992.1181030, 1361.4935303, -2338.8657227, 2334.9858398
4: -885.9243164, 1339.3626709, -897.1439819, 1357.9986572, -2243.9228516, 2236.5065918

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
time: 1.07 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1026.9084473, 1656.6524658, -719.2562866, 1151.8925781, -2178.8002930, 2375.9084473
1: -1155.5075684, 1693.9608154, -808.6398315, 1177.5775146, -2333.0847168, 2502.6005859
2: -1164.3540039, 1692.1992188, -815.0991821, 1177.3046875, -2341.6584473, 2507.2983398
3: -1415.9752197, 1952.0909424, -989.7462158, 1358.2630615, -2774.2377930, 2941.8371582
4: -1274.3468018, 1948.6750488, -894.9945679, 1354.7622070, -2629.1088867, 2843.1633301

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
time: 1.21 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6077972, upper bound: 2204.5268209
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -711.8558960, 1139.8516846, -729.7404785, 1169.5179443, -1881.3737793, 1869.5919189
1: -799.9259644, 1165.1103516, -820.5187378, 1195.4871826, -1995.4130859, 1985.6290283
2: -807.0263062, 1164.8792725, -826.9398804, 1194.9981689, -2002.0241699, 1991.8190918
3: -977.9290161, 1343.6379395, -1004.3226318, 1378.8468018, -2356.7753906, 2347.9604492
4: -886.4426270, 1340.1345215, -907.8271484, 1375.3747559, -2261.8166504, 2247.9611816

Time for backsubstitution: 2.36 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=2549.14794921875
rel_dist={3: [-2204.6289847979, 2204.6289847979006]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5816773, upper bound: 2204.5447071
time: 1.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5456816, upper bound: 2204.5456816
time: 1.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 3, lower bound: -2204.5816773, upper bound: 2204.5447071
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 3, lower bound: -2204.5456816, upper bound: 2204.5456816

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -780.0952148, 1252.1962891, -2017.9642334, 2007.3833008
1: -860.6093140, 1254.4846191, -876.9357910, 1279.8708496, -2140.4802246, 2131.4201660
2: -868.1311646, 1254.1179199, -884.4795532, 1279.2316895, -2147.3625488, 2138.5971680
3: -1052.7419434, 1447.0034180, -1073.1099854, 1476.0378418, -2528.7797852, 2520.1132812
4: -953.8444824, 1442.9805908, -971.7084351, 1472.0739746, -2425.9177246, 2414.6889648

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5776096, upper bound: 2204.5227884
time: 1.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
time: 1.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -895.4413452, 1430.4816895, -779.0708618, 1250.4332275, -2145.8742676, 2209.5524902
1: -1006.0681763, 1462.4388428, -875.7714233, 1278.0535889, -2284.1215820, 2338.2102051
2: -1015.1561890, 1461.2092285, -883.3092041, 1277.3872070, -2292.5434570, 2344.5185547
3: -1227.3483887, 1688.1374512, -1071.6793213, 1474.0098877, -2701.3583984, 2759.8168945
4: -1118.4635010, 1682.0313721, -970.4119263, 1470.0190430, -2588.4821777, 2652.4433594

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5233288, upper bound: 2204.5443998
time: 1.35 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5443349, upper bound: 2204.5443349
time: 1.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.13
Output dim: 3, lower bound: -2204.5776096, upper bound: 2204.5227884
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.13
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.13
Output dim: 3, lower bound: -2204.5233288, upper bound: 2204.5443998
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.13
Output dim: 3, lower bound: -2204.5443349, upper bound: 2204.5443349

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -764.8743896, 1225.8463135, -746.0270996, 1195.2812500, -1960.1556396, 1971.8731689
1: -859.6060791, 1253.0124512, -838.7745361, 1221.9118652, -2081.5175781, 2091.7866211
2: -867.1098022, 1252.6383057, -845.3336792, 1221.3171387, -2088.4265137, 2097.9719238
3: -1051.5416260, 1445.3087158, -1026.6439209, 1409.3378906, -2460.8793945, 2471.9523926
4: -952.6947632, 1441.2817383, -927.8871460, 1405.6160889, -2358.3098145, 2369.1689453

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5740589, upper bound: 2204.5207672
time: 1.26 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
time: 1.12 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -780.0045166, 1252.0537109, -2017.8215332, 2007.2924805
1: -860.6093140, 1254.4846191, -876.8342896, 1279.7246094, -2140.3334961, 2131.3186035
2: -868.1311646, 1254.1179199, -884.3770752, 1279.0852051, -2147.2163086, 2138.4951172
3: -1052.7419434, 1447.0034180, -1072.9860840, 1475.8693848, -2528.6110840, 2519.9892578
4: -953.8444824, 1442.9805908, -971.5972900, 1471.9053955, -2425.7497559, 2414.5778809

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
time: 1.13 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
time: 1.47 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -855.3549805, 1367.5964355, -778.2109985, 1249.0238037, -2104.3789062, 2145.8073730
1: -961.0993652, 1397.7777100, -874.8151855, 1276.6143799, -2237.7138672, 2272.5927734
2: -969.0943604, 1396.9897461, -882.3214111, 1275.9401855, -2245.0346680, 2279.3107910
3: -1174.3461914, 1613.4357910, -1070.5200195, 1472.3582764, -2646.7045898, 2683.9558105
4: -1065.9420166, 1607.8955078, -969.2907104, 1468.3601074, -2534.3015137, 2577.1860352

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5198139, upper bound: 2204.5430670
time: 1.19 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
time: 1.46 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -895.3894043, 1430.3947754, -779.0708618, 1250.4332275, -2145.8217773, 2209.4653320
1: -1006.0085449, 1462.3494873, -875.7714233, 1278.0535889, -2284.0620117, 2338.1208496
2: -1015.0972290, 1461.1177979, -883.3092041, 1277.3872070, -2292.4843750, 2344.4270020
3: -1227.2722168, 1688.0335693, -1071.6793213, 1474.0098877, -2701.2822266, 2759.7128906
4: -1118.3990479, 1681.9255371, -970.4119263, 1470.0190430, -2588.4177246, 2652.3371582

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5434843, upper bound: 2204.5443349
time: 0.95 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5434843, upper bound: 2204.5443349
time: 1.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.63 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5740589, upper bound: 2204.5207672
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5812235, upper bound: 2204.5434868
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5198139, upper bound: 2204.5430670
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5434843, upper bound: 2204.5443349
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.63
Output dim: 3, lower bound: -2204.5434843, upper bound: 2204.5443349

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -761.3682251, 1220.2545166, -730.0594482, 1170.0349121, -1931.4030762, 1950.3139648
1: -855.6706543, 1247.2801514, -820.8773804, 1196.0142822, -2051.6845703, 2068.1569824
2: -863.1489258, 1246.9315186, -827.3017578, 1195.5252686, -2058.6735840, 2074.2326660
3: -1046.7094727, 1438.7098389, -1004.7631836, 1379.4556885, -2426.1650391, 2443.4731445
4: -948.3781738, 1434.7274170, -908.2239380, 1375.9803467, -2324.3583984, 2342.9514160

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
time: 1.16 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -755.7206421, 1211.0975342, -928.8084717, 1488.8227539, -2244.5429688, 2139.9057617
1: -849.3907471, 1237.9671631, -1043.7491455, 1522.0285645, -2371.4194336, 2281.7163086
2: -856.7766113, 1237.5461426, -1051.8295898, 1520.4223633, -2377.1989746, 2289.3754883
3: -1038.8898926, 1427.9293213, -1276.7451172, 1755.2833252, -2794.1733398, 2704.6740723
4: -941.4500732, 1423.8538818, -1151.7391357, 1749.9390869, -2691.3889160, 2575.5930176

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
time: 1.34 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -765.6796265, 1227.1473389, -1992.9151611, 1992.9674072
1: -860.6093140, 1254.4846191, -860.5095825, 1254.3399658, -2114.9492188, 2114.9941406
2: -868.1311646, 1254.1179199, -868.0309448, 1253.9733887, -2122.1042480, 2122.1489258
3: -1052.7419434, 1447.0034180, -1052.6195068, 1446.8366699, -2499.5786133, 2499.6228027
4: -953.8444824, 1442.9805908, -953.7347412, 1442.8133545, -2396.6577148, 2396.7153320

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5792672, upper bound: 2204.5389132
time: 1.42 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -895.3894043, 1430.3947754, -2196.1625977, 2122.6765137
1: -860.6093140, 1254.4846191, -1006.0085449, 1462.3494873, -2322.9584961, 2260.4926758
2: -868.1311646, 1254.1179199, -1015.0972290, 1461.1177979, -2329.2490234, 2269.2150879
3: -1052.7419434, 1447.0034180, -1227.2722168, 1688.0335693, -2740.7753906, 2674.2756348
4: -953.8444824, 1442.9805908, -1118.3990479, 1681.9255371, -2635.7690430, 2561.3793945

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5792672, upper bound: 2204.5389132
time: 1.20 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -851.8142700, 1361.9376221, -761.7991943, 1223.0042725, -2074.8186035, 2123.7363281
1: -957.1284180, 1391.9809570, -856.3837280, 1249.9602051, -2207.0881348, 2248.3647461
2: -965.0946045, 1391.2260742, -863.7846680, 1249.4012451, -2214.4953613, 2255.0107422
3: -1169.4576416, 1606.7481689, -1047.9674072, 1441.5981445, -2611.0556641, 2654.7155762
4: -1061.5828857, 1601.2783203, -949.1043091, 1437.8586426, -2499.4411621, 2550.3825684

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
time: 1.35 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
time: 1.46 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -846.0663452, 1352.4747314, -960.3131714, 1541.9914551, -2388.0578613, 2312.7878418
1: -950.7581787, 1382.4013672, -1079.1201172, 1575.9105225, -2526.6682129, 2461.5209961
2: -958.6314697, 1381.5292969, -1087.9730225, 1574.2895508, -2532.9208984, 2469.5021973
3: -1161.4564209, 1595.6894531, -1319.8493652, 1817.4102783, -2978.8664551, 2915.5385742
4: -1054.5889893, 1590.0411377, -1191.9132080, 1811.8625488, -2866.4509277, 2781.9543457

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
time: 1.02 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
time: 1.28 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -895.3894043, 1430.3947754, -765.7679443, 1227.2880859, -2122.6767578, 2196.1625977
1: -1006.0085449, 1462.3494873, -860.6093140, 1254.4846191, -2260.4926758, 2322.9584961
2: -1015.0972290, 1461.1177979, -868.1311646, 1254.1179199, -2269.2150879, 2329.2490234
3: -1227.2722168, 1688.0335693, -1052.7419434, 1447.0034180, -2674.2756348, 2740.7753906
4: -1118.3990479, 1681.9255371, -953.8444824, 1442.9805908, -2561.3793945, 2635.7690430

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5388711, upper bound: 2204.5429844
time: 1.44 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.18 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -895.3894043, 1430.3947754, -895.4413452, 1430.4816895, -2325.8701172, 2325.8361816
1: -1006.0085449, 1462.3494873, -1006.0681763, 1462.4388428, -2468.4472656, 2468.4174805
2: -1015.0972290, 1461.1177979, -1015.1561890, 1461.2092285, -2476.3063965, 2476.2739258
3: -1227.2722168, 1688.0335693, -1227.3483887, 1688.1374512, -2915.4096680, 2915.3818359
4: -1118.3990479, 1681.9255371, -1118.4635010, 1682.0313721, -2800.4304199, 2800.3881836

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5388711, upper bound: 2204.5429844
time: 1.32 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.05 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.28 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5732061, upper bound: 2204.5177671
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5792672, upper bound: 2204.5389132
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5792672, upper bound: 2204.5389132
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5186293, upper bound: 2204.5389292
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5388711, upper bound: 2204.5429844
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5388711, upper bound: 2204.5429844
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.28
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -748.4260864, 1199.7456055, -730.0594482, 1170.0349121, -1918.4606934, 1929.8050537
1: -841.1757812, 1226.2701416, -820.8773804, 1196.0142822, -2037.1900635, 2047.1474609
2: -848.5261841, 1226.0109863, -827.3017578, 1195.5252686, -2044.0515137, 2053.3125000
3: -1028.9526367, 1414.4559326, -1004.7631836, 1379.4556885, -2408.4082031, 2419.2189941
4: -932.4427490, 1410.6839600, -908.2239380, 1375.9803467, -2308.4230957, 2318.9077148

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5734458, upper bound: 2204.5190710
time: 1.40 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5704005, upper bound: 2204.5171000
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -946.8806152, 1518.8071289, -730.0594482, 1170.0349121, -2116.9155273, 2248.8659668
1: -1063.8168945, 1552.2452393, -820.8773804, 1196.0142822, -2259.8310547, 2373.1225586
2: -1072.6711426, 1550.9163818, -827.3017578, 1195.5252686, -2268.1962891, 2378.2182617
3: -1300.7518311, 1790.2640381, -1004.7631836, 1379.4556885, -2680.2072754, 2795.0270996
4: -1175.1893311, 1784.7138672, -908.2239380, 1375.9803467, -2551.1696777, 2692.9377441

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5734304, upper bound: 2204.5203931
time: 1.22 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5730960, upper bound: 2204.5199608
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -748.4260864, 1199.7456055, -928.8084717, 1488.8227539, -2237.2487793, 2128.5541992
1: -841.1757812, 1226.2701416, -1043.7491455, 1522.0285645, -2363.2043457, 2270.0190430
2: -848.5261841, 1226.0109863, -1051.8295898, 1520.4223633, -2368.9479980, 2277.8400879
3: -1028.9526367, 1414.4559326, -1276.7451172, 1755.2833252, -2784.2358398, 2691.2006836
4: -932.4427490, 1410.6839600, -1151.7391357, 1749.9390869, -2682.3813477, 2562.4230957

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5726068, upper bound: 2204.5158658
time: 1.34 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5698180, upper bound: 2204.5138791
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -946.8806152, 1518.8071289, -928.8084717, 1488.8227539, -2435.7033691, 2447.6149902
1: -1063.8168945, 1552.2452393, -1043.7491455, 1522.0285645, -2585.8454590, 2595.9943848
2: -1072.6711426, 1550.9163818, -1051.8295898, 1520.4223633, -2593.0932617, 2602.7460938
3: -1300.7518311, 1790.2640381, -1276.7451172, 1755.2833252, -3056.0346680, 3067.0085449
4: -1175.1893311, 1784.7138672, -1151.7391357, 1749.9390869, -2925.1279297, 2936.4531250

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5726068, upper bound: 2204.5158658
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5698180, upper bound: 2204.5138791
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -762.1703491, 1221.5499268, -1970.8660889, 1963.3502197
1: -842.1754150, 1227.7322998, -856.5707397, 1248.6022949, -2090.7778320, 2084.3029785
2: -849.5438843, 1227.4807129, -864.0667725, 1248.2612305, -2097.8049316, 2091.5473633
3: -1030.1467285, 1416.1391602, -1047.7828369, 1440.2315674, -2470.3781738, 2463.9218750
4: -933.5875244, 1412.3710938, -949.4144897, 1436.2528076, -2369.8403320, 2361.7851562

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
time: 1.35 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -756.5185547, 1212.3859863, -2160.1530762, 2276.7834473
1: -1064.8012695, 1553.7296143, -850.2864380, 1239.2802734, -2304.0815430, 2404.0158691
2: -1073.6849365, 1552.4042969, -857.6898193, 1238.8663330, -2312.5507812, 2410.0942383
3: -1301.9499512, 1791.9694824, -1039.9569092, 1429.4387207, -2731.3881836, 2831.9260254
4: -1176.3287354, 1786.4251709, -942.4810791, 1425.3680420, -2601.6967773, 2728.9062500

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
time: 1.26 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -891.7757568, 1424.5837402, -2173.8999023, 2092.9555664
1: -842.1754150, 1227.7322998, -1001.9531250, 1456.4133301, -2298.5888672, 2229.6850586
2: -849.5438843, 1227.4807129, -1011.0187988, 1455.1990967, -2304.7426758, 2238.4990234
3: -1030.1467285, 1416.1391602, -1222.2701416, 1681.2009277, -2711.3476562, 2638.4091797
4: -933.5875244, 1412.3710938, -1113.9793701, 1675.1312256, -2608.7182617, 2526.3505859

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.36 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -885.9844971, 1414.9328613, -2362.6997070, 2406.2492676
1: -1064.8012695, 1553.7296143, -995.5231323, 1446.7009277, -2511.5021973, 2549.2526855
2: -1073.6849365, 1552.4042969, -1004.4965210, 1445.3520508, -2519.0366211, 2556.9008789
3: -1301.9499512, 1791.9694824, -1214.2031250, 1669.9685059, -2971.9182129, 3006.1726074
4: -1176.3287354, 1786.4251709, -1106.9653320, 1663.7016602, -2840.0302734, 2892.8186035

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.36 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
time: 1.24 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -761.7991943, 1223.0042725, -2061.8637695, 2103.1425781
1: -942.6451416, 1370.9106445, -856.3837280, 1249.9602051, -2192.6047363, 2227.2944336
2: -950.4666748, 1370.2813721, -863.7846680, 1249.4012451, -2199.8671875, 2234.0659180
3: -1151.6676025, 1582.3807373, -1047.9674072, 1441.5981445, -2593.2653809, 2630.3481445
4: -1045.6632080, 1577.2132568, -949.1043091, 1437.8586426, -2483.5217285, 2526.3173828

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192143, upper bound: 2204.5430670
time: 1.22 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5158295, upper bound: 2204.5223527
time: 1.44 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5158053, upper bound: 2204.5402032
time: 1.50 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -761.7991943, 1223.0042725, -2259.0422363, 2420.5546875
1: -1163.6542969, 1695.2358398, -856.3837280, 1249.9602051, -2413.6142578, 2551.6193848
2: -1173.1274414, 1693.5303955, -863.7846680, 1249.4012451, -2422.5285645, 2557.3146973
3: -1421.6109619, 1955.9685059, -1047.9674072, 1441.5981445, -2863.2089844, 3003.9360352
4: -1286.8170166, 1948.8736572, -949.1043091, 1437.8586426, -2724.6757812, 2897.9772949

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5192143, upper bound: 2204.5430670
time: 1.73 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5158295, upper bound: 2204.5223527
time: 1.26 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5158053, upper bound: 2204.5402032
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -960.3131714, 1541.9914551, -2380.8510742, 2301.6567383
1: -942.6451416, 1370.9106445, -1079.1201172, 1575.9105225, -2518.5546875, 2450.0307617
2: -950.4666748, 1370.2813721, -1087.9730225, 1574.2895508, -2524.7563477, 2458.2543945
3: -1151.6676025, 1582.3807373, -1319.8493652, 1817.4102783, -2969.0773926, 2902.2294922
4: -1045.6632080, 1577.2132568, -1191.9132080, 1811.8625488, -2857.5251465, 2769.1259766

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146654, upper bound: 2204.5177628
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146174, upper bound: 2204.5362842
time: 1.33 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -960.3131714, 1541.9914551, -2578.0297852, 2619.0690918
1: -1163.6542969, 1695.2358398, -1079.1201172, 1575.9105225, -2739.5644531, 2774.3557129
2: -1173.1274414, 1693.5303955, -1087.9730225, 1574.2895508, -2747.4169922, 2781.5034180
3: -1421.6109619, 1955.9685059, -1319.8493652, 1817.4102783, -3239.0207520, 3275.8173828
4: -1286.8170166, 1948.8736572, -1191.9132080, 1811.8625488, -3097.7680664, 3140.7858887

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146654, upper bound: 2204.5177628
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146174, upper bound: 2204.5362842
time: 1.28 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -891.7757568, 1424.5837402, -749.3162842, 1201.1799316, -2092.9555664, 2173.8999023
1: -1001.9531250, 1456.4133301, -842.1754150, 1227.7322998, -2229.6853027, 2298.5888672
2: -1011.0187988, 1455.1990967, -849.5438843, 1227.4807129, -2238.4990234, 2304.7426758
3: -1222.2701416, 1681.2009277, -1030.1467285, 1416.1391602, -2638.4091797, 2711.3476562
4: -1113.9793701, 1675.1312256, -933.5875244, 1412.3710938, -2526.3505859, 2608.7182617

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
time: 0.92 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
time: 1.25 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -885.9844971, 1414.9328613, -947.7670898, 1520.2650146, -2406.2490234, 2362.6997070
1: -995.5231323, 1446.7009277, -1064.8012695, 1553.7296143, -2549.2526855, 2511.5021973
2: -1004.4965210, 1445.3520508, -1073.6849365, 1552.4042969, -2556.9008789, 2519.0366211
3: -1214.2031250, 1669.9685059, -1301.9499512, 1791.9694824, -3006.1726074, 2971.9182129
4: -1106.9653320, 1663.7016602, -1176.3287354, 1786.4251709, -2892.8186035, 2840.0302734

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
time: 1.09 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -891.7757568, 1424.5837402, -878.4448242, 1403.2790527, -2295.0546875, 2303.0285645
1: -1001.9531250, 1456.4133301, -987.0339966, 1434.6789551, -2436.6318359, 2443.4472656
2: -1011.0187988, 1455.1990967, -995.9737549, 1433.5321045, -2444.5507812, 2451.1726074
3: -1222.2701416, 1681.2009277, -1203.9013672, 1656.1054688, -2878.3754883, 2885.1020508
4: -1113.9793701, 1675.1312256, -1097.6905518, 1650.2377930, -2764.2172852, 2772.8212891

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.28 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.10 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -885.9844971, 1414.9328613, -1075.5578613, 1721.3309326, -2607.3146973, 2490.4907227
1: -995.5231323, 1446.7009277, -1207.9868164, 1759.3670654, -2754.8901367, 2654.6877441
2: -1004.4965210, 1445.3520508, -1218.4506836, 1757.5551758, -2762.0517578, 2663.8024902
3: -1214.2031250, 1669.9685059, -1474.0344238, 2029.9600830, -3244.1630859, 3144.0021973
4: -1106.9653320, 1663.7016602, -1338.1461182, 2022.6143799, -3128.8776855, 3001.8476562

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.12 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
time: 1.35 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.94 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5734458, upper bound: 2204.5190710
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5704005, upper bound: 2204.5171000
IS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5734304, upper bound: 2204.5203931
IS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5730960, upper bound: 2204.5199608
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5726068, upper bound: 2204.5158658
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5698180, upper bound: 2204.5138791
IS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5726068, upper bound: 2204.5158658
IS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5698180, upper bound: 2204.5138791
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5970936, upper bound: 2204.5970935
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5762822, upper bound: 2204.5379767
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5158295, upper bound: 2204.5223527
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5158053, upper bound: 2204.5402032
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5158295, upper bound: 2204.5223527
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5158053, upper bound: 2204.5402032
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5146654, upper bound: 2204.5177628
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5146174, upper bound: 2204.5362842
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5146654, upper bound: 2204.5177628
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5146174, upper bound: 2204.5362842
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5379767, upper bound: 2204.5762822
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.94
Output dim: 3, lower bound: -2204.5378976, upper bound: 2204.5388724

## BFS IS instance: IS_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -746.8986206, 1197.3187256, -720.9932861, 1154.6287842, -1901.5273438, 1918.3117676
1: -839.4517212, 1223.7910156, -810.5822754, 1180.3763428, -2019.8277588, 2034.3732910
2: -846.7810669, 1223.5253906, -817.0637817, 1180.1094971, -2026.8903809, 2040.5891113
3: -1026.9119873, 1411.6064453, -992.1181030, 1361.4935303, -2388.4052734, 2403.7246094
4: -930.4861450, 1407.8302002, -897.1439819, 1357.9986572, -2288.4846191, 2304.9741211

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732656, upper bound: 2204.5182724
time: 1.04 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -748.4260864, 1199.7456055, -729.7404785, 1169.5179443, -1917.9438477, 1929.4858398
1: -841.1757812, 1226.2701416, -820.5187378, 1195.4871826, -2036.6628418, 2046.7888184
2: -848.5261841, 1226.0109863, -826.9398804, 1194.9981689, -2043.5244141, 2052.9506836
3: -1028.9526367, 1414.4559326, -1004.3226318, 1378.8468018, -2407.7990723, 2418.7785645
4: -932.4427490, 1410.6839600, -907.8271484, 1375.3747559, -2307.8168945, 2318.5107422

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5695599, upper bound: 2204.5074300
time: 1.05 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -943.3429565, 1513.1239014, -718.8193359, 1152.1289062, -2095.4714355, 2231.9433594
1: -1059.8452148, 1546.4553223, -808.2927856, 1177.7172852, -2237.5625000, 2354.7480469
2: -1068.6732178, 1545.1135254, -814.5812378, 1177.2456055, -2245.9189453, 2359.6948242
3: -1295.8669434, 1783.5759277, -989.4021606, 1358.2993164, -2654.1662598, 2772.9780273
4: -1170.8377686, 1778.0354004, -894.2287598, 1354.9414062, -2525.7792969, 2672.2636719

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5590745, upper bound: 2204.5108272
time: 1.34 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705844, upper bound: 2204.5108347
time: 1.97 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -941.4589844, 1510.0350342, -802.8938599, 1292.0780029, -2233.5371094, 2312.9289551
1: -1057.7227783, 1543.2669678, -903.2955933, 1321.4384766, -2379.1611328, 2446.5622559
2: -1066.5292969, 1541.9229736, -910.3615723, 1319.1328125, -2385.6621094, 2452.2846680
3: -1293.2600098, 1779.9460449, -1105.9705811, 1524.3988037, -2817.6586914, 2885.9162598
4: -1168.4375000, 1774.3676758, -1001.8242188, 1517.3343506, -2685.7717285, 2776.1916504

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5724666, upper bound: 2204.5180491
time: 1.42 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5696404, upper bound: 2204.5167863
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -746.8973389, 1197.3164062, -919.7720947, 1473.4671631, -2220.3640137, 2117.0881348
1: -839.4500732, 1223.7886963, -1033.4727783, 1506.4949951, -2345.9450684, 2257.2614746
2: -846.7794189, 1223.5230713, -1041.6309814, 1504.9711914, -2351.7504883, 2265.1538086
3: -1026.9102783, 1411.6038818, -1264.1079102, 1737.4920654, -2764.4023438, 2675.7111816
4: -930.4843140, 1407.8273926, -1140.7159424, 1731.9567871, -2662.4411621, 2548.5432129

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5753279, upper bound: 2204.5168856
time: 1.55 seconds

## Relational analysis of IS_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5748864, upper bound: 2204.5157994
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -748.4260864, 1199.7456055, -928.4917603, 1488.3095703, -2236.7355957, 2128.2373047
1: -841.1757812, 1226.2701416, -1043.3935547, 1521.5067139, -2362.6823730, 2269.6633301
2: -848.5261841, 1226.0109863, -1051.4704590, 1519.8994141, -2368.4255371, 2277.4812012
3: -1028.9526367, 1414.4559326, -1276.3092041, 1754.6802979, -2783.6328125, 2690.7648926
4: -932.4427490, 1410.6839600, -1151.3472900, 1749.3391113, -2681.7817383, 2562.0305176

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5737015, upper bound: 2204.5147660
time: 1.29 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5725742, upper bound: 2204.5138666
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -945.2483521, 1516.1433105, -919.7720947, 1473.4671631, -2418.7155762, 2435.9150391
1: -1062.0131836, 1549.5283203, -1033.4727783, 1506.4949951, -2568.5083008, 2583.0009766
2: -1070.7987061, 1548.2006836, -1041.6309814, 1504.9711914, -2575.7700195, 2589.8315430
3: -1298.5712891, 1787.1441650, -1264.1079102, 1737.4920654, -3036.0634766, 3051.2514648
4: -1173.0750732, 1781.5924072, -1140.7159424, 1731.9567871, -2905.0317383, 2922.3081055

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5584053, upper bound: 2204.5064074
time: 0.98 seconds

## Relational analysis of IS_A1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5702425, upper bound: 2204.5065758
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -946.8806152, 1518.8071289, -928.4917603, 1488.3095703, -2435.1901855, 2447.2985840
1: -1063.8168945, 1552.2452393, -1043.3935547, 1521.5067139, -2585.3237305, 2595.6386719
2: -1072.6711426, 1550.9163818, -1051.4704590, 1519.8994141, -2592.5705566, 2602.3867188
3: -1300.7518311, 1790.2640381, -1276.3092041, 1754.6802979, -3055.4313965, 3066.5727539
4: -1175.1893311, 1784.7138672, -1151.3472900, 1749.3391113, -2924.5283203, 2936.0607910

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5494125, upper bound: 2204.5038540
time: 1.15 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5684367, upper bound: 2204.5051504
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -749.2294312, 1201.0413818, -1950.3576660, 1950.4091797
1: -842.1754150, 1227.7322998, -842.0776367, 1227.5903320, -2069.7656250, 2069.8095703
2: -849.5438843, 1227.4807129, -849.4456177, 1227.3383789, -2076.8823242, 2076.9260254
3: -1030.1467285, 1416.1391602, -1030.0261230, 1415.9755859, -2446.1223145, 2446.1652832
4: -933.5875244, 1412.3710938, -933.4797974, 1412.2065430, -2345.7939453, 2345.8508301

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6000344, upper bound: 2204.5974939
time: 1.15 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5993618, upper bound: 2204.5971078
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -947.6765747, 1520.1228027, -2269.4389648, 2148.8564453
1: -842.1754150, 1227.7322998, -1064.6992188, 1553.5836182, -2395.7590332, 2292.4316406
2: -849.5438843, 1227.4807129, -1073.5823975, 1552.2584229, -2401.8022461, 2301.0627441
3: -1030.1467285, 1416.1391602, -1301.8259277, 1791.8009033, -2821.9475098, 2717.9650879
4: -933.5875244, 1412.3710938, -1176.2172852, 1786.2565918, -2719.8442383, 2588.5881348

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.6000344, upper bound: 2204.5974939
time: 1.64 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5993618, upper bound: 2204.5971078
time: 1.95 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -749.2294312, 1201.0413818, -2148.8083496, 2269.4943848
1: -1064.8012695, 1553.7296143, -842.0776367, 1227.5903320, -2292.3916016, 2395.8071289
2: -1073.6849365, 1552.4042969, -849.4456177, 1227.3383789, -2301.0231934, 2401.8498535
3: -1301.9499512, 1791.9694824, -1030.0261230, 1415.9755859, -2717.9255371, 2821.9956055
4: -1176.3287354, 1786.4251709, -933.4797974, 1412.2065430, -2588.5351562, 2719.9050293

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5835589, upper bound: 2204.5856102
time: 1.17 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5783437, upper bound: 2204.5783436
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -947.6765747, 1520.1228027, -2467.8894043, 2467.9416504
1: -1064.8012695, 1553.7296143, -1064.6992188, 1553.5836182, -2618.3847656, 2618.4287109
2: -1073.6849365, 1552.4042969, -1073.5823975, 1552.2584229, -2625.9431152, 2625.9868164
3: -1301.9499512, 1791.9694824, -1301.8259277, 1791.8009033, -3093.7507324, 3093.7954102
4: -1176.3287354, 1786.4251709, -1176.2172852, 1786.2565918, -2962.5852051, 2962.6420898

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5667645, upper bound: 2204.5872019
time: 1.04 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5947394, upper bound: 2204.5947394
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -878.3933716, 1403.1925049, -2152.5087891, 2079.5732422
1: -842.1754150, 1227.7322998, -986.9750977, 1434.5900879, -2276.7656250, 2214.7075195
2: -849.5438843, 1227.4807129, -995.9152832, 1433.4410400, -2282.9841309, 2223.3957520
3: -1030.1467285, 1416.1391602, -1203.8254395, 1656.0020752, -2686.1486816, 2619.9645996
4: -933.5875244, 1412.3710938, -1097.6263428, 1650.1330566, -2583.7207031, 2509.9975586

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762915, upper bound: 2204.5205885
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5786948, upper bound: 2204.5372762
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -1075.5012207, 1721.2386475, -2470.5549316, 2276.6806641
1: -842.1754150, 1227.7322998, -1207.9224854, 1759.2728271, -2601.4482422, 2435.6547852
2: -849.5438843, 1227.4807129, -1218.3870850, 1757.4587402, -2607.0026855, 2445.8669434
3: -1030.1467285, 1416.1391602, -1473.9526367, 2029.8526611, -3059.9990234, 2890.0917969
4: -933.5875244, 1412.3710938, -1338.0777588, 2022.5037842, -2956.0913086, 2750.4487305

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5762915, upper bound: 2204.5205885
time: 1.21 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5786948, upper bound: 2204.5372762
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -878.3933716, 1403.1925049, -2350.9592285, 2398.6582031
1: -1064.8012695, 1553.7296143, -986.9750977, 1434.5900879, -2499.3913574, 2540.7045898
2: -1073.6849365, 1552.4042969, -995.9152832, 1433.4410400, -2507.1250000, 2548.3195801
3: -1301.9499512, 1791.9694824, -1203.8254395, 1656.0020752, -2957.9521484, 2995.7949219
4: -1176.3287354, 1786.4251709, -1097.6263428, 1650.1330566, -2826.4616699, 2883.4372559

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732473, upper bound: 2204.5191544
time: 1.46 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756894, upper bound: 2204.5363236
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -1075.5012207, 1721.2386475, -2669.0051270, 2595.7658691
1: -1064.8012695, 1553.7296143, -1207.9224854, 1759.2728271, -2824.0742188, 2761.6520996
2: -1073.6849365, 1552.4042969, -1218.3870850, 1757.4587402, -2831.1435547, 2770.7912598
3: -1301.9499512, 1791.9694824, -1473.9526367, 2029.8526611, -3331.8024902, 3265.9218750
4: -1176.3287354, 1786.4251709, -1338.0777588, 2022.5037842, -3198.8322754, 3122.5883789

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5732473, upper bound: 2204.5191544
time: 1.29 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756894, upper bound: 2204.5363236
time: 1.20 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -837.7816772, 1339.6372070, -749.5993652, 1202.8864746, -2040.6680908, 2089.2363281
1: -941.4290771, 1369.1697998, -842.6661987, 1229.5428467, -2170.9719238, 2211.8359375
2: -949.2374268, 1368.5407715, -849.8372803, 1228.9238281, -2178.1611328, 2218.3776855
3: -1150.2155762, 1580.3697510, -1031.2041016, 1418.0892334, -2568.3046875, 2611.5737305
4: -1044.2952881, 1575.2055664, -933.8363647, 1414.2680664, -2458.5632324, 2509.0412598

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5080427, upper bound: 2204.5185763
time: 1.68 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5070826, upper bound: 2204.5164632
time: 1.15 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -761.5658569, 1222.6368408, -2061.4963379, 2102.9091797
1: -942.6451416, 1370.9106445, -856.1231079, 1249.5834961, -2192.2277832, 2227.0336914
2: -950.4666748, 1370.2813721, -863.5204468, 1249.0253906, -2199.4914551, 2233.8017578
3: -1151.6676025, 1582.3807373, -1047.6505127, 1441.1630859, -2592.8305664, 2630.0312500
4: -1045.6632080, 1577.2132568, -948.8146362, 1437.4260254, -2483.0891113, 2526.0278320

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5188839, upper bound: 2204.5410614
time: 1.43 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5080399, upper bound: 2204.5355529
time: 1.20 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5070628, upper bound: 2204.5316356
time: 1.19 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1034.9523926, 1657.0207520, -749.5993652, 1202.8864746, -2237.8388672, 2406.6196289
1: -1162.4289551, 1693.4609375, -842.6661987, 1229.5428467, -2391.9716797, 2536.1271973
2: -1171.8905029, 1691.7552490, -849.8372803, 1228.9238281, -2400.8144531, 2541.5922852
3: -1420.1429443, 1953.9270020, -1031.2041016, 1418.0892334, -2838.2321777, 2985.1308594
4: -1285.4415283, 1946.8304443, -933.8363647, 1414.2680664, -2699.7094727, 2880.6667480

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154643, upper bound: 2204.5220667
time: 1.14 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146869, upper bound: 2204.5197256
time: 1.16 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -761.5658569, 1222.6368408, -2258.6752930, 2420.3212891
1: -1163.6542969, 1695.2358398, -856.1231079, 1249.5834961, -2413.2375488, 2551.3588867
2: -1173.1274414, 1693.5303955, -863.5204468, 1249.0253906, -2422.1523438, 2557.0502930
3: -1421.6109619, 1955.9685059, -1047.6505127, 1441.1630859, -2862.7739258, 3003.6191406
4: -1286.8170166, 1948.8736572, -948.8146362, 1437.4260254, -2724.2429199, 2897.6879883

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154891, upper bound: 2204.5402033
time: 1.19 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5154277, upper bound: 2204.5397029
time: 1.78 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5146420, upper bound: 2204.5393451
time: 1.23 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -837.7816772, 1339.6372070, -948.6199951, 1522.4893799, -2360.2709961, 2288.2570801
1: -941.4290771, 1369.1697998, -1065.9644775, 1556.2204590, -2497.6491699, 2435.1342773
2: -949.2374268, 1368.5407715, -1074.6245117, 1554.5330811, -2503.7702637, 2443.1650391
3: -1150.2155762, 1580.3697510, -1303.7680664, 1794.6920166, -2944.9072266, 2884.1376953
4: -1044.2952881, 1575.2055664, -1177.3999023, 1789.0861816, -2833.3813477, 2752.6049805

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5175706, upper bound: 2204.5175935
time: 1.25 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5175206, upper bound: 2204.5173970
time: 1.01 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -838.8595581, 1341.3438721, -960.0626221, 1541.5938721, -2380.4533691, 2301.4062500
1: -942.6451416, 1370.9106445, -1078.8389893, 1575.5019531, -2518.1464844, 2449.7495117
2: -950.4666748, 1370.2813721, -1087.6892090, 1573.8823242, -2524.3486328, 2457.9707031
3: -1151.6676025, 1582.3807373, -1319.5059814, 1816.9394531, -2968.6069336, 2901.8867188
4: -1045.6632080, 1577.2132568, -1191.6013184, 1811.3940430, -2857.0571289, 2768.8144531

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5168294, upper bound: 2204.5369069
time: 1.42 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5175011, upper bound: 2204.5362038
time: 1.54 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5174502, upper bound: 2204.5359243
time: 1.28 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1034.9523926, 1657.0207520, -948.6199951, 1522.4893799, -2557.4416504, 2605.6403809
1: -1162.4289551, 1693.4609375, -1065.9644775, 1556.2204590, -2718.6494141, 2759.4252930
2: -1171.8905029, 1691.7552490, -1074.6245117, 1554.5330811, -2726.4235840, 2766.3793945
3: -1420.1429443, 1953.9270020, -1303.7680664, 1794.6920166, -3214.8349609, 3257.6950684
4: -1285.4415283, 1946.8304443, -1177.3999023, 1789.0861816, -3073.7294922, 3124.2304688

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5057425, upper bound: 2204.5122464
time: 1.13 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5058611, upper bound: 2204.5092041
time: 1.18 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -960.0626221, 1541.5938721, -2577.6320801, 2618.8183594
1: -1163.6542969, 1695.2358398, -1078.8389893, 1575.5019531, -2739.1562500, 2774.0744629
2: -1173.1274414, 1693.5303955, -1087.6892090, 1573.8823242, -2747.0097656, 2781.2194824
3: -1421.6109619, 1955.9685059, -1319.5059814, 1816.9394531, -3238.5502930, 3275.4746094
4: -1286.8170166, 1948.8736572, -1191.6013184, 1811.3940430, -3097.2590332, 3140.4748535

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5137063, upper bound: 2204.5362842
time: 1.40 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4957107, upper bound: 2204.5294794
time: 1.18 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5058308, upper bound: 2204.5314248
time: 1.38 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -749.3162842, 1201.1799316, -2079.5732422, 2152.5087891
1: -986.9750977, 1434.5900879, -842.1754150, 1227.7322998, -2214.7075195, 2276.7656250
2: -995.9152832, 1433.4410400, -849.5438843, 1227.4807129, -2223.3955078, 2282.9843750
3: -1203.8254395, 1656.0020752, -1030.1467285, 1416.1391602, -2619.9645996, 2686.1486816
4: -1097.6263428, 1650.1330566, -933.5875244, 1412.3710938, -2509.9975586, 2583.7207031

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5205885, upper bound: 2204.5762915
time: 1.45 seconds

## Relational analysis of IS_A2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5372762, upper bound: 2204.5786948
time: 1.23 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -749.3162842, 1201.1799316, -2276.6806641, 2470.5549316
1: -1207.9224854, 1759.2728271, -842.1754150, 1227.7322998, -2435.6547852, 2601.4482422
2: -1218.3870850, 1757.4587402, -849.5438843, 1227.4807129, -2445.8671875, 2607.0026855
3: -1473.9526367, 2029.8526611, -1030.1467285, 1416.1391602, -2890.0915527, 3059.9987793
4: -1338.0777588, 2022.5037842, -933.5875244, 1412.3710938, -2750.4487305, 2956.0913086

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5205885, upper bound: 2204.5762915
time: 1.23 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5372762, upper bound: 2204.5786948
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -947.7670898, 1520.2650146, -2398.6582031, 2350.9592285
1: -986.9750977, 1434.5900879, -1064.8012695, 1553.7296143, -2540.7045898, 2499.3913574
2: -995.9152832, 1433.4410400, -1073.6849365, 1552.4042969, -2548.3193359, 2507.1252441
3: -1203.8254395, 1656.0020752, -1301.9499512, 1791.9694824, -2995.7949219, 2957.9521484
4: -1097.6263428, 1650.1330566, -1176.3287354, 1786.4251709, -2883.4372559, 2826.4616699

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5191544, upper bound: 2204.5732473
time: 1.84 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5363236, upper bound: 2204.5756894
time: 1.84 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -947.7670898, 1520.2650146, -2595.7658691, 2669.0051270
1: -1207.9224854, 1759.2728271, -1064.8012695, 1553.7296143, -2761.6520996, 2824.0742188
2: -1218.3870850, 1757.4587402, -1073.6849365, 1552.4042969, -2770.7910156, 2831.1435547
3: -1473.9526367, 2029.8526611, -1301.9499512, 1791.9694824, -3265.9218750, 3331.8024902
4: -1338.0777588, 2022.5037842, -1176.3287354, 1786.4251709, -3122.5888672, 3198.8322754

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5191544, upper bound: 2204.5732473
time: 1.44 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5363236, upper bound: 2204.5756894
time: 1.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -878.4448242, 1403.2790527, -2281.6723633, 2281.6372070
1: -986.9750977, 1434.5900879, -987.0339966, 1434.6789551, -2421.6540527, 2421.6240234
2: -995.9152832, 1433.4410400, -995.9737549, 1433.5321045, -2429.4472656, 2429.4143066
3: -1203.8254395, 1656.0020752, -1203.9013672, 1656.1054688, -2859.9309082, 2859.9025879
4: -1097.6263428, 1650.1330566, -1097.6905518, 1650.2377930, -2747.8642578, 2747.8237305

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5213993, upper bound: 2204.5240053
time: 1.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5372013, upper bound: 2204.5412095
time: 2.02 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -878.4448242, 1403.2790527, -2478.7800293, 2599.6833496
1: -1207.9224854, 1759.2728271, -987.0339966, 1434.6789551, -2642.6015625, 2746.3068848
2: -1218.3870850, 1757.4587402, -995.9737549, 1433.5321045, -2651.9189453, 2753.4323730
3: -1473.9526367, 2029.8526611, -1203.9013672, 1656.1054688, -3130.0576172, 3233.7529297
4: -1338.0777588, 2022.5037842, -1097.6905518, 1650.2377930, -2988.3154297, 3119.4340820

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5213993, upper bound: 2204.5240052
time: 2.17 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5372013, upper bound: 2204.5412095
time: 1.53 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -878.3933716, 1403.1925049, -1075.5578613, 1721.3309326, -2599.7236328, 2478.7504883
1: -986.9750977, 1434.5900879, -1207.9868164, 1759.3670654, -2746.3422852, 2642.5766602
2: -995.9152832, 1433.4410400, -1218.4506836, 1757.5551758, -2753.4699707, 2651.8908691
3: -1203.8254395, 1656.0020752, -1474.0344238, 2029.9600830, -3233.7856445, 3130.0361328
4: -1097.6263428, 1650.1330566, -1338.1461182, 2022.6143799, -3119.4965820, 2988.2792969

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5206611, upper bound: 2204.5196232
time: 1.65 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5361878, upper bound: 2204.5370850
time: 1.67 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -1075.5578613, 1721.3309326, -2796.8293457, 2796.7893066
1: -1207.9224854, 1759.2728271, -1207.9868164, 1759.3670654, -2967.2895508, 2967.2595215
2: -1218.3870850, 1757.4587402, -1218.4506836, 1757.5551758, -2975.9414062, 2975.9094238
3: -1473.9526367, 2029.8526611, -1474.0344238, 2029.9600830, -3503.9123535, 3503.8864746
4: -1338.0777588, 2022.5037842, -1338.1461182, 2022.6143799, -3358.6479492, 3358.5895996

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5206611, upper bound: 2204.5196232
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5361878, upper bound: 2204.5370850
time: 1.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.36 seconds
IS_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5732656, upper bound: 2204.5182724
IS_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
IS_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5695599, upper bound: 2204.5074300
IS_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
IS_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5590745, upper bound: 2204.5108272
IS_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5705844, upper bound: 2204.5108347
IS_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5724666, upper bound: 2204.5180491
IS_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5696404, upper bound: 2204.5167863
IS_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5753279, upper bound: 2204.5168856
IS_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5748864, upper bound: 2204.5157994
IS_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5737015, upper bound: 2204.5147660
IS_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5725742, upper bound: 2204.5138666
IS_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5584053, upper bound: 2204.5064074
IS_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5702425, upper bound: 2204.5065758
IS_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5494125, upper bound: 2204.5038540
IS_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5684367, upper bound: 2204.5051504
IS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.6000344, upper bound: 2204.5974939
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5993618, upper bound: 2204.5971078
IS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.6000344, upper bound: 2204.5974939
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5993618, upper bound: 2204.5971078
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5835589, upper bound: 2204.5856102
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5783437, upper bound: 2204.5783436
IS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5667645, upper bound: 2204.5872019
IS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5947394, upper bound: 2204.5947394
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5762915, upper bound: 2204.5205885
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5786948, upper bound: 2204.5372762
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5762915, upper bound: 2204.5205885
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5786948, upper bound: 2204.5372762
IS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5732473, upper bound: 2204.5191544
IS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5756894, upper bound: 2204.5363236
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5732473, upper bound: 2204.5191544
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5756894, upper bound: 2204.5363236
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5080427, upper bound: 2204.5185763
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5070826, upper bound: 2204.5164632
IS_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5080399, upper bound: 2204.5355529
IS_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5070628, upper bound: 2204.5316356
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5154643, upper bound: 2204.5220667
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5146869, upper bound: 2204.5197256
IS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5154277, upper bound: 2204.5397029
IS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5146420, upper bound: 2204.5393451
IS_A2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5175706, upper bound: 2204.5175935
IS_A2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5175206, upper bound: 2204.5173970
IS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5175011, upper bound: 2204.5362038
IS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5174502, upper bound: 2204.5359243
IS_A2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5057425, upper bound: 2204.5122464
IS_A2_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5058611, upper bound: 2204.5092041
IS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.4957107, upper bound: 2204.5294794
IS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5058308, upper bound: 2204.5314248
IS_A2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5205885, upper bound: 2204.5762915
IS_A2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5372762, upper bound: 2204.5786948
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5205885, upper bound: 2204.5762915
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5372762, upper bound: 2204.5786948
IS_A2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5191544, upper bound: 2204.5732473
IS_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5363236, upper bound: 2204.5756894
IS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5191544, upper bound: 2204.5732473
IS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5363236, upper bound: 2204.5756894
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5213993, upper bound: 2204.5240053
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5372013, upper bound: 2204.5412095
IS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5213993, upper bound: 2204.5240052
IS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5372013, upper bound: 2204.5412095
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5206611, upper bound: 2204.5196232
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5361878, upper bound: 2204.5370850
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5206611, upper bound: 2204.5196232
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.36
Output dim: 3, lower bound: -2204.5361878, upper bound: 2204.5370850

## BFS IS instance: IS_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -709.6517334, 1136.3507080, -712.8638916, 1141.2774658, -1850.9289551, 1849.2144775
1: -797.4386597, 1161.5200195, -801.4086914, 1166.7434082, -1964.1821289, 1962.9287109
2: -804.5056763, 1161.2822266, -807.8398438, 1166.4477539, -1970.9533691, 1969.1220703
3: -974.9863892, 1339.5117188, -980.7592773, 1345.7318115, -2320.7182617, 2320.2707520
4: -883.6143799, 1336.0092773, -886.8548584, 1342.2639160, -2225.8784180, 2222.8642578

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
time: 1.25 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1024.7283936, 1653.1175537, -715.5639038, 1146.0478516, -2170.7763672, 2368.6806641
1: -1153.0982666, 1690.3411865, -804.5200195, 1171.6102295, -2324.7084961, 2494.8613281
2: -1161.9074707, 1688.5833740, -810.9267578, 1171.3281250, -2333.2353516, 2499.5102539
3: -1413.0646973, 1947.9179688, -984.7037354, 1351.3599854, -2764.4248047, 2932.6215820
4: -1271.6116943, 1944.5501709, -890.4281006, 1347.8732910, -2619.4848633, 2834.4748535

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
time: 1.63 seconds

## Relational analysis of IS_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5705236, upper bound: 2204.5164417
time: 1.61 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -711.2725220, 1138.9128418, -721.6637573, 1156.2457275, -1867.5183105, 1860.5765381
1: -799.2694702, 1164.1492920, -811.4054565, 1181.9193115, -1981.1887207, 1975.5545654
2: -806.3590698, 1163.9141846, -817.7687988, 1181.4129639, -1987.7717285, 1981.6829834
3: -977.1437378, 1342.5336914, -993.0387573, 1363.1429443, -2340.2866211, 2335.5725098
4: -885.6924438, 1339.0268555, -897.5688477, 1359.7330322, -2245.4255371, 2236.5957031

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
time: 1.33 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
time: 1.58 seconds

## BFS IS instance: IS_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1026.7005615, 1656.2982178, -724.4221191, 1161.0937500, -2187.7941895, 2380.7202148
1: -1155.2761230, 1693.6000977, -814.5828857, 1186.8862305, -2342.1623535, 2508.1831055
2: -1164.1197510, 1691.8369141, -820.9314575, 1186.3818359, -2350.5014648, 2512.7683105
3: -1415.6938477, 1951.6755371, -997.0610962, 1368.9001465, -2784.5937500, 2948.7365723
4: -1274.0827637, 1948.2630615, -901.2508545, 1365.4431152, -2639.5256348, 2849.2868652

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
time: 1.92 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5648876, upper bound: 2204.5065940
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -936.6790161, 1502.2987061, -715.0700684, 1146.0957031, -2082.7746582, 2217.3686523
1: -1052.3406982, 1535.2174072, -804.0463257, 1171.4744873, -2223.8149414, 2339.2631836
2: -1060.9472656, 1534.0428467, -810.2907104, 1171.0708008, -2232.0175781, 2344.3332520
3: -1286.8476562, 1770.5688477, -984.2487183, 1351.0606689, -2637.9082031, 2754.8176270
4: -1161.2202148, 1765.5526123, -889.3143921, 1347.8750000, -2509.0939941, 2654.8666992

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5582418, upper bound: 2204.5093053
time: 1.03 seconds

## Relational analysis of IS_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5485685, upper bound: 2204.5070578
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -938.0631714, 1504.5778809, -718.8193359, 1152.1289062, -2090.1916504, 2223.3969727
1: -1053.8879395, 1537.6842041, -808.2927856, 1177.7172852, -2231.6047363, 2345.9770508
2: -1062.6871338, 1536.3867188, -814.5812378, 1177.2456055, -2239.9326172, 2350.9680176
3: -1288.5396729, 1773.3863525, -989.4021606, 1358.2993164, -2646.8388672, 2762.7885742
4: -1164.2294922, 1768.0191650, -894.2287598, 1354.9414062, -2519.1706543, 2662.2480469

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=2549.14794921875
rel_dist={3: [-2204.604555886248, 2204.604555886248]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5485323, upper bound: 2204.5112927
time: 1.10 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5821340, upper bound: 2204.5821340
time: 1.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.59 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 3, lower bound: -2204.5485323, upper bound: 2204.5112927
IS_B2, status: Status.UNKNOWN, split count: 1, time: 2.59
Output dim: 3, lower bound: -2204.5821340, upper bound: 2204.5821340

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -778.1298828, 1248.9666748, -746.0270996, 1195.2812500, -1973.4108887, 1994.9935303
1: -874.7517700, 1276.5758057, -838.7745361, 1221.9118652, -2096.6635742, 2115.3491211
2: -882.2230225, 1275.9195557, -845.3336792, 1221.3171387, -2103.5397949, 2121.2529297
3: -1070.4603271, 1472.2502441, -1026.6439209, 1409.3378906, -2479.7983398, 2498.8935547
4: -969.1489868, 1468.2788086, -927.8871460, 1405.6160889, -2374.7651367, 2396.1660156

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5106658, upper bound: 2204.5106658
time: 1.22 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5106658, upper bound: 2204.5112927
time: 1.28 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -780.0952148, 1252.1962891, -780.0045166, 1252.0537109, -2032.1486816, 2032.2006836
1: -876.9357910, 1279.8708496, -876.8342896, 1279.7246094, -2156.6604004, 2156.7050781
2: -884.4795532, 1279.2316895, -884.3770752, 1279.0852051, -2163.5646973, 2163.6086426
3: -1073.1099854, 1476.0378418, -1072.9860840, 1475.8693848, -2548.9792480, 2549.0239258
4: -971.7084351, 1472.0739746, -971.5972900, 1471.9053955, -2443.6137695, 2443.6711426

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5296203, upper bound: 2204.5525989
time: 1.22 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5313566
time: 1.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 5.40 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 5.40
Output dim: 3, lower bound: -2204.5106658, upper bound: 2204.5106658
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 3, lower bound: -2204.5106658, upper bound: 2204.5112927
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 3, lower bound: -2204.5296203, upper bound: 2204.5525989
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 5.40
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5313566

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -780.0045166, 1252.0537109, -746.0270996, 1195.2812500, -1975.2854004, 1998.0805664
1: -876.8342896, 1279.7246094, -838.7745361, 1221.9118652, -2098.7456055, 2118.4985352
2: -884.3770752, 1279.0852051, -845.3336792, 1221.3171387, -2105.6940918, 2124.4189453
3: -1072.9860840, 1475.8693848, -1026.6439209, 1409.3378906, -2482.3237305, 2502.5122070
4: -971.5972900, 1471.9053955, -927.8871460, 1405.6160889, -2377.2131348, 2399.7924805

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5066872, upper bound: 2204.5053903
time: 1.68 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -2204.5050645, upper bound: 2204.5050645
time: 1.56 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -780.0952148, 1252.1962891, -765.6796265, 1227.1473389, -2007.2424316, 2017.8756104
1: -876.9357910, 1279.8708496, -860.5095825, 1254.3399658, -2131.2756348, 2140.3803711
2: -884.4795532, 1279.2316895, -868.0309448, 1253.9733887, -2138.4523926, 2147.2626953
3: -1073.1099854, 1476.0378418, -1052.6195068, 1446.8366699, -2519.9467773, 2528.6572266
4: -971.7084351, 1472.0739746, -953.7347412, 1442.8133545, -2414.5217285, 2425.8085938

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5111565, upper bound: 2204.5476366
time: 1.47 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
time: 1.33 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -778.2990112, 1249.1256104, -895.3894043, 1430.3947754, -2208.6938477, 2144.5144043
1: -874.8905640, 1276.7086182, -1006.0085449, 1462.3494873, -2337.2399902, 2282.7172852
2: -882.4204102, 1276.0067139, -1015.0972290, 1461.1177979, -2343.5380859, 2291.1035156
3: -1070.6040039, 1472.5290527, -1227.2722168, 1688.0335693, -2758.6376953, 2699.8012695
4: -969.4312134, 1468.4830322, -1118.3990479, 1681.9255371, -2651.3566895, 2586.8820801

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5296202
time: 1.66 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5313566
time: 1.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.12 seconds
IS_B1_A2_A1, status: Status.VERIFIED, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5066872, upper bound: 2204.5053903
IS_B1_A2_A2, status: Status.VERIFIED, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5050645, upper bound: 2204.5050645
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5111565, upper bound: 2204.5476366
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5296202
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.12
Output dim: 3, lower bound: -2204.5313566, upper bound: 2204.5313566

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -767.3154297, 1231.0120850, -763.3948975, 1223.4837646, -1990.7989502, 1994.4069824
1: -862.5448608, 1258.3776855, -857.9296875, 1250.6003418, -2113.1447754, 2116.3073730
2: -869.8714600, 1257.6816406, -865.4323730, 1250.2254639, -2120.0969238, 2123.1140137
3: -1055.5042725, 1451.2878418, -1049.5124512, 1442.5267334, -2498.0310059, 2500.8002930
4: -955.7076416, 1447.2781982, -950.8513184, 1438.5004883, -2394.2080078, 2398.1291504

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5102593, upper bound: 2204.5444859
time: 1.73 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
time: 1.84 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -779.8619385, 1251.8294678, -765.6796265, 1227.1473389, -2007.0092773, 2017.5086670
1: -876.6753540, 1279.4943848, -860.5095825, 1254.3399658, -2131.0153809, 2140.0039062
2: -884.2153931, 1278.8559570, -868.0309448, 1253.9733887, -2138.1884766, 2146.8869629
3: -1072.7932129, 1475.6033936, -1052.6195068, 1446.8366699, -2519.6298828, 2528.2229004
4: -971.4189453, 1471.6417236, -953.7347412, 1442.8133545, -2414.2324219, 2425.3764648

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
time: 1.79 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
time: 1.50 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -765.7679443, 1227.2880859, -895.3894043, 1430.3947754, -2196.1625977, 2122.6765137
1: -860.6093140, 1254.4846191, -1006.0085449, 1462.3494873, -2322.9584961, 2260.4926758
2: -868.1311646, 1254.1179199, -1015.0972290, 1461.1177979, -2329.2490234, 2269.2150879
3: -1052.7419434, 1447.0034180, -1227.2722168, 1688.0335693, -2740.7753906, 2674.2756348
4: -953.8444824, 1442.9805908, -1118.3990479, 1681.9255371, -2635.7690430, 2561.3793945

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5291979, upper bound: 2204.5254768
time: 1.50 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5254768
time: 1.02 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -895.4413452, 1430.4816895, -895.3894043, 1430.3947754, -2325.8361816, 2325.8701172
1: -1006.0681763, 1462.4388428, -1006.0085449, 1462.3494873, -2468.4174805, 2468.4472656
2: -1015.1561890, 1461.2092285, -1015.0972290, 1461.1177979, -2476.2739258, 2476.3063965
3: -1227.3483887, 1688.1374512, -1227.2722168, 1688.0335693, -2915.3818359, 2915.4096680
4: -1118.4635010, 1682.0313721, -1118.3990479, 1681.9255371, -2800.3881836, 2800.4304199

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5291979, upper bound: 2204.5268785
time: 1.85 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731
time: 1.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.81 seconds
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5102593, upper bound: 2204.5444859
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5280227, upper bound: 2204.5517088
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5291979, upper bound: 2204.5254768
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5254768
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5291979, upper bound: 2204.5268785
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.81
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -751.1117554, 1205.3540039, -753.3646240, 1207.5056152, -1958.6173096, 1958.7186279
1: -844.3523560, 1232.0784912, -846.6795654, 1234.2316895, -2078.5837402, 2078.7580566
2: -851.5693970, 1231.4956055, -854.0999756, 1233.9218750, -2085.4909668, 2085.5957031
3: -1033.2420654, 1420.9379883, -1035.7071533, 1423.6685791, -2456.9104004, 2456.6450195
4: -935.7701416, 1417.1870117, -938.5124512, 1419.7781982, -2355.5476074, 2355.6994629

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
time: 1.28 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
time: 2.08 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -950.0399780, 1524.7962646, -749.7177734, 1201.4722900, -2151.5119629, 2274.5139160
1: -1067.5468750, 1558.5821533, -842.6464844, 1228.1282959, -2295.6752930, 2401.2285156
2: -1076.2495117, 1556.9345703, -849.9912109, 1227.6834717, -2303.9331055, 2406.9257812
3: -1305.6639404, 1797.3331299, -1030.5872803, 1416.5947266, -2722.2587891, 2827.9204102
4: -1179.2060547, 1791.8012695, -934.0432129, 1412.4721680, -2591.6782227, 2725.8444824

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
time: 1.21 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
time: 1.39 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -765.5635986, 1226.9681396, -765.6796265, 1227.1473389, -1992.7109375, 1992.6473389
1: -860.3803101, 1254.1552734, -860.5095825, 1254.3399658, -2114.7202148, 2114.6647949
2: -867.8999023, 1253.7900391, -868.0309448, 1253.9733887, -2121.8730469, 2121.8210449
3: -1052.4631348, 1446.6229248, -1052.6195068, 1446.8366699, -2499.2998047, 2499.2424316
4: -953.5901489, 1442.6029053, -953.7347412, 1442.8133545, -2396.4035645, 2396.3376465

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5275964, upper bound: 2204.5480546
time: 1.42 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.27 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -895.2436523, 1430.1636963, -765.6796265, 1227.1473389, -2122.3906250, 2195.8427734
1: -1005.8454590, 1462.1127930, -860.5095825, 1254.3399658, -2260.1855469, 2322.6220703
2: -1014.9321899, 1460.8818359, -868.0309448, 1253.9733887, -2268.9055176, 2328.9128418
3: -1227.0740967, 1687.7619629, -1052.6195068, 1446.8366699, -2673.9106445, 2740.3811035
4: -1118.2197266, 1681.6555176, -953.7347412, 1442.8133545, -2561.0332031, 2635.3901367

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5512311
time: 1.30 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.30 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -885.0357056, 1413.7781982, -2163.0944824, 2086.2155762
1: -842.1754150, 1227.7322998, -994.4043579, 1445.3883057, -2287.5637207, 2222.1367188
2: -849.5438843, 1227.4807129, -1003.4111938, 1444.2014160, -2293.7453613, 2230.8918457
3: -1030.1467285, 1416.1391602, -1212.9624023, 1668.4932861, -2698.6398926, 2629.1015625
4: -933.5875244, 1412.3710938, -1105.7491455, 1662.5019531, -2596.0893555, 2518.1201172

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A1_A1_B1

### Relational analysis result of IS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
time: 1.51 seconds

## Relational analysis of IS_B2_B2_A1_A1_B2

### Relational analysis result of IS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
time: 1.09 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -881.4982300, 1407.6596680, -2355.4265137, 2401.7631836
1: -1064.8012695, 1553.7296143, -990.5040894, 1439.2949219, -2504.0957031, 2544.2336426
2: -1073.6849365, 1552.4042969, -999.4396362, 1437.9082031, -2511.5925293, 2551.8439941
3: -1301.9499512, 1791.9694824, -1207.9638672, 1661.4476318, -2963.3974609, 2999.9333496
4: -1176.3287354, 1786.4251709, -1101.4860840, 1655.0926514, -2831.4211426, 2887.3283691

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
time: 1.14 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
time: 2.11 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -878.4448242, 1403.2790527, -885.0357056, 1413.7781982, -2292.2231445, 2288.3146973
1: -987.0339966, 1434.6789551, -994.4043579, 1445.3883057, -2432.4223633, 2429.0832520
2: -995.9737549, 1433.5321045, -1003.4111938, 1444.2014160, -2440.1752930, 2436.9433594
3: -1203.9013672, 1656.1054688, -1212.9624023, 1668.4932861, -2872.3940430, 2869.0676270
4: -1097.6905518, 1650.2377930, -1105.7491455, 1662.5019531, -2760.1921387, 2755.9868164

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A2_A1_B1

### Relational analysis result of IS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731
time: 1.47 seconds

## Relational analysis of IS_B2_B2_A2_A1_B2

### Relational analysis result of IS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731
time: 1.30 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -1075.5578613, 1721.3309326, -881.4982300, 1407.6596680, -2483.2175293, 2602.8288574
1: -1207.9868164, 1759.3670654, -990.5040894, 1439.2949219, -2647.2807617, 2749.8710938
2: -1218.4506836, 1757.5551758, -999.4396362, 1437.9082031, -2656.3581543, 2756.9946289
3: -1474.0344238, 2029.9600830, -1207.9638672, 1661.4476318, -3135.4819336, 3237.9238281
4: -1338.1461182, 2022.6143799, -1101.4860840, 1655.0926514, -2993.2387695, 3123.3874512

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268731
time: 1.14 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483
time: 1.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.17 seconds
IS_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
IS_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5071916, upper bound: 2204.5444859
IS_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5275964, upper bound: 2204.5480546
IS_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5512311
IS_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
IS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
IS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
IS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5488341, upper bound: 2204.5254768
IS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731
IS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5254768, upper bound: 2204.5268731
IS_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268731
IS_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.17
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483

## BFS IS instance: IS_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -751.1117554, 1205.3540039, -746.9807739, 1197.4383545, -1948.5500488, 1952.3344727
1: -844.3523560, 1232.0784912, -839.5371094, 1223.9182129, -2068.2702637, 2071.6157227
2: -851.5693970, 1231.4956055, -846.8878784, 1223.6584473, -2075.2272949, 2078.3830566
3: -1033.2420654, 1420.9379883, -1026.9724121, 1411.7440186, -2444.9858398, 2447.9101562
4: -935.7701416, 1417.1870117, -930.6458130, 1407.9719238, -2343.7414551, 2347.8327637

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5079750, upper bound: 2204.5419250
time: 1.68 seconds

## Relational analysis of IS_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5004489, upper bound: 2204.5413808
time: 1.83 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -751.1117554, 1205.3540039, -945.5059814, 1516.5736084, -2267.6853027, 2150.8596191
1: -844.3523560, 1232.0784912, -1062.2741699, 1549.9726562, -2394.3247070, 2294.3525391
2: -851.5693970, 1231.4956055, -1071.1097412, 1548.6457520, -2400.2150879, 2302.6054688
3: -1033.2420654, 1420.9379883, -1298.8649902, 1787.6467285, -2820.8884277, 2719.8024902
4: -935.7701416, 1417.1870117, -1173.4683838, 1782.0917969, -2717.8618164, 2590.6552734

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5093939, upper bound: 2204.5434354
time: 1.63 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5088418, upper bound: 2204.5438197
time: 1.58 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -950.0399780, 1524.7962646, -746.9807739, 1197.4383545, -2147.4780273, 2271.7768555
1: -1067.5468750, 1558.5821533, -839.5371094, 1223.9182129, -2291.4645996, 2398.1191406
2: -1076.2495117, 1556.9345703, -846.8878784, 1223.6584473, -2299.9074707, 2403.8220215
3: -1305.6639404, 1797.3331299, -1026.9724121, 1411.7440186, -2717.4079590, 2824.3054199
4: -1179.2060547, 1791.8012695, -930.6458130, 1407.9719238, -2587.1772461, 2722.4470215

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5041557, upper bound: 2204.5419240
time: 1.52 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4966321, upper bound: 2204.5413574
time: 1.92 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -950.0399780, 1524.7962646, -945.5059814, 1516.5736084, -2466.6130371, 2470.3020020
1: -1067.5468750, 1558.5821533, -1062.2741699, 1549.9726562, -2617.5195312, 2620.8564453
2: -1076.2495117, 1556.9345703, -1071.1097412, 1548.6457520, -2624.8952637, 2628.0444336
3: -1305.6639404, 1797.3331299, -1298.8649902, 1787.6467285, -3093.3105469, 3096.1975098
4: -1179.2060547, 1791.8012695, -1173.4683838, 1782.0917969, -2961.2978516, 2965.2695312

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5041557, upper bound: 2204.5419240
time: 1.25 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4966321, upper bound: 2204.5413574
time: 1.11 seconds

## BFS IS instance: IS_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -749.1112671, 1200.8592529, -755.6222534, 1211.1270752, -1960.2381592, 1956.4814453
1: -841.9461060, 1227.4023438, -849.2305298, 1237.9217529, -2079.8676758, 2076.6328125
2: -849.3118896, 1227.1525879, -856.6679077, 1237.6198730, -2086.9316406, 2083.8198242
3: -1029.8669434, 1415.7583008, -1038.7747803, 1427.9200439, -2457.7866211, 2454.5327148
4: -933.3326416, 1411.9925537, -941.3588867, 1424.0335693, -2357.3662109, 2353.3510742

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A1_A1_B1

### Relational analysis result of IS_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
time: 1.17 seconds

## Relational analysis of IS_B2_B1_A2_A1_A1_B2

### Relational analysis result of IS_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
time: 1.16 seconds

## BFS IS instance: IS_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -947.5657349, 1519.9526367, -751.9766846, 1205.0914307, -2152.6567383, 2271.9291992
1: -1064.5758057, 1553.4077148, -845.1995850, 1231.8175049, -2296.3928223, 2398.6074219
2: -1073.4567871, 1552.0844727, -852.5605469, 1231.3804932, -2304.8374023, 2404.6450195
3: -1301.6760254, 1791.5980225, -1033.6558838, 1420.8444824, -2722.5205078, 2825.2539062
4: -1176.0791016, 1786.0567627, -936.8916016, 1416.7265625, -2592.8056641, 2722.9482422

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
time: 1.53 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
time: 1.27 seconds

## BFS IS instance: IS_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -884.8912964, 1413.5499268, -749.2294312, 1201.0413818, -2085.9326172, 2162.7790527
1: -994.2431641, 1445.1540527, -842.0776367, 1227.5903320, -2221.8334961, 2287.2316895
2: -1003.2477417, 1443.9686279, -849.4456177, 1227.3383789, -2230.5861816, 2293.4143066
3: -1212.7661133, 1668.2246094, -1030.0261230, 1415.9755859, -2628.7414551, 2698.2504883
4: -1105.5716553, 1662.2347412, -933.4797974, 1412.2065430, -2517.7780762, 2595.7145996

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.12 seconds

## Relational analysis of IS_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.33 seconds

## BFS IS instance: IS_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -881.3513794, 1407.4272461, -947.6765747, 1520.1228027, -2401.4741211, 2355.1037598
1: -990.3400879, 1439.0565186, -1064.6992188, 1553.5836182, -2543.9238281, 2503.7558594
2: -999.2733765, 1437.6717529, -1073.5823975, 1552.2584229, -2551.5317383, 2511.2536621
3: -1207.7645264, 1661.1749268, -1301.8259277, 1791.8009033, -2999.5651855, 2963.0004883
4: -1101.3059082, 1654.8208008, -1176.2172852, 1786.2565918, -2886.9545898, 2831.0378418

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.11 seconds

## Relational analysis of IS_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
time: 1.19 seconds

## BFS IS instance: IS_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -878.3933716, 1403.1925049, -2152.5087891, 2079.5732422
1: -842.1754150, 1227.7322998, -986.9750977, 1434.5900879, -2276.7656250, 2214.7075195
2: -849.5438843, 1227.4807129, -995.9152832, 1433.4410400, -2282.9841309, 2223.3957520
3: -1030.1467285, 1416.1391602, -1203.8254395, 1656.0020752, -2686.1486816, 2619.9645996
4: -933.5875244, 1412.3710938, -1097.6263428, 1650.1330566, -2583.7207031, 2509.9975586

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5474736, upper bound: 2204.5072023
time: 1.39 seconds

## Relational analysis of IS_B2_B2_A1_A1_B1_B2

### Relational analysis result of IS_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512311, upper bound: 2204.5237847
time: 1.65 seconds

## BFS IS instance: IS_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -749.3162842, 1201.1799316, -1075.5012207, 1721.2386475, -2470.5549316, 2276.6806641
1: -842.1754150, 1227.7322998, -1207.9224854, 1759.2728271, -2601.4482422, 2435.6547852
2: -849.5438843, 1227.4807129, -1218.3870850, 1757.4587402, -2607.0026855, 2445.8669434
3: -1030.1467285, 1416.1391602, -1473.9526367, 2029.8526611, -3059.9990234, 2890.0917969
4: -933.5875244, 1412.3710938, -1338.0777588, 2022.5037842, -2956.0913086, 2750.4487305

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A1_B2_B1

### Relational analysis result of IS_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5474736, upper bound: 2204.5072023
time: 1.72 seconds

## Relational analysis of IS_B2_B2_A1_A1_B2_B2

### Relational analysis result of IS_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5512311, upper bound: 2204.5237847
time: 1.40 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -878.3933716, 1403.1925049, -2350.9592285, 2398.6582031
1: -1064.8012695, 1553.7296143, -986.9750977, 1434.5900879, -2499.3913574, 2540.7045898
2: -1073.6849365, 1552.4042969, -995.9152832, 1433.4410400, -2507.1250000, 2548.3195801
3: -1301.9499512, 1791.9694824, -1203.8254395, 1656.0020752, -2957.9521484, 2995.7949219
4: -1176.3287354, 1786.4251709, -1097.6263428, 1650.1330566, -2826.4616699, 2883.4372559

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A2_B1_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5444859, upper bound: 2204.5071916
time: 1.37 seconds

## Relational analysis of IS_B2_B2_A1_A2_B1_B2

### Relational analysis result of IS_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5412036, upper bound: 2204.5237847
time: 1.39 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -947.7670898, 1520.2650146, -1075.5012207, 1721.2386475, -2669.0051270, 2595.7658691
1: -1064.8012695, 1553.7296143, -1207.9224854, 1759.2728271, -2824.0742188, 2761.6520996
2: -1073.6849365, 1552.4042969, -1218.3870850, 1757.4587402, -2831.1435547, 2770.7912598
3: -1301.9499512, 1791.9694824, -1473.9526367, 2029.8526611, -3331.8024902, 3265.9218750
4: -1176.3287354, 1786.4251709, -1338.0777588, 2022.5037842, -3198.8322754, 3122.5883789

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5444859, upper bound: 2204.5071916
time: 1.29 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5480241, upper bound: 2204.5237847
time: 1.04 seconds

## BFS IS instance: IS_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -878.4448242, 1403.2790527, -878.3933716, 1403.1925049, -2281.6372070, 2281.6723633
1: -987.0339966, 1434.6789551, -986.9750977, 1434.5900879, -2421.6240234, 2421.6540527
2: -995.9737549, 1433.5321045, -995.9152832, 1433.4410400, -2429.4143066, 2429.4472656
3: -1203.9013672, 1656.1054688, -1203.8254395, 1656.0020752, -2859.9028320, 2859.9309082
4: -1097.6905518, 1650.2377930, -1097.6263428, 1650.1330566, -2747.8237305, 2747.8642578

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5268784
time: 1.44 seconds

## Relational analysis of IS_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5254483
time: 1.11 seconds

## BFS IS instance: IS_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -878.4448242, 1403.2790527, -1075.5012207, 1721.2386475, -2599.6833496, 2478.7800293
1: -987.0339966, 1434.6789551, -1207.9224854, 1759.2728271, -2746.3068848, 2642.6015625
2: -995.9737549, 1433.5321045, -1218.3870850, 1757.4587402, -2753.4326172, 2651.9191895
3: -1203.9013672, 1656.1054688, -1473.9526367, 2029.8526611, -3233.7529297, 3130.0576172
4: -1097.6905518, 1650.2377930, -1338.0777588, 2022.5037842, -3119.4340820, 2988.3154297

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5268786
time: 1.62 seconds

## Relational analysis of IS_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5254483
time: 1.18 seconds

## BFS IS instance: IS_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -1036.0383301, 1658.7559814, -881.4982300, 1407.6596680, -2443.6979980, 2540.2541504
1: -1163.6542969, 1695.2358398, -990.5040894, 1439.2949219, -2602.9487305, 2685.7399902
2: -1173.1274414, 1693.5303955, -999.4396362, 1437.9082031, -2611.0354004, 2692.9699707
3: -1421.6109619, 1955.9685059, -1207.9638672, 1661.4476318, -3083.0585938, 3163.9323730
4: -1286.8170166, 1948.8736572, -1101.4860840, 1655.0926514, -2941.9096680, 3048.5087891

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268730
time: 1.20 seconds

## Relational analysis of IS_B2_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268731
time: 1.44 seconds

## BFS IS instance: IS_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -1075.5012207, 1721.2386475, -881.4982300, 1407.6596680, -2483.1608887, 2602.7368164
1: -1207.9224854, 1759.2728271, -990.5040894, 1439.2949219, -2647.2167969, 2749.7768555
2: -1218.3870850, 1757.4587402, -999.4396362, 1437.9082031, -2656.2946777, 2756.8984375
3: -1473.9526367, 2029.8526611, -1207.9638672, 1661.4476318, -3135.4003906, 3237.8164062
4: -1338.0777588, 2022.5037842, -1101.4860840, 1655.0926514, -2993.1704102, 3123.2612305

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483
time: 1.18 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483
time: 1.41 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.12 seconds
IS_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5079750, upper bound: 2204.5419250
IS_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5004489, upper bound: 2204.5413808
IS_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5093939, upper bound: 2204.5434354
IS_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5088418, upper bound: 2204.5438197
IS_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5041557, upper bound: 2204.5419240
IS_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.4966321, upper bound: 2204.5413574
IS_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5041557, upper bound: 2204.5419240
IS_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.4966321, upper bound: 2204.5413574
IS_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
IS_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
IS_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
IS_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5756188, upper bound: 2204.5756188
IS_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5237848, upper bound: 2204.5480241
IS_B2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5474736, upper bound: 2204.5072023
IS_B2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5512311, upper bound: 2204.5237847
IS_B2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5474736, upper bound: 2204.5072023
IS_B2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5512311, upper bound: 2204.5237847
IS_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5444859, upper bound: 2204.5071916
IS_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5412036, upper bound: 2204.5237847
IS_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5444859, upper bound: 2204.5071916
IS_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5480241, upper bound: 2204.5237847
IS_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5268784
IS_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5254483
IS_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5268786
IS_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5096791, upper bound: 2204.5254483
IS_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268730
IS_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5268731
IS_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483
IS_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.12
Output dim: 3, lower bound: -2204.5063890, upper bound: 2204.5254483

## BFS IS instance: IS_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -739.8888550, 1187.2478027, -738.7090454, 1184.1679688, -1924.0568848, 1925.9566650
1: -831.8164062, 1213.2066650, -830.2191162, 1210.1573486, -2041.9735107, 2043.4256592
2: -838.6459961, 1212.8765869, -837.4002686, 1209.9829102, -2048.6286621, 2050.2768555
3: -1018.1198730, 1399.0572510, -1015.7333984, 1395.7996826, -2413.9189453, 2414.7905273
4: -920.3421631, 1396.0484619, -919.5568237, 1392.3747559, -2312.7167969, 2315.6052246

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5004240, upper bound: 2204.5346996
time: 1.03 seconds

## Relational analysis of IS_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5004240, upper bound: 2204.5448331
time: 1.33 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -746.6878662, 1198.1086426, -745.7267456, 1195.3970947, -1942.0849609, 1943.8353271
1: -839.3478394, 1224.6741943, -838.1273193, 1221.8236084, -2061.1713867, 2062.8015137
2: -846.5509644, 1224.1058350, -845.4638672, 1221.5682373, -2068.1191406, 2069.5695801
3: -1027.0643311, 1412.3280029, -1025.2265625, 1409.3114014, -2436.3757324, 2437.5546875
4: -930.2369385, 1408.7332764, -929.0753174, 1405.5770264, -2335.8137207, 2337.8083496

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5004240, upper bound: 2204.5346996
time: 1.16 seconds

## Relational analysis of IS_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5004240, upper bound: 2204.5448331
time: 1.06 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -739.8997803, 1187.4226074, -936.8412476, 1502.6920166, -2242.5913086, 2124.2639160
1: -831.8004761, 1213.7646484, -1052.5473633, 1535.8186035, -2367.6186523, 2266.3120117
2: -838.8721924, 1213.2177734, -1061.3200684, 1534.4637451, -2373.3356934, 2274.5378418
3: -1017.9088135, 1399.7661133, -1286.9045410, 1771.3000488, -2789.2087402, 2686.6706543
4: -921.8178711, 1396.1634521, -1162.8182373, 1765.7652588, -2687.5827637, 2558.9816895

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4984973, upper bound: 2204.5304653
time: 1.29 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_B2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4982062, upper bound: 2204.5303606
time: 1.34 seconds

## BFS IS instance: IS_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -823.7167969, 1326.8155518, -936.6951904, 1502.2927246, -2326.0095215, 2263.5107422
1: -926.5855713, 1356.8237305, -1052.3698730, 1535.3618164, -2461.9472656, 2409.1933594
2: -934.2412720, 1354.4296875, -1061.1278076, 1534.0087891, -2468.2500000, 2415.5576172
3: -1134.2304688, 1565.1447754, -1286.6811523, 1770.8428955, -2905.0732422, 2851.8259277
4: -1028.6525879, 1557.8830566, -1162.4849854, 1765.2579346, -2793.9101562, 2720.3679199

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_B2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5013588, upper bound: 2204.5346805
time: 1.40 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_B2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4986940, upper bound: 2204.5328976
time: 1.82 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -940.2817993, 1509.0428467, -738.7090454, 1184.1679688, -2124.4497070, 2247.7517090
1: -1056.6192627, 1542.2465820, -830.2191162, 1210.1573486, -2266.7766113, 2372.4658203
2: -1065.0073242, 1540.7747803, -837.4002686, 1209.9829102, -2274.9892578, 2378.1750488
3: -1292.4550781, 1778.4005127, -1015.7333984, 1395.7996826, -2688.2534180, 2794.1335449
4: -1165.8300781, 1773.4591064, -919.5568237, 1392.3747559, -2558.2043457, 2693.0158691

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038358, upper bound: 2204.5444750
time: 1.17 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.5038849, upper bound: 2204.5444745
time: 1.25 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -945.0764160, 1516.7320557, -745.7267456, 1195.3970947, -2140.4736328, 2262.4587402
1: -1061.9445801, 1550.2939453, -838.1273193, 1221.8236084, -2283.7680664, 2388.4213867
2: -1070.6265869, 1548.6778564, -845.4638672, 1221.5682373, -2292.1945801, 2394.1413574
3: -1298.7576904, 1787.7120361, -1025.2265625, 1409.3114014, -2708.0688477, 2812.9384766
4: -1173.0065918, 1782.3546143, -929.0753174, 1405.5770264, -2578.5834961, 2711.4299316

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_B2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4962873, upper bound: 2204.5444422
time: 1.03 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4964179, upper bound: 2204.5443738
time: 1.33 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -940.2817993, 1509.0428467, -938.1754150, 1504.7926025, -2445.0744629, 2447.2182617
1: -1056.6192627, 1542.2465820, -1054.0415039, 1537.7962646, -2594.4155273, 2596.2880859
2: -1065.0073242, 1540.7747803, -1062.7027588, 1536.5505371, -2601.5576172, 2603.4775391
3: -1292.4550781, 1778.4005127, -1288.8828125, 1773.5908203, -3066.0451660, 3067.2827148
4: -1165.8300781, 1773.4591064, -1163.6160889, 1768.3106689, -2934.1406250, 2937.0751953

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B2_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961857, upper bound: 2204.5312182
time: 1.10 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961857, upper bound: 2204.5413574
time: 1.43 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -945.0764160, 1516.7320557, -944.0889893, 1514.2838135, -2459.3603516, 2460.8205566
1: -1061.9445801, 1550.2939453, -1060.6749268, 1547.6190186, -2609.5634766, 2610.9685059
2: -1070.6265869, 1548.6778564, -1069.5040283, 1546.2996826, -2616.9257812, 2618.1813965
3: -1298.7576904, 1787.7120361, -1296.8964844, 1784.9169922, -3083.6748047, 3084.6083984
4: -1173.0065918, 1782.3546143, -1171.6958008, 1779.4077148, -2952.4143066, 2954.0502930

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961857, upper bound: 2204.5312182
time: 1.25 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -2204.4961857, upper bound: 2204.5413574
time: 1.35 seconds

## BFS IS instance: IS_B2_B1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -749.1112671, 1200.8592529, -749.2294312, 1201.0413818, -1950.1525879, 1950.0886230
1: -841.9461060, 1227.4023438, -842.0776367, 1227.5903320, -2069.5361328, 2069.4797363
2: -849.3118896, 1227.1525879, -849.4456177, 1227.3383789, -2076.6503906, 2076.5979004
3: -1029.8669434, 1415.7583008, -1030.0261230, 1415.9755859, -2445.8425293, 2445.7844238
4: -933.3326416, 1411.9925537, -933.4797974, 1412.2065430, -2345.5390625, 2345.4721680

Time for backsubstitution: 2.33 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=2549.14794921875
rel_dist={3: [-2204.582210724847, 2204.5822107248478]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1094.88 seconds
