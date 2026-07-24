## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 3613.31311749156


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031)
1: (-2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305)
2: (-2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000)
3: (-1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141)
4: (-3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883)

## BASE Result
execution time: IAR + LP analysis = 1.67 + 2.04 = 3.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7618777, upper bound: 3614.7618777


# Binary Search by BASE starts (time budget: 1196.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=4653.345703125
rel_dist={0: [-3614.7615518272232, 3614.7615518272232]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=4653.345703125
rel_dist={0: [-3614.75946433006, 3614.75946433006]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=4653.345703125
rel_dist={0: [-3614.757789011659, 3614.757789011659]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=4653.345703125
rel_dist={0: [-3614.7566826429224, 3614.756682642923]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=4653.345703125
rel_dist={0: [-3614.7559892132076, 3614.7559892132085]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=4653.345703125
rel_dist={0: [-3614.7555867120263, 3614.7555867120263]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=4653.345703125
rel_dist={0: [-3614.7553802645316, 3614.7553802645307]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=4653.345703125
rel_dist={0: [-3614.7552770407865, 3614.7552770407856]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=4653.345703125
rel_dist={0: [-3614.7552254289176, 3614.7552254289185]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=4653.345703125
rel_dist={0: [-3614.7551996229913, 3614.755199622992]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=4653.345703125
rel_dist={0: [-3614.7551867114494, 3614.755186711449]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=4653.345703125
rel_dist={0: [-3614.7551802402395, 3614.7551802402395]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=4653.345703125
rel_dist={0: [-3614.755176997902, 3614.755176997902]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=4653.345703125
rel_dist={0: [-3614.7551753768576, 3614.7551753768576]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=4653.345703125
rel_dist={0: [-3614.7551745654505, 3614.755174566577]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=4653.345703125
rel_dist={0: [-3614.755174159827, 3614.755174159827]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=4653.345703125
rel_dist={0: [-3614.7551739681044, 3614.755173968104]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=4653.345703125
rel_dist={0: [-3614.755173863461, 3614.7551740795734]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=4653.345703125
rel_dist={0: [-3614.7551791817696, 3614.755176248431]}

## Binary Search Result
Binary search time: 76.75 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1119.54 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.7414404
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7435825, upper bound: 3614.7435832
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.7414404
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -3614.7435825, upper bound: 3614.7435832

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2497.2197266, 2146.8996582, -4319.6474609, 4389.1464844
1: -1744.2438965, 1855.7540283, -2011.4614258, 2103.5981445, -3847.8420410, 3867.2153320
2: -2557.1499023, 2009.1342773, -2971.3842773, 2280.9895020, -4838.1386719, 4980.5180664
3: -990.0590820, 2595.9711914, -1124.2485352, 2977.7011719, -3967.7600098, 3720.2197266
4: -2815.3046875, 1956.6667480, -3269.0810547, 2222.7011719, -5038.0053711, 5225.7475586

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.1904596
time: 0.79 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484357, upper bound: 3613.6716648
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -2500.9851074, 2149.7963867, -2502.3757324, 2150.9697266, -4651.9545898, 4652.1718750
1: -2014.5847168, 2106.3789062, -2015.7012939, 2107.5605469, -4122.1455078, 4122.0800781
2: -2976.4731445, 2283.9982910, -2978.1022949, 2285.2739258, -5261.7460938, 5262.0996094
3: -1125.7036133, 2982.2333984, -1126.3112793, 2983.9003906, -4109.6040039, 4108.5449219
4: -3274.6416016, 2225.7395020, -3276.4265137, 2226.9489746, -5501.5903320, 5502.1655273

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1925928, upper bound: 3613.6738080
time: 0.84 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6737979, upper bound: 3613.6737986
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.1904596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -3613.0484357, upper bound: 3613.6716648
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -3614.1925928, upper bound: 3613.6738080
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 0, lower bound: -3613.6737979, upper bound: 3613.6737986

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2268.6284180, 1962.2436523, -4134.9912109, 4160.5551758
1: -1744.2438965, 1855.7540283, -1827.9162598, 1924.7995605, -3669.0434570, 3683.6704102
2: -2557.1499023, 2009.1342773, -2706.9870605, 2085.3122559, -4642.4619141, 4716.1201172
3: -990.0590820, 2595.9711914, -1025.1282959, 2719.6457520, -3709.7048340, 3621.0996094
4: -2815.3046875, 1956.6667480, -2976.6870117, 2032.2359619, -4847.5405273, 4933.3535156

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484451, upper bound: 3613.5245842
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0484451, upper bound: 3612.9188905
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2165.7119141, 1886.1955566, -4087.8901367, 3684.4755859, -5671.3876953, 5972.2138672
1: -1738.6062012, 1850.1174316, -3294.9794922, 3627.5119629, -5219.8984375, 5133.2138672
2: -2548.9362793, 2003.0037842, -4968.4643555, 3899.9365234, -6282.3896484, 6906.3403320
3: -986.9508057, 2587.8891602, -1831.7360840, 5011.2729492, -5927.2314453, 4377.7377930
4: -2806.2238770, 1950.7646484, -5436.0039062, 3788.3203125, -6422.0175781, 7330.7817383

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.0463023
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.6716648
time: 0.74 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -2271.4199219, 1964.2563477, -2502.3757324, 2150.9697266, -4422.3891602, 4466.6318359
1: -1830.2175293, 1926.7331543, -2015.7012939, 2107.5605469, -3937.7780762, 3942.4343262
2: -2710.7407227, 2087.3911133, -2978.1022949, 2285.2739258, -4996.0146484, 5065.4916992
3: -1026.1483154, 2722.9445801, -1126.3112793, 2983.9003906, -4010.0488281, 3849.2558594
4: -2980.7939453, 2034.3486328, -3276.4265137, 2226.9489746, -5207.7426758, 5310.7753906

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9210069, upper bound: 3612.9866580
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947112
time: 0.84 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -4060.6059570, 3663.4699707, -2497.2712402, 2147.3784180, -6207.9833984, 5986.9033203
1: -3273.8908691, 3606.7294922, -2011.7054443, 2104.0603027, -5377.9511719, 5473.6933594
2: -4939.3784180, 3877.3940430, -2972.4140625, 2281.4770508, -7191.6923828, 6688.7519531
3: -1818.6785889, 4981.3247070, -1124.3886719, 2978.6611328, -4754.6054688, 6050.8320312
4: -5402.9589844, 3766.9350586, -3270.0480957, 2223.2463379, -7608.6762695, 6867.6860352

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946831, upper bound: 3612.9866486
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947000, upper bound: 3613.3947019
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.0484451, upper bound: 3613.5245842
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.0484451, upper bound: 3612.9188905
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.0463023
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.6716648
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 0, lower bound: -3612.9210069, upper bound: 3612.9866580
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947112
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.3946831, upper bound: 3612.9866486
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 0, lower bound: -3613.3947000, upper bound: 3613.3947019

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2029.9346924, 1766.1118164, -3938.8593750, 3921.8613281
1: -1744.2438965, 1855.7540283, -1635.0686035, 1734.6962891, -3478.9396973, 3490.8227539
2: -2557.1499023, 2009.1342773, -2430.8645020, 1876.3620605, -4433.5117188, 4439.9985352
3: -990.0590820, 2595.9711914, -918.9226685, 2450.8305664, -3440.8894043, 3514.8937988
4: -2815.3046875, 1956.6667480, -2671.9277344, 1829.2399902, -4644.5444336, 4628.5947266

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0472165, upper bound: 3612.7502817
time: 0.75 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484389, upper bound: 3613.5245672
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2165.7119141, 1886.1955566, -4090.8337402, 3686.8488770, -5673.3588867, 5974.6645508
1: -1738.6062012, 1850.1174316, -3297.3720703, 3629.8322754, -5221.9096680, 5135.1918945
2: -2548.9362793, 2003.0037842, -4972.3671875, 3902.3996582, -6284.4887695, 6909.4379883
3: -986.9508057, 2587.8891602, -1832.7989502, 5014.8530273, -5930.3793945, 4378.6147461
4: -2806.2238770, 1950.7646484, -5440.2470703, 3790.7102051, -6424.0703125, 7334.1621094

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6153704, upper bound: 3613.6695967
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6150438, upper bound: 3613.5001244
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2263.2207031, 1957.5310059, -4186.0581055, 3730.1794434, -5845.8500977, 6143.5888672
1: -1823.6866455, 1920.1557617, -3373.2968750, 3671.5925293, -5372.6762695, 5293.4511719
2: -2701.2561035, 2080.2939453, -5074.6147461, 3949.0249023, -6515.0625000, 7143.1655273
3: -1022.5872803, 2713.5651855, -1864.2403564, 5092.6782227, -6084.5161133, 4542.2348633
4: -2970.2231445, 2027.3975830, -5550.1997070, 3836.1955566, -6665.9799805, 7577.3784180

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947133
time: 0.88 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947133
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4059.4133301, 3662.1867676, -2250.3454590, 1947.2305908, -6006.6435547, 5741.1235352
1: -3272.9384766, 3605.4375000, -1813.1789551, 1910.1235352, -5183.0620117, 5275.9306641
2: -4937.9316406, 3876.0070801, -2686.0983887, 2069.4677734, -6983.7387695, 6405.6635742
3: -1818.0600586, 4979.7382812, -1016.8472900, 2699.8176270, -4477.7333984, 5944.0708008
4: -5401.3662109, 3765.5915527, -2953.8735352, 2015.9990234, -7405.3833008, 6554.8442383

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946831, upper bound: 3612.9760415
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946869, upper bound: 3612.9866098
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4062.6835938, 3666.3398438, -4184.2670898, 3728.6977539, -7572.0068359, 7608.4326172
1: -3275.5346680, 3609.7060547, -3371.8955078, 3670.0629883, -6756.6289062, 6777.2973633
2: -4942.0151367, 3880.5754395, -5072.4770508, 3947.3637695, -8615.6123047, 8668.7089844
3: -1819.9547119, 4984.5107422, -1863.3576660, 5090.5214844, -6778.2587891, 6698.2153320
4: -5405.8427734, 3769.9787598, -5547.8950195, 3834.6359863, -8966.9326172, 9030.7382812

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3946870
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3947039
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.37 seconds
IS_A1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.0472165, upper bound: 3612.7502817
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.0484389, upper bound: 3613.5245672
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.6153704, upper bound: 3613.6695967
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.6150438, upper bound: 3613.5001244
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947133
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.9210238, upper bound: 3613.3947133
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.3946831, upper bound: 3612.9760415
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3613.3946869, upper bound: 3612.9866098
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3946870
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3947039

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2023.3359375, 1760.2939453, -3933.0415039, 3915.2626953
1: -1744.2438965, 1855.7540283, -1629.7861328, 1729.0727539, -3473.3161621, 3485.5400391
2: -2557.1499023, 2009.1342773, -2423.2465820, 1870.2154541, -4427.3652344, 4432.3808594
3: -990.0590820, 2595.9711914, -915.8745117, 2443.1425781, -3433.2014160, 3511.8454590
4: -2815.3046875, 1956.6667480, -2663.4672852, 1823.2707520, -4638.5747070, 4620.1337891

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6175070, upper bound: 3613.5242461
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6174883, upper bound: 3613.5242461
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2165.7119141, 1886.1955566, -4114.2641602, 3704.0268555, -5687.0249023, 5999.6142578
1: -1738.6062012, 1850.1174316, -3315.8884277, 3645.3503418, -5234.6313477, 5154.9160156
2: -2548.9362793, 2003.0037842, -4997.7509766, 3918.2749023, -6296.9096680, 6936.8564453
3: -986.9508057, 2587.8891602, -1841.9182129, 5040.5668945, -5956.8339844, 4385.9067383
4: -2806.2238770, 1950.7646484, -5469.8032227, 3807.3759766, -6436.4082031, 7365.7382812

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6174869, upper bound: 3612.9811332
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6175038, upper bound: 3613.3922377
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2165.7119141, 1886.1955566, -4019.4057617, 3619.6508789, -5608.5683594, 5901.8896484
1: -1738.6062012, 1850.1174316, -3240.5986328, 3563.5417480, -5157.7802734, 5076.8686523
2: -2548.9362793, 2003.0037842, -4889.2290039, 3830.9487305, -6215.5908203, 6823.1923828
3: -986.9508057, 2587.8891602, -1798.8720703, 4927.6420898, -5842.3764648, 4345.3681641
4: -2806.2238770, 1950.7646484, -5348.5151367, 3720.9978027, -6357.2910156, 7238.8515625

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6174766, upper bound: 3612.9022832
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6174862, upper bound: 3613.1899909
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2032.5021973, 1767.8198242, -4186.0581055, 3730.1794434, -5618.3520508, 5953.8779297
1: -1637.2110596, 1736.3991699, -3373.2968750, 3671.5925293, -5188.8320312, 5109.6953125
2: -2434.3435059, 1878.1309814, -5074.6147461, 3949.0249023, -6253.6020508, 6947.4174805
3: -919.7875977, 2453.9553223, -1864.2403564, 5092.6782227, -5984.4594727, 4285.9199219
4: -2675.7617188, 1831.0811768, -5550.1997070, 3836.1955566, -6376.9077148, 7381.2807617

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8195212, upper bound: 3613.3946838
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9210027, upper bound: 3613.3946859
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3858.8288574, 3476.9750977, -4186.0581055, 3730.1794434, -7367.3774414, 7427.2763672
1: -3110.0395508, 3427.8459473, -3373.2968750, 3671.5925293, -6593.7617188, 6601.0146484
2: -4695.0322266, 3685.5327148, -5074.6147461, 3949.0249023, -8373.0166016, 8477.8242188
3: -1728.6866455, 4724.4501953, -1864.2403564, 5092.6782227, -6684.7304688, 6452.8906250
4: -5128.0468750, 3577.4528809, -5550.1997070, 3836.1955566, -8692.9257812, 8846.2919922

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209965, upper bound: 3613.0732392
time: 1.34 seconds

## Relational analysis of IS_A2_A1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8767296, upper bound: 3612.9929415
time: 0.83 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -4057.6804199, 3660.3229980, -2033.8878174, 1769.0169678, -5826.6972656, 5526.2773438
1: -3271.5534668, 3603.5625000, -1638.3045654, 1737.5650635, -5009.1181641, 5102.0258789
2: -4935.8295898, 3873.9907227, -2435.9470215, 1879.3935547, -6798.2641602, 6159.0434570
3: -1817.1613770, 4977.4335938, -920.3916626, 2455.5791016, -4236.1142578, 5848.0478516
4: -5399.0551758, 3763.6401367, -2677.5119629, 1832.2690430, -7226.1201172, 6282.0268555

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3937742, upper bound: 3612.4668417
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946789, upper bound: 3612.9760232
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -4065.7246094, 3668.9826660, -3904.8264160, 3541.1750488, -7345.9296875, 7316.2695312
1: -3277.9814453, 3612.2758789, -3146.5612793, 3492.1206055, -6544.8891602, 6538.8120117
2: -4945.5898438, 3883.3540039, -4764.1665039, 3751.3303223, -8382.0712891, 8332.3916016
3: -1821.3348389, 4988.1347656, -1749.9495850, 4810.4135742, -6465.1396484, 6574.3842773
4: -5409.7890625, 3772.7041016, -5208.8427734, 3642.6633301, -8735.8017578, 8661.1875000

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9866316, upper bound: 3612.9865141
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9866316, upper bound: 3612.9866098
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3842.9118652, 3492.4929199, -4184.2670898, 3728.6977539, -7339.7089844, 7415.2895508
1: -3097.1203613, 3444.2551270, -3371.8955078, 3670.0629883, -6568.4628906, 6594.1196289
2: -4690.9912109, 3699.9401855, -5072.4770508, 3947.3637695, -8346.1005859, 8466.3789062
3: -1723.5593262, 4738.4677734, -1863.3576660, 5090.5214844, -6671.7006836, 6440.0712891
4: -5127.6469727, 3592.6191406, -5547.8950195, 3834.6359863, -8668.0654297, 8832.9902344

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9866315, upper bound: 3612.9210071
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9866316, upper bound: 3613.1283603
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6021.9633789, 5527.9223633, -4184.2670898, 3728.6977539, -9391.2226562, 9076.5781250
1: -4849.5991211, 5460.5458984, -3371.8955078, 3670.0629883, -8213.3037109, 8291.8212891
2: -7369.0610352, 5869.3242188, -5072.4770508, 3947.3637695, -10808.7089844, 10246.2753906
3: -2707.7460938, 7437.9379883, -1863.3576660, 5090.5214844, -7503.2143555, 8979.8271484
4: -8038.8769531, 5684.6171875, -5547.8950195, 3834.6359863, -11361.7939453, 10535.6718750

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9047266, upper bound: 3613.3935152
time: 0.70 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9047162, upper bound: 3613.1915617
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.69 seconds
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6175070, upper bound: 3613.5242461
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6174883, upper bound: 3613.5242461
IS_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6174869, upper bound: 3612.9811332
IS_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6175038, upper bound: 3613.3922377
IS_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6174766, upper bound: 3612.9022832
IS_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.6174862, upper bound: 3613.1899909
IS_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.8195212, upper bound: 3613.3946838
IS_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9210027, upper bound: 3613.3946859
IS_A2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9209965, upper bound: 3613.0732392
IS_A2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.8767296, upper bound: 3612.9929415
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3613.3937742, upper bound: 3612.4668417
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3613.3946789, upper bound: 3612.9760232
IS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9866316, upper bound: 3612.9865141
IS_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9866316, upper bound: 3612.9866098
IS_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9866315, upper bound: 3612.9210071
IS_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9866316, upper bound: 3613.1283603
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9047266, upper bound: 3613.3935152
IS_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -3612.9047162, upper bound: 3613.1915617

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2049.6936035, 1780.8721924, -3953.6198730, 3941.6201172
1: -1744.2438965, 1855.7540283, -1650.6102295, 1748.2994385, -3492.5434570, 3506.3642578
2: -2557.1499023, 2009.1342773, -2452.1643066, 1891.1938477, -4448.3437500, 4461.2973633
3: -990.0590820, 2595.9711914, -926.2713013, 2473.0161133, -3463.0749512, 3522.2421875
4: -2815.3046875, 1956.6667480, -2696.8786621, 1842.9656982, -4658.2700195, 4653.5454102

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6165538, upper bound: 3613.5240724
time: 0.79 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6174780, upper bound: 3613.5242380
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -1956.3430176, 1694.6225586, -3867.3698730, 3848.2695312
1: -1744.2438965, 1855.7540283, -1576.6635742, 1664.1124268, -3408.3559570, 3432.4174805
2: -2557.1499023, 2009.1342773, -2345.2800293, 1800.6148682, -4357.7646484, 4354.4140625
3: -990.0590820, 2595.9711914, -883.7121582, 2360.5505371, -3350.6091309, 3479.6833496
4: -2815.3046875, 1956.6667480, -2577.6018066, 1754.5244141, -4569.8286133, 4534.2685547

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6165451, upper bound: 3613.5240724
time: 0.73 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6174693, upper bound: 3613.5242380
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2155.8420410, 1877.9102783, -5965.5454102, 5450.8798828, -7058.3911133, 7705.1518555
1: -1730.7153320, 1842.0518799, -4798.3579102, 5385.4531250, -6654.1420898, 6515.8149414
2: -2537.3911133, 1994.2508545, -7287.9199219, 5786.1547852, -7769.6875000, 8992.2021484
3: -982.5767822, 2576.3366699, -2676.8581543, 7347.0234375, -8101.8461914, 5055.6904297
4: -2793.4101562, 1942.1538086, -7959.1752930, 5606.0366211, -7838.0830078, 9613.2109375

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.2730059, upper bound: 3613.3911481
time: 0.92 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6174966, upper bound: 3613.3922262
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4186.0581055, 3730.1794434, -5374.7109375, 5739.3427734
1: -1441.5299072, 1524.0225830, -3373.2968750, 3671.5925293, -4993.7675781, 4897.3188477
2: -2148.8415527, 1648.9160156, -5074.6147461, 3949.0249023, -5969.1113281, 6717.8681641
3: -806.1847534, 2158.9660645, -1864.2403564, 5092.6782227, -5871.3969727, 3991.8305664
4: -2360.2592773, 1605.6569824, -5550.1997070, 3836.1955566, -6062.3540039, 7155.8564453

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3945422, upper bound: 3613.3946658
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3945335, upper bound: 3613.1924186
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -4186.0581055, 3730.1794434, -5574.7729492, 5917.1362305
1: -1603.6752930, 1700.3911133, -3373.2968750, 3671.5925293, -5153.7353516, 5073.6875000
2: -2385.5007324, 1838.9254150, -5074.6147461, 3949.0249023, -6202.5258789, 6906.4501953
3: -900.0618896, 2404.4448242, -1864.2403564, 5092.6782227, -5965.3105469, 4233.5693359
4: -2622.1757812, 1793.2502441, -5550.1997070, 3836.1955566, -6321.0170898, 7343.4492188

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5266992, upper bound: 3613.3946755
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5266905, upper bound: 3613.1924283
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4057.6245117, 3660.2626953, -1789.6997070, 1554.3094482, -5611.9340820, 5282.4555664
1: -3271.5090332, 3603.5021973, -1442.5327148, 1525.0668945, -4796.5756836, 4906.8222656
2: -4935.7617188, 3873.9255371, -2150.3056641, 1650.0454102, -6568.4863281, 5874.3583984
3: -1817.1323242, 4977.3598633, -806.7396851, 2160.4394531, -3941.8493652, 5734.8388672
4: -5398.9814453, 3763.5776367, -2361.8547363, 1606.7463379, -6999.8012695, 5967.2631836

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866300, upper bound: 3613.3945399
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866300, upper bound: 3613.3945426
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4057.5842285, 3660.2194824, -1991.7487793, 1732.1446533, -5789.7290039, 5482.4692383
1: -3271.4770508, 3603.4582520, -1604.6556396, 1701.4029541, -4972.8798828, 5066.7290039
2: -4935.7124023, 3873.8791504, -2386.9431152, 1840.0159912, -6757.0058594, 6107.7148438
3: -1817.1115723, 4977.3051758, -900.5989380, 2405.9113770, -4183.5668945, 5828.7070312
4: -5398.9272461, 3763.5319824, -2623.7451172, 1794.3088379, -7185.6015625, 6225.8662109

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946736, upper bound: 3613.5266994
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1924265, upper bound: 3613.5266906
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6021.8437500, 5527.7919922, -4207.0458984, 3746.7861328, -9404.0937500, 9099.3017578
1: -4849.5019531, 5460.4155273, -3389.9304199, 3686.7707520, -8225.5341797, 8309.6630859
2: -7368.9150391, 5869.1826172, -5097.8564453, 3964.5322266, -10820.3466797, 10271.2851562
3: -2707.6833496, 7437.7758789, -1872.2589111, 5116.9790039, -7529.3627930, 8986.6269531
4: -8038.7148438, 5684.4814453, -5577.7045898, 3852.5622559, -11373.5175781, 10564.8476562

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7600551, upper bound: 3613.0368708
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7600553, upper bound: 3613.3935144
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.88 seconds
IS_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.6165538, upper bound: 3613.5240724
IS_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.6174780, upper bound: 3613.5242380
IS_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.6165451, upper bound: 3613.5240724
IS_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.6174693, upper bound: 3613.5242380
IS_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3611.2730059, upper bound: 3613.3911481
IS_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.6174966, upper bound: 3613.3922262
IS_A2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.3945422, upper bound: 3613.3946658
IS_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.3945335, upper bound: 3613.1924186
IS_A2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.5266992, upper bound: 3613.3946755
IS_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.5266905, upper bound: 3613.1924283
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.9866300, upper bound: 3613.3945399
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.9866300, upper bound: 3613.3945426
IS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.3946736, upper bound: 3613.5266994
IS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3613.1924265, upper bound: 3613.5266906
IS_A2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.7600551, upper bound: 3613.0368708
IS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 0, lower bound: -3612.7600553, upper bound: 3613.3935144

## BFS IS instance: IS_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1938.7265625, 1684.0543213, -2049.6936035, 1780.8721924, -3719.5983887, 3733.7478027
1: -1555.8155518, 1651.3876953, -1650.6102295, 1748.2994385, -3304.1149902, 3301.9978027
2: -2281.1577148, 1789.5345459, -2452.1643066, 1891.1938477, -4172.3515625, 4241.6987305
3: -881.9197388, 2309.5590820, -926.2713013, 2473.0161133, -3354.9357910, 3235.8298340
4: -2510.6838379, 1741.0147705, -2696.8786621, 1842.9656982, -4353.6489258, 4437.8935547

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9795672, upper bound: 3613.2769570
time: 0.86 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9154866, upper bound: 3613.2771106
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -2129.9020996, 1855.5412598, -2049.6936035, 1780.8721924, -3910.7739258, 3905.2343750
1: -1709.9429932, 1820.0081787, -1650.6102295, 1748.2994385, -3458.2424316, 3470.6181641
2: -2506.8229980, 1970.3999023, -2452.1643066, 1891.1938477, -4398.0166016, 4422.5634766
3: -970.4000244, 2545.3256836, -926.2713013, 2473.0161133, -3443.4160156, 3471.5966797
4: -2759.9887695, 1919.1589355, -2696.8786621, 1842.9656982, -4602.9531250, 4616.0375977

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9795672, upper bound: 3613.2772762
time: 0.74 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9154866, upper bound: 3613.2771670
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1938.7265625, 1684.0543213, -1956.3430176, 1694.6225586, -3633.3486328, 3640.3972168
1: -1555.8155518, 1651.3876953, -1576.6635742, 1664.1124268, -3219.9274902, 3228.0512695
2: -2281.1577148, 1789.5345459, -2345.2800293, 1800.6148682, -4081.7724609, 4134.8144531
3: -881.9197388, 2309.5590820, -883.7121582, 2360.5505371, -3242.4702148, 3193.2712402
4: -2510.6838379, 1741.0147705, -2577.6018066, 1754.5244141, -4265.2075195, 4318.6166992

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9795585, upper bound: 3613.2769570
time: 0.81 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9154779, upper bound: 3613.2771106
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2129.9020996, 1855.5412598, -1956.3430176, 1694.6225586, -3824.5244141, 3811.8835449
1: -1709.9429932, 1820.0081787, -1576.6635742, 1664.1124268, -3374.0551758, 3396.6718750
2: -2506.8229980, 1970.3999023, -2345.2800293, 1800.6148682, -4307.4379883, 4315.6796875
3: -970.4000244, 2545.3256836, -883.7121582, 2360.5505371, -3330.9506836, 3429.0378418
4: -2759.9887695, 1919.1589355, -2577.6018066, 1754.5244141, -4514.5122070, 4496.7607422

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9795585, upper bound: 3613.2772762
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9154779, upper bound: 3613.2771670
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2235.2900391, 1957.7905273, -5968.4672852, 5453.5551758, -7138.7568359, 7779.3178711
1: -1793.8936768, 1920.2342529, -4800.7568359, 5388.0439453, -6718.3959961, 6589.4716797
2: -2629.9228516, 2079.1403809, -7292.0561523, 5789.0473633, -7862.7338867, 9071.9541016
3: -1019.3221436, 2674.8505859, -2678.0366211, 7350.8681641, -8141.1386719, 5152.2299805
4: -2895.8215332, 2025.9530029, -7963.6948242, 5608.8359375, -7941.1850586, 9691.6376953

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2525925, upper bound: 3613.2196459
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2526840, upper bound: 3613.2796617
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -2149.7707520, 1872.8029785, -5964.2509766, 5449.5366211, -7051.2221680, 7698.1298828
1: -1725.8635254, 1837.0974121, -4797.3344727, 5384.1108398, -6648.1430664, 6509.3398438
2: -2530.3732910, 1988.8936768, -7286.3725586, 5784.7270508, -7761.4282227, 8984.7041016
3: -979.9031372, 2569.2810059, -2676.2380371, 7345.3378906, -8097.3452148, 5047.9853516
4: -2785.6091309, 1936.9130859, -7957.4707031, 5604.6630859, -7829.1093750, 9605.6357422

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6165500, upper bound: 3613.3920525
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6174742, upper bound: 3613.3922181
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4210.3022461, 3749.5021973, -5388.7700195, 5763.5854492
1: -1441.5299072, 1524.0225830, -3392.5166016, 3689.5263672, -5007.2045898, 4916.5380859
2: -2148.8415527, 1648.9160156, -5101.7768555, 3967.5014648, -5982.0468750, 6744.6757812
3: -806.1847534, 2158.9660645, -1873.7744141, 5120.8842773, -5899.2851562, 3999.3864746
4: -2360.2592773, 1605.6569824, -5581.9750977, 3855.4123535, -6075.3500977, 7187.6318359

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3924048, upper bound: 3613.0393472
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3924056, upper bound: 3613.3946658
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4116.9589844, 3665.5043945, -5311.7016602, 5670.2421875
1: -1441.5299072, 1524.0225830, -3318.1269531, 3608.3474121, -4932.0795898, 4842.1494141
2: -2148.8415527, 1648.9160156, -4994.3994141, 3880.8623047, -5902.7143555, 6633.9453125
3: -806.1847534, 2158.9660645, -1832.8724365, 5008.6264648, -5786.2792969, 3960.0456543
4: -2360.2592773, 1605.6569824, -5461.3847656, 3768.9030762, -5997.1464844, 7067.0419922

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3920972, upper bound: 3612.6174678
time: 0.89 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3920980, upper bound: 3613.1924186
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -4210.3022461, 3749.5021973, -5588.8315430, 5941.3798828
1: -1603.6752930, 1700.3911133, -3392.5166016, 3689.5263672, -5167.1723633, 5092.9077148
2: -2385.5007324, 1838.9254150, -5101.7768555, 3967.5014648, -6215.4614258, 6933.2583008
3: -900.0618896, 2404.4448242, -1873.7744141, 5120.8842773, -5993.1987305, 4241.1250000
4: -2622.1757812, 1793.2502441, -5581.9750977, 3855.4123535, -6334.0131836, 7375.2246094

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6030687, upper bound: 3613.3933603
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5266879, upper bound: 3613.3946673
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -4116.9589844, 3665.5043945, -5511.7636719, 5848.0366211
1: -1603.6752930, 1700.3911133, -3318.1269531, 3608.3474121, -5092.0483398, 5018.5180664
2: -2385.5007324, 1838.9254150, -4994.3994141, 3880.8623047, -6136.1289062, 6822.5273438
3: -900.0618896, 2404.4448242, -1832.8724365, 5008.6264648, -5880.1923828, 4201.7846680
4: -2622.1757812, 1793.2502441, -5461.3847656, 3768.9030762, -6255.8100586, 7254.6025391

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5253190, upper bound: 3611.8082872
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5266810, upper bound: 3613.1924202
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -3837.0563965, 3486.1735840, -1789.6997070, 1554.3094482, -5391.3652344, 5088.6777344
1: -3092.4372559, 3437.8784180, -1442.5327148, 1525.0668945, -4617.5039062, 4723.1367188
2: -4683.8974609, 3693.0834961, -2150.3056641, 1650.0454102, -6297.9667969, 5671.4028320
3: -1720.5197754, 4730.6855469, -806.7396851, 2160.4394531, -3834.8193359, 5475.8750000
4: -5119.8427734, 3586.0124512, -2361.8547363, 1606.7463379, -6699.7998047, 5768.8081055

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9835574, upper bound: 3613.3945399
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9047074, upper bound: 3613.3945329
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5843.0512695, 5366.2119141, -1789.6997070, 1554.3094482, -7303.4877930, 6616.6005859
1: -4704.1357422, 5300.9726562, -1442.5327148, 1525.0668945, -6137.8154297, 6286.3569336
2: -7149.7797852, 5696.5449219, -2150.3056641, 1650.0454102, -8563.0771484, 7307.0366211
3: -2629.5771484, 7214.8666992, -806.7396851, 2160.4394531, -4598.2495117, 7813.4555664
4: -7799.6826172, 5516.6000977, -2361.8547363, 1606.7463379, -9177.1074219, 7331.6484375

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9835574, upper bound: 3613.3945426
time: 0.97 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9047074, upper bound: 3613.3945339
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4067.3071289, 3666.9631348, -1991.7487793, 1732.1446533, -5799.4516602, 5485.5209961
1: -3279.3613281, 3608.7050781, -1604.6556396, 1701.4029541, -4980.7641602, 5068.9414062
2: -4945.5136719, 3878.7243652, -2386.9431152, 1840.0159912, -6768.6342773, 6108.8789062
3: -1820.4641113, 4987.3916016, -900.5989380, 2405.9113770, -4184.6796875, 5839.4169922
4: -5411.1875000, 3769.4685059, -2623.7451172, 1794.3088379, -7199.8095703, 6227.3256836

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3933584, upper bound: 3612.6030693
time: 0.83 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3946655, upper bound: 3613.5266881
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -3986.8007812, 3593.5764160, -1991.7487793, 1732.1446533, -5718.9453125, 5418.6943359
1: -3215.3093262, 3537.6296387, -1604.6556396, 1701.4029541, -4916.7124023, 5003.4233398
2: -4853.5830078, 3802.9924316, -2386.9431152, 1840.0159912, -6672.2270508, 6039.8720703
3: -1783.4815674, 4890.9497070, -900.5989380, 2405.9113770, -4150.5849609, 5741.8798828
4: -5308.3466797, 3694.5129395, -2623.7451172, 1794.3088379, -7091.9814453, 6160.2685547

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1911113, upper bound: 3612.6030622
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1924184, upper bound: 3613.5266811
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -6021.8437500, 5527.7919922, -4205.3129883, 3745.3464355, -9402.5673828, 9097.5351562
1: -4849.5019531, 5460.4155273, -3388.5356445, 3685.3740234, -8224.0585938, 8308.2441406
2: -7368.9155273, 5869.1826172, -5095.8505859, 3963.0239258, -10818.7382812, 10269.2314453
3: -2707.6833496, 7437.7758789, -1871.5095215, 5114.9458008, -7527.2924805, 8985.8330078
4: -8038.7153320, 5684.4814453, -5575.4780273, 3851.0932617, -11371.9560547, 10562.5703125

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6146783, upper bound: 3613.1658211
time: 0.89 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7600553, upper bound: 3613.3935030
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.88 seconds
IS_A1_B1_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9795672, upper bound: 3613.2769570
IS_A1_B1_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9154866, upper bound: 3613.2771106
IS_A1_B1_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9795672, upper bound: 3613.2772762
IS_A1_B1_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9154866, upper bound: 3613.2771670
IS_A1_B1_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9795585, upper bound: 3613.2769570
IS_A1_B1_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9154779, upper bound: 3613.2771106
IS_A1_B1_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9795585, upper bound: 3613.2772762
IS_A1_B1_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.9154779, upper bound: 3613.2771670
IS_A1_B2_B2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.2525925, upper bound: 3613.2196459
IS_A1_B2_B2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3611.2526840, upper bound: 3613.2796617
IS_A1_B2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.6165500, upper bound: 3613.3920525
IS_A1_B2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.6174742, upper bound: 3613.3922181
IS_A2_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3924048, upper bound: 3613.0393472
IS_A2_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3924056, upper bound: 3613.3946658
IS_A2_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3920972, upper bound: 3612.6174678
IS_A2_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3920980, upper bound: 3613.1924186
IS_A2_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.6030687, upper bound: 3613.3933603
IS_A2_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.5266879, upper bound: 3613.3946673
IS_A2_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.5253190, upper bound: 3611.8082872
IS_A2_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.5266810, upper bound: 3613.1924202
IS_A2_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.9835574, upper bound: 3613.3945399
IS_A2_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.9047074, upper bound: 3613.3945329
IS_A2_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.9835574, upper bound: 3613.3945426
IS_A2_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.9047074, upper bound: 3613.3945339
IS_A2_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3933584, upper bound: 3612.6030693
IS_A2_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.3946655, upper bound: 3613.5266881
IS_A2_A2_B1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.1911113, upper bound: 3612.6030622
IS_A2_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3613.1924184, upper bound: 3613.5266811
IS_A2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.6146783, upper bound: 3613.1658211
IS_A2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 0, lower bound: -3612.7600553, upper bound: 3613.3935030

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -1915.8194580, 1665.0987549, -5931.5107422, 5423.1655273, -6792.0551758, 7458.6796875
1: -1537.5178223, 1632.7774658, -4771.0380859, 5357.9721680, -6435.5278320, 6279.3613281
2: -2254.5195312, 1769.3609619, -7246.0122070, 5755.7211914, -7458.5141602, 8726.9462891
3: -871.8582153, 2283.0390625, -2662.5112305, 7306.3784180, -7952.1811523, 4749.2265625
4: -2481.1389160, 1721.4637451, -7913.1245117, 5576.3261719, -7497.9243164, 9348.7460938

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5723179, upper bound: 3613.2203955
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5725498, upper bound: 3613.2804114
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -2106.0563965, 1835.7084961, -5957.2504883, 5443.3623047, -7000.6870117, 7651.8002930
1: -1690.8659668, 1800.6257324, -4791.7509766, 5378.0166016, -6606.3759766, 6465.2900391
2: -2479.0529785, 1949.3743896, -7277.8906250, 5778.0927734, -7702.4829102, 8935.5419922
3: -959.8629761, 2517.6696777, -2673.1513672, 7336.6796875, -8069.7216797, 4990.8925781
4: -2729.1818848, 1898.6842041, -7948.2246094, 5598.2563477, -7765.0825195, 9556.2441406

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5728907, upper bound: 3613.2206297
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5731226, upper bound: 3613.2806455
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -3892.0253906, 3518.0207520, -5151.9770508, 5445.3095703
1: -1441.5299072, 1524.0225830, -3131.6613770, 3472.1547852, -4784.9057617, 4655.6831055
2: -2148.8415527, 1648.9160156, -4673.8378906, 3741.0480957, -5739.0288086, 6322.7539062
3: -806.1847534, 2158.9660645, -1775.8906250, 4742.7900391, -5529.4545898, 3894.9711914
4: -2360.2592773, 1605.6569824, -5112.8828125, 3627.9084473, -5830.0141602, 6718.5395508

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2361247, upper bound: 3612.0378441
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2361285, upper bound: 3612.7338801
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4208.7470703, 3748.2109375, -5387.3666992, 5762.0302734
1: -1441.5299072, 1524.0225830, -3391.2639160, 3688.2768555, -5005.8549805, 4915.2866211
2: -2148.8415527, 1648.9160156, -5099.9887695, 3966.1494141, -5980.5717773, 6742.8232422
3: -806.1847534, 2158.9660645, -1873.0988770, 5119.0659180, -5897.4165039, 3998.6621094
4: -2360.2592773, 1605.6569824, -5579.9882812, 3854.0974121, -6073.9135742, 7185.6455078

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3919810, upper bound: 3613.2231258
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3922106, upper bound: 3613.2831415
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -3804.6166992, 3439.4851074, -5080.7451172, 5357.9013672
1: -1441.5299072, 1524.0225830, -3062.0456543, 3396.3466797, -4715.3632812, 4586.0678711
2: -2148.8415527, 1648.9160156, -4573.2197266, 3660.8818359, -5666.5063477, 6222.1357422
3: -806.1847534, 2158.9660645, -1738.2762451, 4637.6718750, -5424.3295898, 3858.8737793
4: -2360.2592773, 1605.6569824, -4999.2993164, 3547.1049805, -5757.9907227, 6604.9560547

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3910105, upper bound: 3611.2729681
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3920804, upper bound: 3612.6174606
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4115.3471680, 3664.1657715, -5310.2524414, 5668.6308594
1: -1441.5299072, 1524.0225830, -3316.8300781, 3607.0505371, -4930.6811523, 4840.8515625
2: -2148.8415527, 1648.9160156, -4992.5527344, 3879.4641113, -5901.1933594, 6632.0283203
3: -806.1847534, 2158.9660645, -1832.1707764, 5006.7402344, -5784.3398438, 3959.2963867
4: -2360.2592773, 1605.6569824, -5459.3300781, 3767.5402832, -5995.6616211, 7064.9873047

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3910113, upper bound: 3611.8082774
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3920812, upper bound: 3613.1924115
time: 0.70 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2089.6127930, 1831.0540771, -4209.7631836, 3749.0412598, -5685.1196289, 6040.8173828
1: -1682.4387207, 1796.9393311, -3392.0820312, 3689.0764160, -5243.5620117, 5189.0214844
2: -2501.8356934, 1944.2260742, -5101.1494141, 3967.0153809, -6327.0400391, 7025.2377930
3: -946.0217896, 2526.3615723, -1873.5330811, 5120.2421875, -6036.3652344, 4357.8139648
4: -2750.0595703, 1895.3664551, -5581.2817383, 3854.9387207, -6457.5742188, 7466.3364258

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6009227, upper bound: 3613.0380417
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6009240, upper bound: 3613.3933603
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1983.0423584, 1724.5026855, -4210.3022461, 3749.5021973, -5581.3291016, 5934.8046875
1: -1597.6961670, 1694.0699463, -3392.5166016, 3689.5263672, -5161.1337891, 5086.5864258
2: -2376.8635254, 1832.0233154, -5101.7768555, 3967.5014648, -6206.6826172, 6925.5009766
3: -896.6072388, 2395.7373047, -1873.7744141, 5120.8842773, -5989.4204102, 4232.2246094
4: -2612.5957031, 1786.4636230, -5581.9750977, 3855.4123535, -6324.3300781, 7368.4384766

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2797137, upper bound: 3612.5300047
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2797254, upper bound: 3613.0667628
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1989.9539795, 1730.5888672, -4213.5312500, 3767.7465820, -5596.1748047, 5944.1196289
1: -1603.2304688, 1699.9106445, -3395.3032227, 3708.0187988, -5177.1577148, 5095.2133789
2: -2384.8439941, 1838.4045410, -5109.1562500, 3989.5222168, -6227.9121094, 6932.7026367
3: -899.8075562, 2403.7790527, -1879.4097900, 5130.7924805, -5997.3071289, 4245.6811523
4: -2621.4545898, 1792.7440186, -5585.9731445, 3876.1411133, -6343.5273438, 7375.6064453

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2781930, upper bound: 3611.5635376
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5231678, upper bound: 3611.2729767
time: 0.89 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5231680, upper bound: 3611.8082872
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -4111.2578125, 3660.6984863, -5505.3862305, 5842.3359375
1: -1603.6752930, 1700.3911133, -3313.5341797, 3603.7492676, -5086.1049805, 5013.9252930
2: -2385.5007324, 1838.9254150, -4987.8647461, 3875.8947754, -6129.4926758, 6815.4291992
3: -900.0618896, 2404.4448242, -1830.3489990, 5001.9877930, -5873.0478516, 4198.5776367
4: -2622.1757812, 1793.2502441, -5454.0888672, 3763.9772949, -6249.2016602, 7246.7490234

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5242377, upper bound: 3612.6174693
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5242380, upper bound: 3613.1924202
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -3845.9035645, 3492.2597656, -1789.6997070, 1554.3094482, -5400.2128906, 5090.7358398
1: -3099.6030273, 3442.3327637, -1442.5327148, 1525.0668945, -4624.6694336, 4724.1923828
2: -4692.5229492, 3697.0458984, -2150.3056641, 1650.0454102, -6308.6333008, 5671.1357422
3: -1723.4627686, 4739.6284180, -806.7396851, 2160.4394531, -3835.7990723, 5485.5844727
4: -5130.8085938, 3591.0749512, -2361.8547363, 1606.7463379, -6712.8413086, 5768.9443359

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3114206, upper bound: 3613.2382525
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5618716, upper bound: 3613.2382618
time: 0.73 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -3766.7919922, 3419.7233887, -1789.6997070, 1554.3094482, -5321.1010742, 5024.8466797
1: -3036.7458496, 3372.2465820, -1442.5327148, 1525.0668945, -4561.8125000, 4659.8447266
2: -4602.5317383, 3622.5043945, -2150.3056641, 1650.0454102, -6213.2348633, 5603.6811523
3: -1687.0184326, 4644.7900391, -806.7396851, 2160.4394531, -3802.2568359, 5389.1523438
4: -5030.0473633, 3517.2265625, -2361.8547363, 1606.7463379, -6606.2539062, 5703.2480469

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9047003, upper bound: 3613.3945216
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -5829.7641602, 5352.7568359, -1789.6997070, 1554.3094482, -7291.4892578, 6599.6352539
1: -4693.0844727, 5287.2465820, -1442.5327148, 1525.0668945, -6127.9663086, 6269.3793945
2: -7132.1376953, 5680.8759766, -2150.3056641, 1650.0454102, -8547.4726562, 7287.2797852
3: -2622.9379883, 7197.2265625, -806.7396851, 2160.4394531, -4589.6635742, 7796.8935547
4: -7782.1025391, 5502.5141602, -2361.8547363, 1606.7463379, -9161.3427734, 7312.9985352

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5300008, upper bound: 3613.2382521
time: 0.83 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0667573, upper bound: 3613.2382661
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -5770.8466797, 5298.7988281, -1789.6997070, 1554.3094482, -7228.3911133, 6549.9223633
1: -4646.5073242, 5234.4062500, -1442.5327148, 1525.0668945, -6077.3559570, 6220.4301758
2: -7066.0180664, 5624.9755859, -2150.3056641, 1650.0454102, -8473.8818359, 7236.2871094
3: -2595.4965820, 7127.1879883, -806.7396851, 2160.4394531, -4564.2270508, 7723.4941406
4: -7706.8969727, 5446.2475586, -2361.8547363, 1606.7463379, -9078.3339844, 7262.6733398

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1924080, upper bound: 3613.3945244
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4066.7573242, 3666.4936523, -2090.7937012, 1832.0306396, -5898.7875977, 5581.7490234
1: -3278.9201660, 3608.2434082, -1683.3825684, 1797.9187012, -5076.8383789, 5145.2915039
2: -4944.8676758, 3878.2277832, -2503.2128906, 1945.2852783, -6860.5947266, 6220.3911133
3: -1820.2193604, 4986.7363281, -946.5480347, 2527.7529297, -4301.2827148, 5882.5761719
4: -5410.4755859, 3768.9851074, -2751.5642090, 1896.4006348, -7287.5483398, 6350.8217773

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2218601, upper bound: 3612.6027124
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2818761, upper bound: 3612.6029443
time: 0.75 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4067.0693359, 3666.7084961, -1984.2878418, 1725.5747070, -5792.6440430, 5477.8125000
1: -3279.1716309, 3608.4494629, -1598.6805420, 1695.0849609, -4974.2563477, 5062.6860352
2: -4945.2250977, 3878.4499512, -2378.3122559, 1833.1173096, -6760.6005859, 6099.8740234
3: -1820.3411865, 4987.0766602, -897.1458740, 2397.2082520, -4175.6782227, 5835.3422852
4: -5410.8701172, 3769.2019043, -2614.1726074, 1787.5251465, -7191.8212891, 6217.4306641

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5300040, upper bound: 3613.2382423
time: 0.80 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0667611, upper bound: 3613.2797255
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3986.5727539, 3593.3315430, -1984.2878418, 1725.5747070, -5712.1474609, 5410.9941406
1: -3215.1269531, 3537.3828125, -1598.6805420, 1695.0849609, -4910.2119141, 4997.1767578
2: -4853.3061523, 3802.7277832, -2378.3122559, 1833.1173096, -6664.2060547, 6030.8774414
3: -1783.3632812, 4890.6464844, -897.1458740, 2397.2082520, -4141.5878906, 5737.8168945
4: -5308.0429688, 3694.2561035, -2614.1726074, 1787.5251465, -7084.0053711, 6150.3818359

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5174760, upper bound: 3613.2797138
time: 0.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5269326, upper bound: 3613.2797194
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -6021.7944336, 5527.7392578, -4180.1640625, 3724.4790039, -9377.0058594, 9070.6806641
1: -4849.4633789, 5460.3623047, -3368.2951660, 3665.2407227, -8199.6533203, 8286.2138672
2: -7368.8574219, 5869.1259766, -5066.7084961, 3941.2128906, -10792.8076172, 10237.0546875
3: -2707.6579590, 7437.7104492, -1860.2724609, 5085.5952148, -7494.3593750, 8974.1669922
4: -8038.6503906, 5684.4272461, -5543.1694336, 3829.7854004, -11345.8212891, 10527.1552734

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8767289, upper bound: 3612.8856013
time: 0.72 seconds

## Relational analysis of IS_A2_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8767289, upper bound: 3613.2104436
time: 0.87 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.98 seconds
IS_A1_B2_B2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5723179, upper bound: 3613.2203955
IS_A1_B2_B2_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5725498, upper bound: 3613.2804114
IS_A1_B2_B2_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5728907, upper bound: 3613.2206297
IS_A1_B2_B2_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5731226, upper bound: 3613.2806455
IS_A2_A1_B2_A1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2361247, upper bound: 3612.0378441
IS_A2_A1_B2_A1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2361285, upper bound: 3612.7338801
IS_A2_A1_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3919810, upper bound: 3613.2231258
IS_A2_A1_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3922106, upper bound: 3613.2831415
IS_A2_A1_B2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3910105, upper bound: 3611.2729681
IS_A2_A1_B2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3920804, upper bound: 3612.6174606
IS_A2_A1_B2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3910113, upper bound: 3611.8082774
IS_A2_A1_B2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.3920812, upper bound: 3613.1924115
IS_A2_A1_B2_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.6009227, upper bound: 3613.0380417
IS_A2_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.6009240, upper bound: 3613.3933603
IS_A2_A1_B2_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2797137, upper bound: 3612.5300047
IS_A2_A1_B2_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2797254, upper bound: 3613.0667628
IS_A2_A1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.5231678, upper bound: 3611.2729767
IS_A2_A1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.5231680, upper bound: 3611.8082872
IS_A2_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.5242377, upper bound: 3612.6174693
IS_A2_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.5242380, upper bound: 3613.1924202
IS_A2_A2_B1_B1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.3114206, upper bound: 3613.2382525
IS_A2_A2_B1_B1_B1_A1_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5618716, upper bound: 3613.2382618
IS_A2_A2_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
IS_A2_A2_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.9047003, upper bound: 3613.3945216
IS_A2_A2_B1_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5300008, upper bound: 3613.2382521
IS_A2_A2_B1_B1_B1_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.0667573, upper bound: 3613.2382661
IS_A2_A2_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
IS_A2_A2_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.1924080, upper bound: 3613.3945244
IS_A2_A2_B1_B1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2218601, upper bound: 3612.6027124
IS_A2_A2_B1_B1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.2818761, upper bound: 3612.6029443
IS_A2_A2_B1_B1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5300040, upper bound: 3613.2382423
IS_A2_A2_B1_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3613.0667611, upper bound: 3613.2797255
IS_A2_A2_B1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5174760, upper bound: 3613.2797138
IS_A2_A2_B1_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.5269326, upper bound: 3613.2797194
IS_A2_A2_B2_A2_B1_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.8767289, upper bound: 3612.8856013
IS_A2_A2_B2_A2_B1_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.98
Output dim: 0, lower bound: -3612.8767289, upper bound: 3613.2104436

## BFS IS instance: IS_A2_A1_B2_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4155.6694336, 3700.3916016, -5330.3388672, 5708.9531250
1: -1441.5299072, 1524.0225830, -3348.5026855, 3640.7321777, -4950.8032227, 4872.5253906
2: -2148.8415527, 1648.9160156, -5039.0688477, 3915.8454590, -5921.5200195, 6674.0883789
3: -806.1847534, 2158.9660645, -1847.6013184, 5054.5366211, -5827.1010742, 3970.7490234
4: -2360.2592773, 1605.6569824, -5511.5649414, 3804.2490234, -6014.6318359, 7114.6499023

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2374698, upper bound: 3612.3003331
time: 0.89 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2377862, upper bound: 3612.9741242
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4196.3862305, 3737.3549805, -5376.0214844, 5749.6708984
1: -1441.5299072, 1524.0225830, -3381.3188477, 3677.5395508, -4994.7275391, 4905.3403320
2: -2148.8415527, 1648.9160156, -5085.4819336, 3954.6193848, -5968.5239258, 6728.0156250
3: -806.1847534, 2158.9660645, -1867.4978027, 5104.1430664, -5882.3129883, 3992.9619141
4: -2360.2592773, 1605.6569824, -5563.9941406, 3842.9626465, -6062.2246094, 7169.6513672

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381314, upper bound: 3612.4516718
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381393, upper bound: 3612.9618775
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1787.7852783, 1552.7274170, -3893.2695312, 3533.3742676, -5157.2866211, 5445.9960938
1: -1441.0001221, 1523.4553223, -3132.1398926, 3487.8605957, -4792.7197266, 4655.5952148
2: -2148.0627441, 1648.3005371, -4677.8745117, 3760.3027344, -5750.0874023, 6326.0859375
3: -805.8786621, 2158.1826172, -1780.7159424, 4748.9350586, -5530.8491211, 3898.6811523
4: -2359.4020996, 1605.0545654, -5113.2617188, 3645.7424316, -5837.8286133, 6718.3159180

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2346469, upper bound: 3611.0308015
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8440912, upper bound: 3611.0726456
time: 0.79 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8088106, upper bound: 3611.2548844
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3910105, upper bound: 3611.2729681
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -3799.2736816, 3435.0346680, -5074.7851562, 5352.5576172
1: -1441.5299072, 1524.0225830, -3057.7436523, 3392.0942383, -4709.8129883, 4581.7651367
2: -2148.8415527, 1648.9160156, -4567.0595703, 3656.3027344, -5660.3212891, 6215.9755859
3: -806.1847534, 2158.9660645, -1735.9165039, 4631.4780273, -5417.6806641, 3855.8569336
4: -2360.2592773, 1605.6569824, -4992.4560547, 3542.5493164, -5751.8139648, 6598.1132812

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2358041, upper bound: 3611.9795480
time: 0.93 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2356948, upper bound: 3611.9154692
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -1787.7852783, 1552.7274170, -4211.9047852, 3766.3974609, -5394.5439453, 5764.6323242
1: -1441.0001221, 1523.4553223, -3393.9875488, 3706.7153320, -5015.6933594, 4917.4428711
2: -2148.0627441, 1648.3005371, -5107.2866211, 3988.1188965, -5992.8398438, 6742.1015625
3: -805.8786621, 2158.1826172, -1878.7052002, 5128.8876953, -5901.3828125, 4003.0710449
4: -2359.4020996, 1605.0545654, -5583.9013672, 3874.7670898, -6083.2250977, 7187.5249023

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2367213, upper bound: 3611.5635301
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8462422, upper bound: 3611.6319425
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8109617, upper bound: 3611.7901946
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3931620, upper bound: 3611.8082771
time: 0.86 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1788.4462891, 1553.2846680, -4109.6464844, 3659.3586426, -5303.8730469, 5662.9311523
1: -1441.5299072, 1524.0225830, -3312.2368164, 3602.4523926, -4924.7377930, 4836.2583008
2: -2148.8415527, 1648.9160156, -4986.0180664, 3874.4963379, -5894.5551758, 6624.9301758
3: -806.1847534, 2158.9660645, -1829.6464844, 5000.1010742, -5777.1943359, 3956.0874023
4: -2360.2592773, 1605.6569824, -5452.0336914, 3762.6140137, -5989.0522461, 7057.6904297

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2382419, upper bound: 3612.5174673
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2382476, upper bound: 3612.5269240
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2089.6127930, 1831.0540771, -4207.9370117, 3747.5329590, -5683.5024414, 6038.9907227
1: -1682.4387207, 1796.9393311, -3390.6147461, 3687.6118164, -5242.0024414, 5187.5527344
2: -2501.8356934, 1944.2260742, -5099.0517578, 3965.4316406, -6325.3374023, 7023.0722656
3: -946.0217896, 2526.3615723, -1872.7385254, 5118.1093750, -6034.1806641, 4356.9697266
4: -2750.0595703, 1895.3664551, -5578.9545898, 3853.3989258, -6455.9194336, 7463.9394531

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6004907, upper bound: 3613.2218617
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6007276, upper bound: 3613.2818775
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1989.9539795, 1730.5888672, -3893.2695312, 3533.3742676, -5358.2915039, 5623.8579102
1: -1603.2304688, 1699.9106445, -3132.1398926, 3487.8605957, -4953.5048828, 4832.0493164
2: -2384.8439941, 1838.4045410, -4677.8745117, 3760.3027344, -5984.9741211, 6515.0869141
3: -899.8075562, 2403.7790527, -1780.7159424, 4748.9350586, -5624.8110352, 4141.1835938
4: -2621.4545898, 1792.7440186, -5113.2617188, 3645.7424316, -6098.0434570, 6906.0053711

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2761187, upper bound: 3611.0308102
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2055790, upper bound: 3611.2582341
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5231678, upper bound: 3611.2729753
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -1989.9539795, 1730.5888672, -4211.9047852, 3766.3974609, -5594.7104492, 5942.4936523
1: -1603.2304688, 1699.9106445, -3393.9875488, 3706.7153320, -5175.7460938, 5093.8979492
2: -2384.8439941, 1838.4045410, -5107.2866211, 3988.1188965, -6226.3764648, 6930.7578125
3: -899.8075562, 2403.7790527, -1878.7052002, 5128.8876953, -5995.3447266, 4244.9267578
4: -2621.4545898, 1792.7440186, -5583.9013672, 3874.7670898, -6342.0258789, 7373.4472656

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2761190, upper bound: 3611.5635376
time: 0.86 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2055790, upper bound: 3611.7935628
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5231680, upper bound: 3611.8082872
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -3799.2736816, 3435.0346680, -5275.6850586, 5530.3515625
1: -1603.6752930, 1700.3911133, -3057.7436523, 3392.0942383, -4870.5131836, 4758.1347656
2: -2385.5007324, 1838.9254150, -4567.0595703, 3656.3027344, -5895.0834961, 6405.9848633
3: -900.0618896, 2404.4448242, -1735.9165039, 4631.4780273, -5511.5942383, 4098.2426758
4: -2622.1757812, 1793.2502441, -4992.4560547, 3542.5493164, -6011.8906250, 6785.7055664

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2772759, upper bound: 3611.9795567
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2771667, upper bound: 3611.9154779
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1990.5086670, 1731.0780029, -4109.6464844, 3659.3586426, -5503.9355469, 5840.7246094
1: -1603.6752930, 1700.3911133, -3312.2368164, 3602.4523926, -5084.7055664, 5012.6269531
2: -2385.5007324, 1838.9254150, -4986.0180664, 3874.4963379, -6127.9687500, 6813.5117188
3: -900.0618896, 2404.4448242, -1829.6464844, 5000.1010742, -5871.1083984, 4197.8271484
4: -2622.1757812, 1793.2502441, -5452.0336914, 3762.6140137, -6247.7148438, 7244.6191406

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2772762, upper bound: 3612.5174762
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2771670, upper bound: 3612.5269329
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -3859.0175781, 3518.9118652, -1789.0394287, 1553.7523193, -5412.7700195, 5104.4409180
1: -3109.3752441, 3470.1640625, -1442.0032959, 1524.4995117, -4633.8745117, 4741.2314453
2: -4711.1235352, 3730.4829102, -2149.5278320, 1649.4299316, -6317.3198242, 5690.9531250
3: -1735.1317139, 4760.4790039, -806.4337769, 2159.6562500, -3843.0075684, 5499.5830078
4: -5148.9082031, 3620.7448730, -2360.9973145, 1606.1440430, -6720.9970703, 5785.5893555

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.4468984, upper bound: 3613.2367712
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5386904, upper bound: 3611.8462992
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6874214, upper bound: 3613.3932194
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -3760.5947266, 3414.5329590, -1789.6997070, 1554.3094482, -5314.9038086, 5017.8593750
1: -3031.7561035, 3367.2983398, -1442.5327148, 1525.0668945, -4556.8217773, 4653.3427734
2: -4595.4418945, 3617.1870117, -2150.3056641, 1650.0454102, -6205.3764648, 5596.4418945
3: -1684.2226562, 4637.5922852, -806.7396851, 2160.4394531, -3798.7216797, 5381.2734375
4: -5022.1411133, 3511.9458008, -2361.8547363, 1606.7463379, -6597.5488281, 5696.0581055

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3224048, upper bound: 3613.2382427
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2469313, upper bound: 3613.2382451
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -5861.5605469, 5396.7363281, -1789.0394287, 1553.7523193, -7317.3325195, 6630.9003906
1: -4717.6069336, 5330.3613281, -1442.0032959, 1524.4995117, -6147.4638672, 6301.8491211
2: -7172.2138672, 5730.7680664, -2149.5278320, 1649.4299316, -8577.9541016, 7323.5556641
3: -2642.5119629, 7240.2539062, -806.4337769, 2159.6562500, -4605.9921875, 7833.2578125
4: -7823.1733398, 5547.4809570, -2360.9973145, 1606.1440430, -9193.0439453, 7345.8681641

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5635281, upper bound: 3613.2367216
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6319409, upper bound: 3611.8462422
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
time: 0.91 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8082787, upper bound: 3613.3931624
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -5764.6738281, 5293.8017578, -1789.6997070, 1554.3094482, -7221.5854492, 6542.7866211
1: -4641.5361328, 5229.5292969, -1442.5327148, 1525.0668945, -6071.8701172, 6213.6933594
2: -7058.9257812, 5619.7089844, -2150.3056641, 1650.0454102, -8465.7919922, 7228.7324219
3: -2592.7531738, 7120.0683594, -806.7396851, 2160.4394531, -4560.5898438, 7715.5400391
4: -7699.0053711, 5441.0517578, -2361.8547363, 1606.7463379, -9069.3974609, 7255.2407227

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5174656, upper bound: 3613.2382423
time: 0.80 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5269221, upper bound: 3613.2382479
time: 0.79 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.99 seconds
IS_A2_A1_B2_A1_A1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2374698, upper bound: 3612.3003331
IS_A2_A1_B2_A1_A1_B1_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2377862, upper bound: 3612.9741242
IS_A2_A1_B2_A1_A1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2381314, upper bound: 3612.4516718
IS_A2_A1_B2_A1_A1_B1_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2381393, upper bound: 3612.9618775
IS_A2_A1_B2_A1_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.8088106, upper bound: 3611.2548844
IS_A2_A1_B2_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.3910105, upper bound: 3611.2729681
IS_A2_A1_B2_A1_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2358041, upper bound: 3611.9795480
IS_A2_A1_B2_A1_A1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2356948, upper bound: 3611.9154692
IS_A2_A1_B2_A1_A1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.8109617, upper bound: 3611.7901946
IS_A2_A1_B2_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.3931620, upper bound: 3611.8082771
IS_A2_A1_B2_A1_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2382419, upper bound: 3612.5174673
IS_A2_A1_B2_A1_A1_B2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2382476, upper bound: 3612.5269240
IS_A2_A1_B2_A1_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.6004907, upper bound: 3613.2218617
IS_A2_A1_B2_A1_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.6007276, upper bound: 3613.2818775
IS_A2_A1_B2_A1_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2055790, upper bound: 3611.2582341
IS_A2_A1_B2_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.5231678, upper bound: 3611.2729753
IS_A2_A1_B2_A1_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2055790, upper bound: 3611.7935628
IS_A2_A1_B2_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.5231680, upper bound: 3611.8082872
IS_A2_A1_B2_A1_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2772759, upper bound: 3611.9795567
IS_A2_A1_B2_A1_A2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2771667, upper bound: 3611.9154779
IS_A2_A1_B2_A1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2772762, upper bound: 3612.5174762
IS_A2_A1_B2_A1_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3613.2771670, upper bound: 3612.5269329
IS_A2_A2_B1_B1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
IS_A2_A2_B1_B1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3611.6874214, upper bound: 3613.3932194
IS_A2_A2_B1_B1_B1_A1_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.3224048, upper bound: 3613.2382427
IS_A2_A2_B1_B1_B1_A1_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.2469313, upper bound: 3613.2382451
IS_A2_A2_B1_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
IS_A2_A2_B1_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 3.99
Output dim: 0, lower bound: -3611.8082787, upper bound: 3613.3931624
IS_A2_A2_B1_B1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.5174656, upper bound: 3613.2382423
IS_A2_A2_B1_B1_B1_A2_A2_A2_A2, status: Status.VERIFIED, split count: 9, time: 3.99
Output dim: 0, lower bound: -3612.5269221, upper bound: 3613.2382479

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1748.8229980, 1523.1208496, -3893.2695312, 3533.3742676, -5118.8637695, 5416.3896484
1: -1410.8298340, 1490.6773682, -3132.1398926, 3487.8605957, -4763.0117188, 4622.8168945
2: -2102.9204102, 1612.2811279, -4677.8745117, 3760.3027344, -5706.5097656, 6290.1552734
3: -789.2527466, 2114.3483887, -1780.7159424, 4748.9350586, -5514.7250977, 3854.1862793
4: -2308.2128906, 1570.9088135, -5113.2617188, 3645.7424316, -5788.4067383, 6684.1704102

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2346469, upper bound: 3611.0308015
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8662198, upper bound: 3611.1067866
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1558231, upper bound: 3611.2658288
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2804488, upper bound: 3611.2715377
time: 0.79 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1748.8229980, 1523.1208496, -4211.9047852, 3766.3974609, -5356.5556641, 5735.0253906
1: -1410.8298340, 1490.6773682, -3393.9875488, 3706.7153320, -4986.3251953, 4884.6650391
2: -2102.9204102, 1612.2811279, -5107.2866211, 3988.1188965, -5949.8906250, 6710.4316406
3: -789.2527466, 2114.3483887, -1878.7052002, 5128.8876953, -5885.2592773, 3959.0102539
4: -2308.2128906, 1570.9088135, -5583.9013672, 3874.7670898, -6034.4394531, 7154.8100586

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2367213, upper bound: 3611.5635301
time: 0.77 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8683405, upper bound: 3611.7931655
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1579746, upper bound: 3611.8024848
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2826003, upper bound: 3611.8081937
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1937.9049072, 1687.5341797, -3893.2695312, 3533.3742676, -5306.5454102, 5580.8027344
1: -1562.5092773, 1655.6096191, -3132.1398926, 3487.8605957, -4912.9907227, 4787.7485352
2: -2324.3305664, 1790.9176025, -4677.8745117, 3760.3027344, -5925.6254883, 6468.2363281
3: -877.5751953, 2344.1418457, -1780.7159424, 4748.9350586, -5602.9028320, 4080.7456055
4: -2553.3735352, 1747.1284180, -5113.2617188, 3645.7424316, -6031.2792969, 6860.3896484

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2761187, upper bound: 3611.0308102
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4430990, upper bound: 3611.2707464
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4562629, upper bound: 3611.2715451
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1937.9049072, 1687.5341797, -4211.9047852, 3766.3974609, -5543.3090820, 5899.4389648
1: -1562.5092773, 1655.6096191, -3393.9875488, 3706.7153320, -5135.5053711, 5049.5966797
2: -2324.3305664, 1790.9176025, -5107.2866211, 3988.1188965, -6167.5458984, 6883.9072266
3: -877.5751953, 2344.1418457, -1878.7052002, 5128.8876953, -5973.4365234, 4184.8530273
4: -2553.3735352, 1747.1284180, -5583.9013672, 3874.7670898, -6275.8149414, 7328.6123047

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2781930, upper bound: 3611.5635387
time: 0.99 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5070687, upper bound: 3611.8046584
time: 0.78 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4452502, upper bound: 3611.8074013
time: 0.68 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4584142, upper bound: 3611.8082011
time: 0.76 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3858.1254883, 3517.8981934, -1781.4322510, 1545.8439941, -5403.9692383, 5094.6958008
1: -3108.6789551, 3469.1357422, -1435.8663330, 1515.6203613, -4623.0234375, 4733.5258789
2: -4710.0732422, 3729.4184570, -2139.4426270, 1639.3646240, -6298.7573242, 5678.5122070
3: -1734.6885986, 4759.2797852, -801.7914429, 2149.5541992, -3830.7165527, 5490.3901367
4: -5147.7416992, 3619.7167969, -2351.1601562, 1596.4366455, -6699.7431641, 5773.2573242

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.4468984, upper bound: 3613.2367712
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5386904, upper bound: 3611.8462992
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6693366, upper bound: 3612.8110187
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
time: 0.89 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3858.2724609, 3518.0715332, -1750.3654785, 1516.0241699, -5374.2968750, 5065.0688477
1: -3108.7912598, 3469.3134766, -1411.5675049, 1485.7277832, -4594.5185547, 4710.0131836
2: -4710.2431641, 3729.5966797, -2104.8874512, 1607.6319580, -6279.8286133, 5645.2924805
3: -1734.7581787, 4759.4809570, -787.0106201, 2111.9821777, -3795.5510254, 5480.5717773
4: -5147.9326172, 3619.8889160, -2311.7551270, 1564.6967773, -6683.3837891, 5735.0527344

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.4469019, upper bound: 3613.2367712
time: 0.73 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6816279, upper bound: 3613.1580320
time: 0.81 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6873369, upper bound: 3613.2826578
time: 0.80 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -5853.4238281, 5389.2792969, -1781.4322510, 1545.8439941, -7293.7714844, 6615.4921875
1: -4711.0771484, 5322.8022461, -1435.8663330, 1515.6203613, -6125.5917969, 6288.3457031
2: -7162.5502930, 5722.6894531, -2139.4426270, 1639.3646240, -8551.1044922, 7304.9697266
3: -2638.8830566, 7230.4995117, -801.7914429, 2149.5541992, -4590.7573242, 7815.7314453
4: -7812.6225586, 5539.6005859, -2351.1601562, 1596.4366455, -9162.7167969, 7327.5703125

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5635281, upper bound: 3613.2367216
time: 0.83 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.6319409, upper bound: 3611.8462422
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.7901929, upper bound: 3612.8109617
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
time: 0.74 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -5852.9716797, 5389.1948242, -1750.3654785, 1516.0241699, -7274.9179688, 6585.5996094
1: -4710.6997070, 5322.6567383, -1411.5675049, 1485.7277832, -6106.2534180, 6264.5078125
2: -7161.9902344, 5722.3881836, -2104.8874512, 1607.6319580, -8531.5410156, 7271.2797852
3: -2638.7072754, 7230.0019531, -787.0106201, 2111.9821777, -4555.4077148, 7805.2846680
4: -7811.8710938, 5539.1606445, -2311.7551270, 1564.6967773, -9145.5478516, 7288.8032227

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.5635322, upper bound: 3613.2367216
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8024852, upper bound: 3613.1579750
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8081941, upper bound: 3613.2826008
time: 0.84 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 7.55 seconds
IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.1558231, upper bound: 3611.2658288
IS_A2_A1_B2_A1_A1_B2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.2804488, upper bound: 3611.2715377
IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.1579746, upper bound: 3611.8024848
IS_A2_A1_B2_A1_A1_B2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.2826003, upper bound: 3611.8081937
IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.4430990, upper bound: 3611.2707464
IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.4562629, upper bound: 3611.2715451
IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.4452502, upper bound: 3611.8074013
IS_A2_A1_B2_A1_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3613.4584142, upper bound: 3611.8082011
IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.6693366, upper bound: 3612.8110187
IS_A2_A2_B1_B1_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.6874196, upper bound: 3613.3932194
IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.6816279, upper bound: 3613.1580320
IS_A2_A2_B1_B1_B1_A1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.6873369, upper bound: 3613.2826578
IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.7901929, upper bound: 3612.8109617
IS_A2_A2_B1_B1_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.8082766, upper bound: 3613.3931624
IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.8024852, upper bound: 3613.1579750
IS_A2_A2_B1_B1_B1_A2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 10, time: 7.55
Output dim: 0, lower bound: -3611.8081941, upper bound: 3613.2826008

## BFS IS instance: IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1868.9536133, 1630.1148682, -3893.2695312, 3533.3742676, -5237.6562500, 5523.3828125
1: -1507.3166504, 1597.7042236, -3132.1398926, 3487.8605957, -4857.8759766, 4729.8427734
2: -2244.7404785, 1727.6901855, -4677.8745117, 3760.3027344, -5846.2036133, 6405.5644531
3: -845.9409790, 2263.9008789, -1780.7159424, 4748.9350586, -5571.6333008, 4000.5173340
4: -2465.5895996, 1685.2757568, -5113.2617188, 3645.7424316, -5943.6391602, 6798.5366211

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1671871, upper bound: 3611.0280014
time: 0.69 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4335983, upper bound: 3611.2677783
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_B2_A1_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.7378704
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7432036, upper bound: 3614.7432043
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -3613.0484451, upper bound: 3614.7378704
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -3614.7432036, upper bound: 3614.7432043

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2483.5041504, 2136.1025391, -4308.8496094, 4375.4306641
1: -1744.2438965, 1855.7540283, -2000.1508789, 2093.1101074, -3837.3540039, 3855.9047852
2: -2557.1499023, 2009.1342773, -2953.3486328, 2269.6425781, -4826.7924805, 4962.4829102
3: -990.0590820, 2595.9711914, -1118.7990723, 2961.1840820, -3951.2431641, 3714.7702637
4: -2815.3046875, 1956.6667480, -3249.4038086, 2211.4111328, -5026.7148438, 5206.0703125

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.0463023
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0463023, upper bound: 3614.7378704
time: 0.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -2500.9851074, 2149.7963867, -2502.3757324, 2150.9697266, -4651.9545898, 4652.1718750
1: -2014.5847168, 2106.3789062, -2015.7012939, 2107.5605469, -4122.1455078, 4122.0800781
2: -2976.4731445, 2283.9982910, -2978.1022949, 2285.2739258, -5261.7460938, 5262.0996094
3: -1125.7036133, 2982.2333984, -1126.3112793, 2983.9003906, -4109.6040039, 4108.5449219
4: -3274.6416016, 2225.7395020, -3276.4265137, 2226.9489746, -5501.5903320, 5502.1655273

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2467415, upper bound: 3613.7401196
time: 0.76 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7401187, upper bound: 3613.7401196
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.35 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 0, lower bound: -3613.0463023, upper bound: 3613.0463023
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3613.0463023, upper bound: 3614.7378704
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3614.2467415, upper bound: 3613.7401196
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -3613.7401187, upper bound: 3613.7401196

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2492.6989746, 2143.4062500, -4316.1538086, 4384.6259766
1: -1744.2438965, 1855.7540283, -2007.7033691, 2100.2153320, -3844.4592285, 3863.4572754
2: -2557.1499023, 2009.1342773, -2965.3178711, 2277.3183594, -4834.4682617, 4974.4521484
3: -990.0590820, 2595.9711914, -1122.5045166, 2972.1970215, -3962.2561035, 3718.4755859
4: -2815.3046875, 1956.6667480, -3262.5205078, 2219.0126953, -5034.3164062, 5219.1865234

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0253722, upper bound: 3613.9143632
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0462852, upper bound: 3613.0462852
time: 0.81 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -2253.7849121, 1949.6795654, -2502.3757324, 2150.9697266, -4404.7543945, 4452.0551758
1: -1815.8967285, 1912.4958496, -2015.7012939, 2107.5605469, -3923.4572754, 3928.1962891
2: -2689.9379883, 2072.0576172, -2978.1022949, 2285.2739258, -4975.2119141, 5050.1586914
3: -1018.2312012, 2703.3862305, -1126.3112793, 2983.9003906, -4002.1315918, 3829.6975098
4: -2958.2136230, 2018.5753174, -3276.4265137, 2226.9489746, -5185.1625977, 5295.0019531

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5266955, upper bound: 3613.3947133
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3946870
time: 0.74 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -4052.8181152, 3629.6472168, -2480.2546387, 2133.4521484, -6186.2705078, 5957.1157227
1: -3268.6315918, 3572.1374512, -1998.1865234, 2090.4851074, -5359.1166992, 5443.0429688
2: -4916.9809570, 3842.0942383, -2952.6774902, 2266.7866211, -7171.4755859, 6651.9565430
3: -1810.5534668, 4941.9809570, -1117.0773926, 2959.1574707, -4730.1660156, 6027.1811523
4: -5374.5024414, 3732.8491211, -3248.1149902, 2208.8212891, -7583.3232422, 6832.8784180

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947111, upper bound: 3612.9210240
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3947017, upper bound: 3613.3947001
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.22 seconds
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3613.0253722, upper bound: 3613.9143632
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 3.22
Output dim: 0, lower bound: -3613.0462852, upper bound: 3613.0462852
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3613.5266955, upper bound: 3613.3947133
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3612.9866480, upper bound: 3613.3946870
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3613.3947111, upper bound: 3612.9210240
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.22
Output dim: 0, lower bound: -3613.3947017, upper bound: 3613.3947001

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2167.9160156, 1887.7866211, -2581.3503418, 2232.3024902, -4400.2187500, 4469.1367188
1: -1740.3665771, 1851.6635742, -2078.9414062, 2186.4660645, -3926.8322754, 3930.6047363
2: -2551.4448242, 2004.6934814, -3070.6633301, 2370.3322754, -4921.7768555, 5075.3564453
3: -987.8737183, 2590.2285156, -1164.3217773, 3082.0656738, -4069.9387207, 3754.5502930
4: -2809.0334473, 1952.3853760, -3378.4309082, 2312.7006836, -5121.7333984, 5330.8149414

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9935758, upper bound: 3613.4718599
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0253722, upper bound: 3612.7482594
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -2032.5021973, 1767.8198242, -2502.3757324, 2150.9697266, -4183.4716797, 4270.1953125
1: -1637.2110596, 1736.3991699, -2015.7012939, 2107.5605469, -3744.7714844, 3752.1000977
2: -2434.3435059, 1878.1309814, -2978.1022949, 2285.2739258, -4719.6171875, 4856.2324219
3: -919.7875977, 2453.9553223, -1126.3112793, 2983.9003906, -3903.6875000, 3580.2666016
4: -2675.7617188, 1831.0811768, -3276.4265137, 2226.9489746, -4902.7109375, 5107.5078125

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9760414, upper bound: 3612.9210071
time: 0.92 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9760414, upper bound: 3613.3946870
time: 0.73 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -3836.9106445, 3486.1018066, -2488.1984863, 2140.8833008, -5977.7939453, 5781.3383789
1: -3092.3166504, 3437.8122559, -2004.5759277, 2097.7246094, -5190.0410156, 5280.0585938
2: -4683.6884766, 3693.0117188, -2962.2211914, 2274.6040039, -6910.6552734, 6472.6806641
3: -1720.4913330, 4730.5346680, -1120.9125977, 2969.2133789, -4636.8237305, 5784.5556641
4: -5119.6196289, 3585.9409180, -3258.6638184, 2216.5571289, -7297.9301758, 6655.0590820

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9866312, upper bound: 3612.9866317
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9866312, upper bound: 3613.3946870
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -4052.3271484, 3629.2258301, -2250.3500977, 1947.1267090, -5999.4536133, 5729.0063477
1: -3268.2348633, 3571.7207031, -1813.4694824, 1909.9956055, -5178.2299805, 5259.7895508
2: -4916.3447266, 3841.6464844, -2686.4743652, 2069.2993164, -6979.1625977, 6389.5590820
3: -1810.3562012, 4941.3852539, -1017.0298462, 2699.0520020, -4472.5239258, 5928.8173828
4: -5373.8144531, 3732.4162598, -2953.7026367, 2016.5915527, -7390.4057617, 6542.3178711

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9210238, upper bound: 3612.9210240
time: 0.91 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9210238, upper bound: 3612.9210240
time: 0.82 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -4050.8161621, 3627.9782715, -4120.4204102, 3711.1000977, -7527.2377930, 7524.8212891
1: -3267.1457520, 3570.2890625, -3321.5488281, 3653.8020020, -6721.3237305, 6699.5356445
2: -4914.8544922, 3840.0681152, -5010.3110352, 3927.9245605, -8566.1201172, 8572.4111328
3: -1809.3121338, 4939.4960938, -1844.2166748, 5051.1689453, -6709.2172852, 6653.5566406
4: -5372.2358398, 3731.0759277, -5481.7441406, 3815.8579102, -8909.7607422, 8934.9072266

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3925678, upper bound: 3613.0484357
time: 0.87 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3925684, upper bound: 3613.3947001
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.45 seconds
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9935758, upper bound: 3613.4718599
IS_A1_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 0, lower bound: -3613.0253722, upper bound: 3612.7482594
IS_A2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9760414, upper bound: 3612.9210071
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9760414, upper bound: 3613.3946870
IS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9866312, upper bound: 3612.9866317
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9866312, upper bound: 3613.3946870
IS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9210238, upper bound: 3612.9210240
IS_A2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.45
Output dim: 0, lower bound: -3612.9210238, upper bound: 3612.9210240
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -3613.3925678, upper bound: 3613.0484357
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.45
Output dim: 0, lower bound: -3613.3925684, upper bound: 3613.3947001

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2167.9160156, 1887.7866211, -2358.2717285, 2050.9274902, -4218.8432617, 4246.0585938
1: -1740.3665771, 1851.6635742, -1899.6748047, 2011.0880127, -3751.4545898, 3751.3383789
2: -2551.4448242, 2004.6934814, -2813.1901855, 2178.9150391, -4730.3598633, 4817.8833008
3: -987.8737183, 2590.2285156, -1066.3853760, 2830.0002441, -3817.8735352, 3656.6137695
4: -2809.0334473, 1952.3853760, -3093.3518066, 2125.5549316, -4934.5869141, 5045.7373047

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9935758, upper bound: 3612.7499357
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5269993, upper bound: 3613.2314473
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9142865, upper bound: 3613.4716853
time: 0.71 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2032.5021973, 1767.8198242, -4118.9829102, 3708.0341797, -5575.2148438, 5886.8027344
1: -1637.2110596, 1736.3991699, -3320.4133301, 3650.5236816, -5150.3232422, 5056.8125000
2: -2434.3435059, 1878.1309814, -5008.2875977, 3924.4572754, -6210.9174805, 6869.0732422
3: -919.7875977, 2453.9553223, -1842.9790039, 5048.2690430, -5918.2714844, 4261.6665039
4: -2675.7617188, 1831.0811768, -5479.5566406, 3812.5839844, -6332.0761719, 7304.8974609

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2938388, upper bound: 3613.2903833
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5248534, upper bound: 3613.2904687
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -3842.9118652, 3492.4929199, -4181.3920898, 3726.3173828, -7337.3691406, 7412.4399414
1: -3097.1203613, 3444.2551270, -3369.6718750, 3667.5957031, -6566.0410156, 6591.8959961
2: -4690.9912109, 3699.9401855, -5069.0449219, 3944.6782227, -8343.4814453, 8463.0097656
3: -1723.5593262, 4738.4677734, -1861.9174805, 5087.0522461, -6668.2500000, 6438.6455078
4: -5127.6469727, 3592.6191406, -5544.2031250, 3832.1306152, -8665.6191406, 8829.3251953

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9577055, upper bound: 3613.2231209
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9579360, upper bound: 3613.2904297
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -4042.3808594, 3621.1086426, -3755.4191895, 3436.7604980, -7223.9414062, 7146.0874023
1: -3260.3381348, 3563.4863281, -3022.4970703, 3393.2045898, -6435.0883789, 6391.2719727
2: -4904.1665039, 3832.7185059, -4533.4946289, 3652.9567871, -8240.9951172, 8090.6215820
3: -1805.8609619, 4929.3979492, -1720.2464600, 4619.7963867, -6281.0791016, 6499.6718750
4: -5360.6274414, 3723.9399414, -4960.5527344, 3546.3264160, -8587.6855469, 8411.0019531

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3473213, upper bound: 3613.0253722
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3925507, upper bound: 3613.0484284
time: 0.81 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -4050.8161621, 3627.9782715, -4118.7343750, 3709.6984863, -7525.7065430, 7523.0742188
1: -3267.1457520, 3570.2890625, -3320.1940918, 3652.4467773, -6719.8413086, 6698.1323242
2: -4914.8544922, 3840.0681152, -5008.3706055, 3926.4611816, -8564.5039062, 8570.3779297
3: -1809.3121338, 4939.4960938, -1843.4637451, 5049.2045898, -6707.1826172, 6652.7294922
4: -5372.2358398, 3731.0759277, -5479.5917969, 3814.4267578, -8908.1875000, 8932.6533203

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.3473222, upper bound: 3613.3807152
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3925514, upper bound: 3613.3946929
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.51 seconds
IS_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.5269993, upper bound: 3613.2314473
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.9142865, upper bound: 3613.4716853
IS_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 0, lower bound: -3613.2938388, upper bound: 3613.2903833
IS_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -3613.5248534, upper bound: 3613.2904687
IS_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.9577055, upper bound: 3613.2231209
IS_A2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.9579360, upper bound: 3613.2904297
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.3473213, upper bound: 3613.0253722
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -3613.3925507, upper bound: 3613.0484284
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -3612.3473222, upper bound: 3613.3807152
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 0, lower bound: -3613.3925514, upper bound: 3613.3946929

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -2152.8498535, 1874.3758545, -2358.2658691, 2050.9230957, -4203.7724609, 4232.6416016
1: -1728.2363281, 1838.3602295, -1899.6701660, 2011.0839844, -3739.3203125, 3738.0302734
2: -2533.5168457, 1990.3399658, -2813.1828613, 2178.9106445, -4712.4267578, 4803.5229492
3: -980.9119873, 2572.0026855, -1066.3830566, 2829.9931641, -3810.9050293, 3638.3857422
4: -2789.3542480, 1938.5400391, -3093.3442383, 2125.5500488, -4914.9042969, 5031.8842773

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9142865, upper bound: 3612.7497682
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8278823, upper bound: 3613.3169504
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5715331, upper bound: 3613.3168675
time: 0.95 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2016.1315918, 1753.1184082, -4118.7739258, 3707.8098145, -5558.7055664, 5871.8911133
1: -1624.1541748, 1721.8516846, -3320.2460938, 3650.2983398, -5137.0996094, 5042.0976562
2: -2415.1628418, 1862.4095459, -5008.0327148, 3924.2148438, -6191.6035156, 6853.2387695
3: -912.2088013, 2434.4267578, -1842.8704834, 5047.9916992, -5910.5761719, 4242.0781250
4: -2654.6584473, 1815.8958740, -5479.2773438, 3812.3491211, -6310.8525391, 7289.5087891

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5225928, upper bound: 3612.9457886
time: 0.74 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5225933, upper bound: 3613.2904654
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4036.8623047, 3616.4575195, -3755.4191895, 3436.7604980, -7218.0380859, 7139.7270508
1: -3255.8879395, 3559.0324707, -3022.4970703, 3393.2045898, -6430.3281250, 6385.3574219
2: -4897.8276367, 3827.8811035, -4533.4946289, 3652.9567871, -8234.0380859, 8083.9575195
3: -1803.4227295, 4922.9467773, -1720.2464600, 4619.7963867, -6277.8867188, 6492.6704102
4: -5353.5512695, 3719.1760254, -4960.5527344, 3546.3264160, -8579.9980469, 8404.3916016

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2208807, upper bound: 3612.9456765
time: 0.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2881942, upper bound: 3612.9457675
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4149.9946289, 3731.7597656, -4114.1821289, 3705.8334961, -7617.8525391, 7604.0507812
1: -3345.0632324, 3672.0166016, -3316.5407715, 3648.6547852, -6791.5932617, 6781.0585938
2: -5031.8164062, 3950.8940430, -5003.0346680, 3922.3750000, -8672.2626953, 8658.2177734
3: -1857.3454590, 5062.9272461, -1841.4328613, 5043.8066406, -6746.1782227, 6768.9555664
4: -5499.6914062, 3839.9152832, -5473.7167969, 3810.4533691, -9027.0947266, 9015.8544922

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.1955508, upper bound: 3612.4410265
time: 0.77 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.1955510, upper bound: 3612.4410275
time: 0.83 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4045.3002930, 3623.3295898, -4118.7343750, 3709.6984863, -7519.8486328, 7516.8251953
1: -3262.7060547, 3565.8386230, -3320.1940918, 3652.4467773, -6715.1279297, 6692.3110352
2: -4908.5209961, 3835.2497559, -5008.3706055, 3926.4611816, -8557.6269531, 8563.8535156
3: -1806.8747559, 4933.0483398, -1843.4637451, 5049.2045898, -6704.0444336, 6645.7749023
4: -5365.1650391, 3726.3144531, -5479.5917969, 3814.4267578, -8900.5771484, 8926.1738281

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1019188, upper bound: 3612.9866408
time: 0.82 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1019198, upper bound: 3613.3937277
time: 0.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.48 seconds
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -3612.8278823, upper bound: 3613.3169504
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -3612.5715331, upper bound: 3613.3168675
IS_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.5225928, upper bound: 3612.9457886
IS_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.5225933, upper bound: 3613.2904654
IS_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.2208807, upper bound: 3612.9456765
IS_A2_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.2881942, upper bound: 3612.9457675
IS_A2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -3612.1955508, upper bound: 3612.4410265
IS_A2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -3612.1955510, upper bound: 3612.4410275
IS_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.1019188, upper bound: 3612.9866408
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 0, lower bound: -3613.1019198, upper bound: 3613.3937277

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -2184.7751465, 1897.3660889, -2355.0437012, 2048.1782227, -4232.9526367, 4252.4096680
1: -1752.8153076, 1859.8566895, -1897.1069336, 2008.3842773, -3761.1997070, 3756.9636230
2: -2567.8337402, 2013.2548828, -2809.4445801, 2175.9846191, -4743.8183594, 4822.6992188
3: -992.5417480, 2605.2426758, -1064.8831787, 2826.1616211, -3818.7033691, 3670.1259766
4: -2828.9016113, 1961.2380371, -3089.2280273, 2122.7258301, -4951.6264648, 5050.4658203

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8278823, upper bound: 3612.6004625
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7317185, upper bound: 3613.0658181
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8278823, upper bound: 3613.3169498
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -2094.1301270, 1816.3482666, -2358.2268066, 2050.8940430, -4145.0234375, 4174.5751953
1: -1681.2548828, 1781.1154785, -1899.6380615, 2011.0559082, -3692.3100586, 3680.7534180
2: -2464.6276855, 1928.8029785, -2813.1347656, 2178.8801270, -4643.5073242, 4741.9375000
3: -953.1464233, 2498.3322754, -1066.3676758, 2829.9482422, -3783.0939941, 3564.6999512
4: -2713.5559082, 1877.9372559, -3093.2917480, 2125.5195312, -4839.0747070, 4971.2290039

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5715331, upper bound: 3612.6003809
time: 1.01 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5715162, upper bound: 3613.3166998
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.9612828, upper bound: 3613.0289806
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5715331, upper bound: 3613.0299010
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -2016.1315918, 1753.1184082, -3753.5053711, 3433.4379883, -5270.1020508, 5506.6235352
1: -1624.1541748, 1721.8516846, -3021.0063477, 3389.5837402, -4863.0546875, 4742.8569336
2: -2415.1628418, 1862.4095459, -4530.9873047, 3649.0476074, -5886.2827148, 6386.4462891
3: -912.2088013, 2434.4267578, -1718.8004150, 4616.4389648, -5486.3051758, 4104.4902344
4: -2654.6584473, 1815.8958740, -4957.7919922, 3542.8515625, -6009.5380859, 6773.6879883

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3921854, upper bound: 3612.9457471
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5225928, upper bound: 3612.9457539
time: 0.83 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2016.1315918, 1753.1184082, -4109.4062500, 3698.8452148, -5550.6508789, 5862.5244141
1: -1624.1541748, 1721.8516846, -3312.7565918, 3641.3803711, -5128.9506836, 5034.6079102
2: -2415.1628418, 1862.4095459, -4996.5737305, 3914.7431641, -6183.0322266, 6842.1821289
3: -912.2088013, 2434.4267578, -1838.6700439, 5036.2285156, -5899.2133789, 4238.3720703
4: -2654.6584473, 1815.8958740, -5466.6860352, 3803.1638184, -6302.5996094, 7277.3579102

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3921870, upper bound: 3613.2904328
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5225933, upper bound: 3613.2904307
time: 0.85 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4045.3002930, 3623.3295898, -6216.7109375, 5674.2143555, -9099.0136719, 9469.4541016
1: -3262.7060547, 3565.8386230, -5003.6396484, 5605.8374023, -8339.7226562, 8254.8554688
2: -4908.5209961, 3835.2497559, -7601.1987305, 6024.5136719, -10253.7744141, 10916.1835938
3: -1806.8747559, 4933.0483398, -2785.5385742, 7659.9350586, -9143.9746094, 7428.4394531
4: -5365.1650391, 3726.3144531, -8297.8203125, 5835.6684570, -10520.2138672, 11497.0751953

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209953, upper bound: 3613.1122418
time: 0.90 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209953, upper bound: 3613.3936238
time: 0.89 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.63 seconds
IS_A1_B2_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -3612.7317185, upper bound: 3613.0658181
IS_A1_B2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3612.8278823, upper bound: 3613.3169498
IS_A1_B2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -3610.9612828, upper bound: 3613.0289806
IS_A1_B2_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -3612.5715331, upper bound: 3613.0299010
IS_A2_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3613.3921854, upper bound: 3612.9457471
IS_A2_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3613.5225928, upper bound: 3612.9457539
IS_A2_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3613.3921870, upper bound: 3613.2904328
IS_A2_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3613.5225933, upper bound: 3613.2904307
IS_A2_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.63
Output dim: 0, lower bound: -3612.9209953, upper bound: 3613.1122418
IS_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 0, lower bound: -3612.9209953, upper bound: 3613.3936238

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -2158.2770996, 1875.3184814, -2354.9560547, 2048.1142578, -4206.3911133, 4230.2744141
1: -1731.6958008, 1838.1845703, -1897.0347900, 2008.3218994, -3740.0175781, 3735.2189941
2: -2536.8283691, 1989.5699463, -2809.3322754, 2175.9172363, -4712.7456055, 4798.9018555
3: -980.3413696, 2574.0781250, -1064.8494873, 2826.0581055, -3806.3994141, 3638.9277344
4: -2794.7087402, 1938.5864258, -3089.1066895, 2122.6579590, -4917.3666992, 5027.6928711

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8278823, upper bound: 3612.6004625
time: 0.72 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8278786, upper bound: 3613.3168281
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8278420, upper bound: 3613.3169502
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8274466, upper bound: 3613.0300748
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1773.9840088, 1540.9189453, -3740.1625977, 3422.1906738, -5015.0483398, 5281.0805664
1: -1429.9652100, 1511.2940674, -3010.2429199, 3378.7294922, -4656.7709961, 4521.5371094
2: -2131.8408203, 1635.1129150, -4515.4404297, 3637.3796387, -5589.3056641, 6142.4619141
3: -799.3773193, 2141.7551270, -1712.7176514, 4600.6972656, -5357.9946289, 3805.2402344
4: -2341.5251465, 1592.2371826, -4940.6308594, 3531.3344727, -5682.7504883, 6532.8681641

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2359073, upper bound: 3612.2691269
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2359079, upper bound: 3612.5959547
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1974.8409424, 1716.9732666, -3753.3920898, 3433.3173828, -5226.9038086, 5470.3652344
1: -1591.1787109, 1686.4777832, -3020.9165039, 3389.4611816, -4828.2285156, 4707.3925781
2: -2367.1381836, 1823.9008789, -4530.8505859, 3648.9160156, -5835.5737305, 6346.0019531
3: -892.7864990, 2385.7099609, -1718.7421875, 4616.2890625, -5467.3066406, 4052.6704102
4: -2601.9707031, 1778.6447754, -4957.6430664, 3542.7260742, -5954.0341797, 6736.2875977

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7497682, upper bound: 3612.9142867
time: 0.77 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5225693, upper bound: 3612.9457378
time: 1.51 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1773.9840088, 1540.9189453, -4094.6027832, 3686.0998535, -5295.7573242, 5635.5209961
1: -1429.9652100, 1511.2940674, -3300.8657227, 3628.9780273, -4922.5795898, 4812.1596680
2: -2131.8408203, 1635.1129150, -4979.3652344, 3901.4514160, -5886.9365234, 6597.3002930
3: -799.3773193, 2141.7551270, -1831.9272461, 5018.6054688, -5769.0605469, 3939.6254883
4: -2341.5251465, 1592.2371826, -5447.6240234, 3790.1713867, -5976.9946289, 7033.5258789

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3927340, upper bound: 3612.2980321
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3944000, upper bound: 3613.2904148
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1974.8409424, 1716.9732666, -4108.2153320, 3697.5288086, -5506.6718750, 5825.1884766
1: -1591.1787109, 1686.4777832, -3311.8232422, 3640.0505371, -5093.2700195, 4998.3002930
2: -2367.1381836, 1823.9008789, -4995.1645508, 3913.3632812, -6131.6059570, 6800.4956055
3: -892.7864990, 2385.7099609, -1838.0966797, 5034.6440430, -5878.8452148, 4186.3266602
4: -2601.9707031, 1778.6447754, -5465.1235352, 3801.8361816, -6246.4936523, 7236.1616211

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5236063, upper bound: 3612.2980438
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5248143, upper bound: 3613.2904170
time: 0.83 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6019.3583984, 5526.6630859, -6223.2539062, 5681.0043945, -10930.5244141, 10962.4345703
1: -4847.5068359, 5459.3930664, -5008.9394531, 5612.5854492, -9804.0361328, 9797.7597656
2: -7366.2753906, 5868.0766602, -7609.2509766, 6031.7763672, -12465.6552734, 12524.8789062
3: -2706.8198242, 7435.5366211, -2788.7897949, 7668.5429688, -9885.0312500, 9737.8417969
4: -8035.7031250, 5683.2744141, -8306.6455078, 5842.6674805, -12938.9912109, 13029.1230469

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8767170, upper bound: 3613.2104436
time: 0.88 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8767173, upper bound: 3613.1915390
time: 0.87 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.61 seconds
IS_A1_B2_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3612.8278420, upper bound: 3613.3169502
IS_A1_B2_B1_B1_A2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3612.8274466, upper bound: 3613.0300748
IS_A2_A1_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.2359073, upper bound: 3612.2691269
IS_A2_A1_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.2359079, upper bound: 3612.5959547
IS_A2_A1_A1_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3612.7497682, upper bound: 3612.9142867
IS_A2_A1_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.5225693, upper bound: 3612.9457378
IS_A2_A1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.3927340, upper bound: 3612.2980321
IS_A2_A1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.3944000, upper bound: 3613.2904148
IS_A2_A1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.5236063, upper bound: 3612.2980438
IS_A2_A1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.61
Output dim: 0, lower bound: -3613.5248143, upper bound: 3613.2904170
IS_A2_A2_B2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3612.8767170, upper bound: 3613.2104436
IS_A2_A2_B2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.61
Output dim: 0, lower bound: -3612.8767173, upper bound: 3613.1915390

## BFS IS instance: IS_A1_B2_B1_B1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -2158.2770996, 1875.3184814, -2277.4016113, 1982.0769043, -4140.3540039, 4152.7202148
1: -1731.6958008, 1838.1845703, -1834.7592773, 1944.2355957, -3675.9313965, 3672.9431152
2: -2536.8283691, 1989.5699463, -2719.4689941, 2106.4101562, -4643.2382812, 4709.0385742
3: -980.3413696, 2574.0781250, -1030.1179199, 2735.5720215, -3715.9133301, 3604.1960449
4: -2794.7087402, 1938.5864258, -2989.7753906, 2054.7587891, -4849.4677734, 4928.3618164

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8278420, upper bound: 3612.5905723
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8278419, upper bound: 3613.3168285
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2836599, upper bound: 3613.1222413
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.6624986, upper bound: 3613.3166383
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7885319, upper bound: 3613.1054540
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8277071, upper bound: 3612.3042116
time: 0.71 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1967.3288574, 1710.3632812, -3751.7744141, 3431.5451660, -5217.8227539, 5462.1367188
1: -1585.1627197, 1680.1243896, -3019.6406250, 3387.6687012, -4820.5722656, 4699.7646484
2: -2358.4465332, 1816.9644775, -4528.9379883, 3647.0231934, -5825.0439453, 6336.3549805
3: -889.3146973, 2376.9501953, -1717.9342041, 4614.1440430, -5461.4741211, 4042.9384766
4: -2592.3256836, 1771.8234863, -4955.5214844, 3540.9140625, -5942.6557617, 6727.3442383

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2705035, upper bound: 3612.2691141
time: 0.78 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2705041, upper bound: 3612.5959385
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1769.1997070, 1536.8846436, -4181.8173828, 3780.3823242, -5367.5874023, 5718.7021484
1: -1426.1329346, 1507.1867676, -3369.6630859, 3721.7609863, -4996.8203125, 4876.8496094
2: -2126.1992188, 1630.6564941, -5082.0380859, 4003.9072266, -5965.4326172, 6691.8769531
3: -797.1609497, 2136.0798340, -1876.6162109, 5128.3085938, -5871.8906250, 3973.9606934
4: -2335.3125000, 1587.8741455, -5559.8203125, 3888.5200195, -6050.3447266, 7137.8437500

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2363721, upper bound: 3611.9639534
time: 0.69 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3926879, upper bound: 3611.7728990
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3926879, upper bound: 3611.6546248
time: 0.88 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1773.9840088, 1540.9189453, -4088.8339844, 3681.2702637, -5289.3085938, 5629.7519531
1: -1429.9652100, 1511.2940674, -3296.2221680, 3624.3420410, -4916.5498047, 4807.5161133
2: -2131.8408203, 1635.1129150, -4972.7314453, 3896.4514160, -5880.2124023, 6590.0742188
3: -799.3773193, 2141.7551270, -1829.3315430, 5011.9047852, -5761.8007812, 3936.2998047
4: -2341.5251465, 1592.2371826, -5440.2436523, 3785.2375488, -5970.3300781, 7025.5273438

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3944003, upper bound: 3613.2831114
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3943878, upper bound: 3613.1922797
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1971.0959473, 1713.6619873, -4196.2299805, 3792.2695312, -5580.0654297, 5909.8920898
1: -1588.1738281, 1683.2177734, -3381.2126465, 3733.2424316, -5168.8173828, 5064.4296875
2: -2362.7097168, 1820.3715820, -5098.7021484, 4016.2585449, -6211.8364258, 6896.7558594
3: -891.0711060, 2381.1997070, -1882.9768066, 5145.1215820, -5982.9892578, 4222.1240234
4: -2597.1013184, 1775.2158203, -5578.2963867, 3900.6691895, -6321.7641602, 7342.4179688

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2713631, upper bound: 3611.9639634
time: 0.92 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5234992, upper bound: 3611.7729116
time: 0.77 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5234992, upper bound: 3611.7728976
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1974.8409424, 1716.9732666, -4102.2382812, 3692.5268555, -5500.0419922, 5819.2114258
1: -1591.1787109, 1686.4777832, -3307.0063477, 3635.2514648, -5087.0712891, 4993.4843750
2: -2367.1381836, 1823.9008789, -4988.2895508, 3908.1894531, -6124.6987305, 6793.0170898
3: -892.7864990, 2385.7099609, -1835.4100342, 5027.6997070, -5871.3334961, 4182.9047852
4: -2601.9707031, 1778.6447754, -5457.4741211, 3796.7270508, -6239.6455078, 7227.8823242

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5248143, upper bound: 3613.2831201
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5248076, upper bound: 3613.1922929
time: 0.83 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.47 seconds
IS_A1_B2_B1_B1_A2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 3.47
Output dim: 0, lower bound: -3612.7885319, upper bound: 3613.1054540
IS_A1_B2_B1_B1_A2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 3.47
Output dim: 0, lower bound: -3612.8277071, upper bound: 3612.3042116
IS_A2_A1_A1_B2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.2705035, upper bound: 3612.2691141
IS_A2_A1_A1_B2_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.2705041, upper bound: 3612.5959385
IS_A2_A1_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.3926879, upper bound: 3611.7728990
IS_A2_A1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.3926879, upper bound: 3611.6546248
IS_A2_A1_A1_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.3944003, upper bound: 3613.2831114
IS_A2_A1_A1_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.3943878, upper bound: 3613.1922797
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.5234992, upper bound: 3611.7729116
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.5234992, upper bound: 3611.7728976
IS_A2_A1_A1_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.5248143, upper bound: 3613.2831201
IS_A2_A1_A1_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 3.47
Output dim: 0, lower bound: -3613.5248076, upper bound: 3613.1922929

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1763.3666992, 1530.2395020, -4175.6010742, 3774.3015137, -5353.3637695, 5705.8408203
1: -1421.4409180, 1499.6523438, -3364.7504883, 3715.7150879, -4984.4804688, 4864.4023438
2: -2118.2578125, 1622.1214600, -5074.8125000, 3997.4833984, -5948.4746094, 6668.3588867
3: -793.3493042, 2127.9675293, -1873.6568604, 5120.5668945, -5856.6738281, 3960.5600586
4: -2327.8366699, 1579.6760254, -5551.8559570, 3882.3312988, -6033.8315430, 7111.3798828

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2363269, upper bound: 3611.5208195
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8898792, upper bound: 3611.6939932
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9454524, upper bound: 3611.7657368
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1730.8385010, 1499.4067383, -4180.2895508, 3778.6684570, -5327.7456055, 5679.6962891
1: -1395.8912354, 1468.6442871, -3368.4589844, 3720.0307617, -4965.0449219, 4837.1035156
2: -2081.8608398, 1589.1110840, -5080.2231445, 4002.1010742, -5919.2993164, 6653.8188477
3: -777.9437866, 2088.7438965, -1875.8529053, 5126.2592773, -5851.9970703, 3926.5380859
4: -2286.3974609, 1546.6060791, -5557.8129883, 3886.7836914, -5999.4013672, 7099.4790039

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2363269, upper bound: 3611.5203091
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0999183, upper bound: 3611.6542573
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2821227, upper bound: 3611.6542718
time: 0.82 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1769.1436768, 1536.4554443, -4109.0830078, 3695.6638184, -5295.6420898, 5645.5385742
1: -1426.1171875, 1507.0335693, -3312.2670898, 3637.1420898, -4923.0283203, 4819.3007812
2: -2126.1977539, 1630.5471191, -4994.6430664, 3909.3940430, -5884.4467773, 6607.4868164
3: -797.1737671, 2135.8898926, -1836.9516602, 5033.8520508, -5781.6279297, 3935.8190918
4: -2335.3293457, 1587.8358154, -5465.9467773, 3799.0473633, -5974.0517578, 7046.6918945

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381109, upper bound: 3612.4516406
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381153, upper bound: 3612.9612026
time: 0.81 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1773.9840088, 1540.9189453, -4016.2272949, 3612.9208984, -5223.9389648, 5557.1459961
1: -1429.9652100, 1511.2940674, -3238.5327148, 3556.8889160, -4851.7358398, 4749.8266602
2: -2131.8408203, 1635.1129150, -4888.2993164, 3823.7756348, -5810.6884766, 6502.7866211
3: -799.3773193, 2141.7551270, -1794.8391113, 4923.3056641, -5672.6875000, 3902.5117188
4: -2341.5251465, 1592.2371826, -5347.0791016, 3714.3898926, -5903.0581055, 6929.0488281

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381100, upper bound: 3612.5173308
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2381124, upper bound: 3612.5141496
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2008.4277344, 1744.6876221, -4190.1909180, 3786.3881836, -5609.1606445, 5934.8784180
1: -1617.9400635, 1712.6351318, -3376.4394531, 3727.3979492, -5191.0952148, 5089.0747070
2: -2404.7031250, 1852.3420410, -5091.6840820, 4010.0424805, -6245.1557617, 6912.0449219
3: -907.0118408, 2424.3813477, -1880.1043701, 5137.6176758, -5987.4423828, 4260.1464844
4: -2644.7629395, 1805.8095703, -5570.5615234, 3894.6801758, -6360.6464844, 7354.8725586

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2712580, upper bound: 3611.5208212
time: 0.72 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4331667, upper bound: 3611.7721278
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4527742, upper bound: 3611.7727771
time: 0.99 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1907.1134033, 1650.5535889, -4194.0517578, 3789.8122559, -5514.0371094, 5844.6054688
1: -1537.4365234, 1620.7979736, -3379.5002441, 3730.7595215, -5115.8588867, 5000.2983398
2: -2288.3122559, 1753.0501709, -5096.1269531, 4013.6750488, -6134.9643555, 6831.3681641
3: -860.2935791, 2302.1557617, -1881.8945312, 5142.1953125, -5950.5332031, 4142.6367188
4: -2515.1235352, 1708.8535156, -5575.4423828, 3898.1872559, -6237.1582031, 7278.2353516

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2712580, upper bound: 3611.5208179
time: 0.69 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4331667, upper bound: 3611.7720121
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4527740, upper bound: 3611.7727682
time: 1.00 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1972.1157227, 1714.6021729, -4125.2138672, 3709.0881348, -5510.8320312, 5839.8159180
1: -1589.0211182, 1684.1212158, -3325.1887207, 3650.1582031, -5097.4873047, 5009.3085938
2: -2363.9597168, 1821.3619385, -5013.2705078, 3923.4035645, -6133.8637695, 6815.6767578
3: -891.5067749, 2382.3999023, -1844.2091064, 5052.7412109, -5895.2758789, 4186.2783203
4: -2598.4758301, 1776.2236328, -5486.5805664, 3812.7636719, -6248.4799805, 7254.6347656

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2727350, upper bound: 3612.4516505
time: 0.72 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2727397, upper bound: 3612.9616848
time: 0.86 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1974.8409424, 1716.9732666, -4029.8068848, 3624.2827148, -5434.7851562, 5746.7802734
1: -1591.1787109, 1686.4777832, -3249.4353027, 3567.8945312, -5022.3764648, 4935.9121094
2: -2367.1381836, 1823.9008789, -4904.0380859, 3835.6140137, -6055.3159180, 6705.9472656
3: -892.7864990, 2385.7099609, -1800.9842529, 4939.2626953, -5782.4155273, 4149.1958008
4: -2601.9707031, 1778.6447754, -5364.5053711, 3725.9509277, -6172.4516602, 7131.6411133

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2727350, upper bound: 3612.5173456
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2727395, upper bound: 3612.5141650
time: 0.75 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.43 seconds
IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3612.8898792, upper bound: 3611.6939932
IS_A2_A1_A1_B2_A2_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3612.9454524, upper bound: 3611.7657368
IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.0999183, upper bound: 3611.6542573
IS_A2_A1_A1_B2_A2_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2821227, upper bound: 3611.6542718
IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2381109, upper bound: 3612.4516406
IS_A2_A1_A1_B2_A2_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2381153, upper bound: 3612.9612026
IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2381100, upper bound: 3612.5173308
IS_A2_A1_A1_B2_A2_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2381124, upper bound: 3612.5141496
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.4331667, upper bound: 3611.7721278
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.4527742, upper bound: 3611.7727771
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.4331667, upper bound: 3611.7720121
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.4527740, upper bound: 3611.7727682
IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2727350, upper bound: 3612.4516505
IS_A2_A1_A1_B2_A2_B2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2727397, upper bound: 3612.9616848
IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2727350, upper bound: 3612.5173456
IS_A2_A1_A1_B2_A2_B2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 10, time: 3.43
Output dim: 0, lower bound: -3613.2727395, upper bound: 3612.5141650

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -1939.8754883, 1684.8797607, -4188.6352539, 3784.6672363, -5539.3969727, 5873.5151367
1: -1563.0395508, 1654.7657471, -3375.2087402, 3725.6618652, -5135.0317383, 5029.9736328
2: -2325.5151367, 1789.0622559, -5089.8310547, 4008.2202148, -6165.1157227, 6847.9726562
3: -875.8602295, 2344.4206543, -1879.3211670, 5135.5410156, -5954.4931641, 4179.9794922
4: -2557.3989258, 1744.4675293, -5568.5136719, 3892.9260254, -6272.4770508, 7292.3784180

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1542968, upper bound: 3611.5181205
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4229990, upper bound: 3611.7669517
time: 0.82 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7418487, upper bound: 3611.6924690
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3492714, upper bound: 3611.7649460
time: 0.88 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1958.3699951, 1703.8114014, -4179.1411133, 3777.5427246, -5550.3100586, 5882.9526367
1: -1578.6502686, 1667.3239746, -3367.5698242, 3718.8703613, -5142.8803711, 5034.8935547
2: -2348.4206543, 1803.0620117, -5078.8940430, 4000.8166504, -6179.2656250, 6851.9448242
3: -883.0504150, 2367.0510254, -1875.3029785, 5124.8740234, -5950.7651367, 4197.0004883
4: -2581.6030273, 1756.6071777, -5556.3842773, 3885.6042480, -6288.6870117, 7294.6914062

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1888896, upper bound: 3611.5207365
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7948869, upper bound: 3611.6944120
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3149333, upper bound: 3611.7655941
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1836.8626709, 1591.5273438, -4192.5297852, 3788.1174316, -5442.8505859, 5784.0571289
1: -1481.0885010, 1561.5603027, -3378.2973633, 3729.0483398, -5058.4174805, 4939.8564453
2: -2207.0605469, 1688.5159912, -5094.3168945, 4011.8801270, -6053.0180664, 6766.3510742
3: -827.8502808, 2220.2121582, -1881.1254883, 5140.1596680, -5916.7089844, 4060.6093750
4: -2425.5727539, 1645.6485596, -5573.4379883, 3896.4602051, -6146.9833984, 7214.1513672

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1542968, upper bound: 3611.5180049
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5916992, upper bound: 3611.7714841
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5916992, upper bound: 3611.7720133
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1859.0166016, 1613.8051758, -4183.0810547, 3781.0847168, -5457.3671875, 5796.8862305
1: -1499.5905762, 1579.9342041, -3370.6911621, 3722.3557129, -5069.2343750, 4950.6250000
2: -2234.2780762, 1705.2036133, -5083.4326172, 4004.5749512, -6071.5375977, 6773.2504883
3: -836.6526489, 2247.6237793, -1877.1365967, 5129.5781250, -5914.2622070, 4082.4123535
4: -2454.4228516, 1659.8635254, -5561.3725586, 3889.2285156, -6167.7958984, 7219.2919922

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1888896, upper bound: 3611.5207342
time: 0.78 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4108901, upper bound: 3611.7689152
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3220614, upper bound: 3611.7716094
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3220617, upper bound: 3611.7727671
time: 0.84 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 9.66 seconds
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A1, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.7418487, upper bound: 3611.6924690
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 9.66
Output dim: 0, lower bound: -3613.3492714, upper bound: 3611.7649460
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.7948869, upper bound: 3611.6944120
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 9.66
Output dim: 0, lower bound: -3613.3149333, upper bound: 3611.7655941
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.5916992, upper bound: 3611.7714841
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.5916992, upper bound: 3611.7720133
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.3220614, upper bound: 3611.7716094
IS_A2_A1_A1_B2_A2_B2_A2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 11, time: 9.66
Output dim: 0, lower bound: -3612.3220617, upper bound: 3611.7727671

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -1953.4754639, 1698.8833008, -4183.6879883, 3780.3486328, -5548.8061523, 5882.5708008
1: -1574.0810547, 1668.6213379, -3371.2473145, 3721.3820801, -5141.5268555, 5039.8681641
2: -2342.0554199, 1803.6210938, -5084.1093750, 4003.5119629, -6176.5537109, 6857.2216797
3: -882.9833984, 2361.6035156, -1876.9501953, 5129.5263672, -5955.3959961, 4194.4194336
4: -2575.2687988, 1759.3178711, -5562.2783203, 3888.5061035, -6285.8183594, 7301.1582031

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0275400, upper bound: 3611.4786475
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3455588, upper bound: 3611.7597675
time: 0.73 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430794, upper bound: 3611.4945442
time: 0.92 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3156890, upper bound: 3611.4968812
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -1969.6729736, 1715.0246582, -4174.1333008, 3773.0576172, -5557.2895508, 5889.1577148
1: -1587.9559326, 1679.1789551, -3363.5651855, 3714.4165039, -5147.5214844, 5042.7441406
2: -2362.3776855, 1815.3627930, -5073.0957031, 3995.9375000, -6188.0327148, 6859.2998047
3: -889.2548828, 2381.9387207, -1872.8836670, 5118.7250977, -5950.6552734, 4209.0874023
4: -2596.6403809, 1767.9736328, -5550.0629883, 3881.0231934, -6299.1210938, 7301.4780273

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9994440, upper bound: 3611.4809970
time: 0.91 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3148884, upper bound: 3611.7655091
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3149193, upper bound: 3611.4945535
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3147539, upper bound: 3611.4975141
time: 0.85 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 15.59 seconds
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 12, time: 15.59
Output dim: 0, lower bound: -3613.3430794, upper bound: 3611.4945442
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 12, time: 15.59
Output dim: 0, lower bound: -3613.3156890, upper bound: 3611.4968812
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 12, time: 15.59
Output dim: 0, lower bound: -3613.3149193, upper bound: 3611.4945535
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 12, time: 15.59
Output dim: 0, lower bound: -3613.3147539, upper bound: 3611.4975141

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -1922.8616943, 1671.3262939, -4180.4858398, 3777.4819336, -5515.3476562, 5851.8115234
1: -1549.4771729, 1641.5109863, -3368.6782227, 3718.5698242, -5114.1010742, 5010.1894531
2: -2306.1970215, 1774.3907471, -5080.3823242, 4000.4868164, -6137.7075195, 6824.4072266
3: -868.9390869, 2324.9682617, -1875.4855957, 5125.6669922, -5937.5366211, 4156.2797852
4: -2535.8706055, 1730.4984131, -5558.1665039, 3885.5556641, -6243.5751953, 7268.2553711

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0218982, upper bound: 3611.2188286
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3410477, upper bound: 3611.4917126
time: 0.73 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430293, upper bound: 3611.4945395
time: 0.80 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3429565, upper bound: 3611.1182541
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3429565, upper bound: 3611.4945442
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -1938.6213379, 1687.3520508, -4176.8256836, 3774.5893555, -5527.7138672, 5864.1777344
1: -1562.9025879, 1652.6264648, -3365.7219238, 3715.7751465, -5124.3583984, 5018.3486328
2: -2326.4042969, 1785.0865479, -5076.0605469, 3997.5058594, -6154.2299805, 6832.9077148
3: -873.0864258, 2345.3659668, -1873.9593506, 5121.3837891, -5937.8432617, 4174.6030273
4: -2556.8542480, 1738.2724609, -5553.3730469, 3882.5852051, -6261.0815430, 7274.9667969

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0081171, upper bound: 3611.2212738
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3156386, upper bound: 3611.4968263
time: 0.78 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3155624, upper bound: 3611.1205925
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3155624, upper bound: 3611.4968826
time: 0.74 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -1938.4755859, 1688.5548096, -4170.9843750, 3770.2438965, -5523.2875977, 5859.5385742
1: -1562.8795166, 1653.3640137, -3361.0380859, 3711.6579590, -5119.6733398, 5014.4023438
2: -2325.8425293, 1786.1995850, -5069.4316406, 3992.9699707, -6148.5600586, 6826.3300781
3: -874.9345093, 2344.9584961, -1871.4443359, 5114.9335938, -5932.5834961, 4170.6152344
4: -2556.4851074, 1739.0479736, -5546.0190430, 3878.1284180, -6256.1625977, 7268.4331055

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9927105, upper bound: 3611.2188752
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3148689, upper bound: 3611.4945494
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3147925, upper bound: 3611.1182635
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3147925, upper bound: 3611.2694065
time: 1.69 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -1956.7026367, 1707.4906006, -4167.3544922, 3767.3488770, -5538.1533203, 5874.8452148
1: -1578.1352539, 1671.3721924, -3358.1057129, 3708.8571777, -5131.7895508, 5029.4780273
2: -2348.8227539, 1801.2366943, -5065.1430664, 3989.9836426, -6167.8901367, 6836.1689453
3: -879.5598145, 2368.3505859, -1869.9232178, 5110.6660156, -5933.4047852, 4191.9716797
4: -2580.5351562, 1756.0341797, -5541.2602539, 3875.1562500, -6276.7348633, 7276.5932617

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9994440, upper bound: 3611.2238331
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3147036, upper bound: 3611.4974291
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3146272, upper bound: 3611.1212255
time: 0.81 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3146272, upper bound: 3611.4975141
time: 0.89 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 15.95 seconds
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3429565, upper bound: 3611.1182541
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3429565, upper bound: 3611.4945442
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3155624, upper bound: 3611.1205925
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3155624, upper bound: 3611.4968826
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3147925, upper bound: 3611.1182635
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3147925, upper bound: 3611.2694065
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3146272, upper bound: 3611.1212255
IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 13, time: 15.95
Output dim: 0, lower bound: -3613.3146272, upper bound: 3611.4975141

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1922.8616943, 1671.3262939, -4111.1201172, 3720.1408691, -5452.5810547, 5782.4462891
1: -1549.4771729, 1641.5109863, -3312.9382324, 3663.5229492, -5054.1582031, 4954.4492188
2: -2306.1970215, 1774.3907471, -5001.1518555, 3940.6999512, -6072.0410156, 6741.3129883
3: -868.9390869, 2324.9682617, -1844.7463379, 5045.9829102, -5854.7495117, 4122.2807617
4: -2535.8706055, 1730.4984131, -5470.5522461, 3827.0551758, -6179.4213867, 7176.3403320

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0218200, upper bound: 3610.8139470
time: 0.87 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3409954, upper bound: 3611.1154181
time: 0.94 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3429153, upper bound: 3611.1182495
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3286926, upper bound: 3610.6384615
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3307470, upper bound: 3611.1182541
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1922.8616943, 1671.3262939, -4137.3232422, 3747.2163086, -5475.8725586, 5808.6494141
1: -1549.4771729, 1641.5109863, -3334.3752441, 3689.6540527, -5076.9443359, 4975.8862305
2: -2306.1970215, 1774.3907471, -5031.7812500, 3969.2153320, -6096.6411133, 6771.8862305
3: -868.9390869, 2324.9682617, -1857.2689209, 5079.4204102, -5886.8657227, 4133.9365234
4: -2535.8706055, 1730.4984131, -5502.9072266, 3855.0319824, -6203.2709961, 7209.2495117

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0218200, upper bound: 3611.2188303
time: 0.93 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3409954, upper bound: 3611.4917126
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3429153, upper bound: 3611.4945381
time: 0.68 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_A1_B2_A2_B2_A2_B1_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=4653.345703125
rel_dist={0: [-3614.7615518272232, 3614.7615518272232]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0483754, upper bound: 3614.5921021
time: 0.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.7410835, upper bound: 3614.7410842
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -3613.0483754, upper bound: 3614.5921021
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -3614.7410835, upper bound: 3614.7410842

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2464.8896484, 2121.3957520, -4294.1435547, 4356.8164062
1: -1744.2438965, 1855.7540283, -1984.8518066, 2078.8129883, -3823.0568848, 3840.6059570
2: -2557.1499023, 2009.1342773, -2929.2561035, 2254.1416016, -4811.2915039, 4938.3896484
3: -990.0590820, 2595.9711914, -1111.2934570, 2939.0102539, -3929.0693359, 3707.2646484
4: -2815.3046875, 1956.6667480, -3223.0842285, 2196.0478516, -5011.3525391, 5179.7504883

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0462037, upper bound: 3613.0462037
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0462037, upper bound: 3614.5921021
time: 0.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -2500.9851074, 2149.7963867, -2502.3757324, 2150.9697266, -4651.9545898, 4652.1718750
1: -2014.5847168, 2106.3789062, -2015.7012939, 2107.5605469, -4122.1455078, 4122.0800781
2: -2976.4731445, 2283.9982910, -2978.1022949, 2285.2739258, -5261.7460938, 5262.0996094
3: -1125.7036133, 2982.2333984, -1126.3112793, 2983.9003906, -4109.6040039, 4108.5449219
4: -3274.6416016, 2225.7395020, -3276.4265137, 2226.9489746, -5501.5903320, 5502.1655273

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5921021, upper bound: 3613.0483754
time: 0.93 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5921021, upper bound: 3614.7410842
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.47 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.0462037, upper bound: 3613.0462037
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3613.0462037, upper bound: 3614.5921021
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3614.5921021, upper bound: 3613.0483754
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 0, lower bound: -3614.5921021, upper bound: 3614.7410842

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2486.5666504, 2138.6867676, -4311.4340820, 4378.4931641
1: -1744.2438965, 1855.7540283, -2002.6257324, 2095.6591797, -3839.9030762, 3858.3798828
2: -2557.1499023, 2009.1342773, -2957.1662598, 2272.3825684, -4829.5322266, 4966.3007812
3: -990.0590820, 2595.9711914, -1120.1232910, 2964.8525391, -3954.9116211, 3716.0944824
4: -2815.3046875, 1956.6667480, -3253.6584473, 2214.0566406, -5029.3598633, 5210.3251953

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9434662, upper bound: 3614.4412417
time: 0.73 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9434455, upper bound: 3614.5341154
time: 0.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -2486.5668945, 2138.6870117, -2172.7478027, 1891.9267578, -4378.4936523, 4311.4340820
1: -2002.6257324, 2095.6591797, -1744.2438965, 1855.7540283, -3858.3798828, 3839.9028320
2: -2957.1667480, 2272.3825684, -2557.1499023, 2009.1342773, -4966.3002930, 4829.5322266
3: -1120.1232910, 2964.8525391, -990.0590820, 2595.9711914, -3716.0944824, 3954.9116211
4: -3253.6586914, 2214.0566406, -2815.3046875, 1956.6667480, -5210.3251953, 5029.3603516

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4412377, upper bound: 3612.9455803
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5341111, upper bound: 3612.9457070
time: 0.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -2500.9851074, 2149.7963867, -2500.9851074, 2149.7963867, -4650.7812500, 4650.7812500
1: -2014.5847168, 2106.3789062, -2014.5847168, 2106.3789062, -4120.9638672, 4120.9638672
2: -2976.4731445, 2283.9982910, -2976.4731445, 2283.9982910, -5260.4716797, 5260.4711914
3: -1125.7036133, 2982.2333984, -1125.7036133, 2982.2333984, -4107.9370117, 4107.9370117
4: -3274.6416016, 2225.7395020, -3274.6416016, 2225.7395020, -5500.3803711, 5500.3803711

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0435825, upper bound: 3613.6737203
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6715273, upper bound: 3613.6737138
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.50 seconds
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3612.9434662, upper bound: 3614.4412417
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3612.9434455, upper bound: 3614.5341154
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3614.4412377, upper bound: 3612.9455803
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3614.5341111, upper bound: 3612.9457070
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3614.0435825, upper bound: 3613.6737203
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.50
Output dim: 0, lower bound: -3613.6715273, upper bound: 3613.6737138

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2165.9155273, 1885.7924805, -2387.7915039, 2049.8315430, -4215.7465820, 4273.5839844
1: -1738.7598877, 1849.6748047, -1922.9685059, 2007.4107666, -3746.1706543, 3772.6430664
2: -2549.0258789, 2002.5859375, -2838.2163086, 2177.9599609, -4726.9853516, 4840.8022461
3: -986.8766479, 2587.6313477, -1074.4951172, 2842.6259766, -3829.5026855, 3662.1262207
4: -2806.3664551, 1950.3398438, -3122.8662109, 2122.0007324, -4928.3671875, 5073.2060547

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5727301, upper bound: 3614.1332942
time: 0.89 seconds

## Relational analysis of IS_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9453781, upper bound: 3614.3741973
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9453575, upper bound: 3614.4412377
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2469.5837402, 2123.7131348, -4296.4604492, 4361.5102539
1: -1744.2438965, 1855.7540283, -1988.9925537, 2080.8293457, -3825.0732422, 3844.7465820
2: -2557.1499023, 2009.1342773, -2937.1555176, 2256.4079590, -4813.5576172, 4946.2900391
3: -990.0590820, 2595.9711914, -1112.3453369, 2944.5651855, -3934.6240234, 3708.3161621
4: -2815.3046875, 1956.6667480, -3231.6235352, 2198.5351562, -5013.8388672, 5188.2900391

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8181391, upper bound: 3613.7167255
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9456960, upper bound: 3614.5341053
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2387.7915039, 2049.8315430, -2165.9155273, 1885.7924805, -4273.5839844, 4215.7465820
1: -1922.9685059, 2007.4107666, -1738.7598877, 1849.6748047, -3772.6433105, 3746.1706543
2: -2838.2163086, 2177.9599609, -2549.0258789, 2002.5859375, -4840.8022461, 4726.9853516
3: -1074.4951172, 2842.6259766, -986.8766479, 2587.6313477, -3662.1262207, 3829.5026855
4: -3122.8662109, 2122.0007324, -2806.3664551, 1950.3398438, -5073.2060547, 4928.3671875

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1332941, upper bound: 3612.5727301
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.3741973, upper bound: 3612.9453781
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4412377, upper bound: 3612.9453575
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2469.5837402, 2123.7131348, -2172.7478027, 1891.9267578, -4361.5102539, 4296.4604492
1: -1988.9925537, 2080.8293457, -1744.2438965, 1855.7540283, -3844.7465820, 3825.0732422
2: -2937.1555176, 2256.4079590, -2557.1499023, 2009.1342773, -4946.2900391, 4813.5576172
3: -1112.3453369, 2944.5651855, -990.0590820, 2595.9711914, -3708.3161621, 3934.6237793
4: -3231.6235352, 2198.5351562, -2815.3046875, 1956.6667480, -5188.2900391, 5013.8388672

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7167255, upper bound: 3612.8181391
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.9456960
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -2271.4199219, 1964.2563477, -2500.9851074, 2149.7963867, -4421.2163086, 4465.2407227
1: -1830.2175293, 1926.7331543, -2014.5847168, 2106.3789062, -3936.5964355, 3941.3178711
2: -2710.7407227, 2087.3911133, -2976.4731445, 2283.9982910, -4994.7392578, 5063.8627930
3: -1026.1483154, 2722.9445801, -1125.7036133, 2982.2333984, -4008.3818359, 3848.6481934
4: -2980.7939453, 2034.3486328, -3274.6416016, 2225.7395020, -5206.5322266, 5308.9902344

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209437, upper bound: 3612.9865603
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946496
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4050.7668457, 3654.0148926, -2472.9548340, 2129.6313477, -6180.3979492, 5954.5102539
1: -3265.9843750, 3597.2951660, -1992.5217285, 2086.7145996, -5352.6992188, 5446.2368164
2: -4927.0400391, 3867.2863770, -2944.8867188, 2262.6599121, -7161.0864258, 6652.6411133
3: -1814.2457275, 4968.8481445, -1114.9156494, 2952.9328613, -4725.3315430, 6029.5385742
4: -5389.4619141, 3757.1284180, -3239.4138184, 2204.9694824, -7577.3750000, 6828.7993164

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0637048, upper bound: 3612.9865508
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3936351, upper bound: 3613.3946425
time: 0.87 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.51 seconds
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9453781, upper bound: 3614.3741973
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9453575, upper bound: 3614.4412377
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.8181391, upper bound: 3613.7167255
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9456960, upper bound: 3614.5341053
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3614.3741973, upper bound: 3612.9453781
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3614.4412377, upper bound: 3612.9453575
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.7167255, upper bound: 3612.8181391
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.9456960
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9209437, upper bound: 3612.9865603
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946496
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.0637048, upper bound: 3612.9865508
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.3936351, upper bound: 3613.3946425

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2105.3405762, 1834.7270508, -2141.9545898, 1835.3210449, -3940.6613770, 3976.6806641
1: -1690.3289795, 1799.9450684, -1725.0408936, 1795.6173096, -3485.9460449, 3524.9858398
2: -2478.3439941, 1948.9697266, -2549.2658691, 1949.4248047, -4427.7685547, 4498.2353516
3: -959.5479126, 2515.9379883, -961.4118652, 2543.8723145, -3503.4201660, 3477.3498535
4: -2728.0854492, 1898.0230713, -2803.5693359, 1899.5701904, -4627.6557617, 4701.5917969

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5724992, upper bound: 3614.0492920
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2683613, upper bound: 3614.1497614
time: 0.80 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5956089, upper bound: 3614.1498117
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2149.4924316, 1871.9265137, -2345.7243652, 2013.4931641, -4162.9848633, 4217.6508789
1: -1725.6081543, 1836.0386963, -1889.3408203, 1971.7060547, -3697.3142090, 3725.3793945
2: -2529.7453613, 1987.8299561, -2789.2082520, 2139.0021973, -4668.7475586, 4777.0380859
3: -979.3604126, 2568.2536621, -1054.9140625, 2793.0563965, -3772.4167480, 3623.1677246
4: -2785.1750488, 1936.0506592, -3068.9025879, 2084.4362793, -4869.6113281, 5004.9531250

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5725129, upper bound: 3614.1331729
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7874078, upper bound: 3614.4408692
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7874078, upper bound: 3614.4412377
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2161.3437500, 1882.1640625, -2558.5917969, 2213.0500488, -4374.3935547, 4440.7558594
1: -1735.0933838, 1846.1088867, -2060.5642090, 2167.4465332, -3902.5400391, 3906.6728516
2: -2543.6848145, 1998.6596680, -3042.9968262, 2349.8449707, -4893.5292969, 5041.6557617
3: -984.8987427, 2582.4138184, -1154.4128418, 3054.9592285, -4039.8579102, 3736.8266602
4: -2800.5014648, 1946.5699463, -3348.0634766, 2292.7133789, -5093.2148438, 5294.6328125

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7659803, upper bound: 3613.3010119
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8181391, upper bound: 3612.7036195
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2172.7478027, 1891.9267578, -2462.8464355, 2117.9519043, -4290.6997070, 4354.7734375
1: -1744.2438965, 1855.7540283, -1983.6042480, 2075.1950684, -3819.4389648, 3839.3581543
2: -2557.1499023, 2009.1342773, -2929.3710938, 2250.2844238, -4807.4345703, 4938.5048828
3: -990.0590820, 2595.9711914, -1109.3238525, 2936.6984863, -3926.7575684, 3705.2949219
4: -2815.3046875, 1956.6667480, -3222.9858398, 2192.6169434, -5007.9213867, 5179.6523438

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.5341053
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5341096
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -2141.9548340, 1835.3212891, -2105.3405762, 1834.7270508, -3976.6813965, 3940.6611328
1: -1725.0412598, 1795.6174316, -1690.3289795, 1799.9450684, -3524.9863281, 3485.9462891
2: -2549.2661133, 1949.4248047, -2478.3439941, 1948.9697266, -4498.2358398, 4427.7685547
3: -961.4119263, 2543.8728027, -959.5479126, 2515.9379883, -3477.3498535, 3503.4206543
4: -2803.5698242, 1899.5704346, -2728.0854492, 1898.0230713, -4701.5922852, 4627.6557617

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0492860, upper bound: 3612.5724992
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1497614, upper bound: 3612.2683613
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1498117, upper bound: 3612.5956089
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -2345.7243652, 2013.4931641, -2149.4924316, 1871.9265137, -4217.6508789, 4162.9853516
1: -1889.3408203, 1971.7060547, -1725.6081543, 1836.0386963, -3725.3793945, 3697.3142090
2: -2789.2082520, 2139.0021973, -2529.7453613, 1987.8299561, -4777.0380859, 4668.7475586
3: -1054.9140625, 2793.0563965, -979.3604126, 2568.2536621, -3623.1677246, 3772.4167480
4: -3068.9025879, 2084.4362793, -2785.1750488, 1936.0506592, -5004.9531250, 4869.6113281

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1331776, upper bound: 3612.5725129
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4408678, upper bound: 3612.7874078
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.4408678, upper bound: 3612.9453575
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -2558.5917969, 2213.0500488, -2161.3437500, 1882.1640625, -4440.7558594, 4374.3935547
1: -2060.5642090, 2167.4465332, -1735.0933838, 1846.1088867, -3906.6728516, 3902.5400391
2: -3042.9968262, 2349.8449707, -2543.6848145, 1998.6596680, -5041.6557617, 4893.5292969
3: -1154.4128418, 3054.9592285, -984.8987427, 2582.4138184, -3736.8266602, 4039.8579102
4: -3348.0634766, 2292.7133789, -2800.5014648, 1946.5699463, -5294.6328125, 5093.2148438

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3010078, upper bound: 3612.7659803
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7036208, upper bound: 3612.8181391
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -2462.8464355, 2117.9519043, -2172.7478027, 1891.9267578, -4354.7734375, 4290.6997070
1: -1983.6042480, 2075.1950684, -1744.2438965, 1855.7540283, -3839.3581543, 3819.4387207
2: -2929.3710938, 2250.2844238, -2557.1499023, 2009.1342773, -4938.5048828, 4807.4345703
3: -1109.3238525, 2936.6984863, -990.0590820, 2595.9711914, -3705.2949219, 3926.7575684
4: -3222.9858398, 2192.6169434, -2815.3046875, 1956.6667480, -5179.6523438, 5007.9218750

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.5730209
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.5729872
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2228.4621582, 1929.0208740, -4049.2326660, 3626.5078125, -5704.7202148, 5978.2534180
1: -1795.9824219, 1892.2448730, -3265.7373047, 3569.0261230, -5239.8959961, 5157.9824219
2: -2660.9748535, 2050.0754395, -4912.3940430, 3838.7568359, -6361.5717773, 6955.9633789
3: -1007.4152222, 2673.7551270, -1809.0644531, 4937.6064453, -5915.4951172, 4446.0932617
4: -2925.4301758, 1997.8076172, -5369.5214844, 3729.6098633, -6511.6225586, 7367.3291016

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946516
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946516
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4050.4179688, 3655.4787598, -4046.6298828, 3624.4450684, -7452.0561523, 7464.8994141
1: -3265.7065430, 3599.0815430, -3263.8991699, 3566.6010742, -6640.2553711, 6660.9487305
2: -4927.5507812, 3869.0996094, -4909.8535156, 3836.0556641, -8486.0214844, 8499.3603516
3: -1814.3466797, 4969.7260742, -1807.2124023, 4934.3696289, -6617.1333008, 6625.6088867
4: -5389.8920898, 3758.7514648, -5366.8544922, 3727.3874512, -8839.8710938, 8844.4628906

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9865105, upper bound: 3613.3946381
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9865105, upper bound: 3613.3946445
time: 1.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.63 seconds
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.2683613, upper bound: 3614.1497614
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.5956089, upper bound: 3614.1498117
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.7874078, upper bound: 3614.4408692
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.7874078, upper bound: 3614.4412377
IS_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.7659803, upper bound: 3613.3010119
IS_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.8181391, upper bound: 3612.7036195
IS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.5341053
IS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5341096
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.1497614, upper bound: 3612.2683613
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.1498117, upper bound: 3612.5956089
IS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.4408678, upper bound: 3612.7874078
IS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.4408678, upper bound: 3612.9453575
IS_A2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 0, lower bound: -3613.3010078, upper bound: 3612.7659803
IS_A2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.7036208, upper bound: 3612.8181391
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.5730209
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3614.5341053, upper bound: 3612.5729872
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946516
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.9209512, upper bound: 3613.3946516
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.9865105, upper bound: 3613.3946381
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 0, lower bound: -3612.9865105, upper bound: 3613.3946445

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1945.4641113, 1692.4307861, -2086.4152832, 1786.7084961, -3732.1726074, 3778.8457031
1: -1561.5529785, 1659.3345947, -1680.3137207, 1747.5551758, -3309.1081543, 3339.6479492
2: -2290.3945312, 1797.9466553, -2483.3825684, 1897.6339111, -4188.0283203, 4281.3291016
3: -886.5736694, 2323.7197266, -936.0081787, 2477.0488281, -3363.6220703, 3259.7275391
4: -2522.0895996, 1751.0017090, -2731.4265137, 1849.0607910, -4371.1499023, 4482.4282227

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8517568, upper bound: 3613.7421156
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2683613, upper bound: 3613.6279643
time: 0.88 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2683572, upper bound: 3613.2289236
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -2069.0957031, 1803.6733398, -2124.3757324, 1820.7862549, -3889.8818359, 3928.0488281
1: -1661.0068359, 1769.8718262, -1710.8574219, 1781.4042969, -3442.4106445, 3480.7287598
2: -2435.4772949, 1916.4532471, -2528.8212891, 1934.0708008, -4369.5478516, 4445.2744141
3: -943.4463501, 2473.7744141, -953.7944946, 2523.5544434, -3467.0004883, 3427.5683594
4: -2681.1193848, 1865.8546143, -2781.1242676, 1884.5234375, -4565.6425781, 4646.9785156

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8421590, upper bound: 3613.7428876
time: 0.92 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5956089, upper bound: 3613.6279948
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5955984, upper bound: 3613.2289376
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1931.3188477, 1677.3739014, -2345.6459961, 2013.4422607, -3944.7600098, 4023.0200195
1: -1549.8355713, 1644.7679443, -1889.2794189, 1971.6561279, -3521.4916992, 3534.0468750
2: -2272.2954102, 1782.4145508, -2789.1215820, 2138.9494629, -4411.2451172, 4571.5361328
3: -878.4545288, 2300.4899902, -1054.8847656, 2792.9724121, -3671.4262695, 3355.3747559
4: -2500.9545898, 1734.1303711, -3068.8073730, 2084.3842773, -4585.3383789, 4802.9375000

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7874078, upper bound: 3613.8947430
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7874060, upper bound: 3613.5036760
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -2122.8266602, 1849.2138672, -2345.6164551, 2013.4104004, -4136.2368164, 4194.8300781
1: -1704.2628174, 1813.7387695, -1889.2519531, 1971.6265869, -3675.8891602, 3702.9907227
2: -2498.4062500, 1963.6425781, -2789.0683594, 2138.9160156, -4637.3222656, 4752.7109375
3: -967.1054077, 2536.7055664, -1054.8717041, 2792.9277344, -3760.0332031, 3591.5771484
4: -2750.7316895, 1912.6314697, -3068.7497559, 2084.3496094, -4835.0810547, 4981.3813477

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5719521, upper bound: 3614.1328906
time: 0.87 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2683553, upper bound: 3614.1497614
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955495, upper bound: 3614.1498117
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -2159.2155762, 1880.2059326, -2485.3222656, 2134.3266602, -4293.5419922, 4365.5283203
1: -1733.4216309, 1844.2053223, -2001.1912842, 2090.3337402, -3823.7553711, 3845.3964844
2: -2541.3471680, 1996.6114502, -2953.8076172, 2266.0375977, -4807.3847656, 4950.4189453
3: -983.7364502, 2579.6550293, -1117.3603516, 2960.7731934, -3944.5097656, 3697.0151367
4: -2797.9587402, 1944.6219482, -3251.3383789, 2208.3642578, -5006.3222656, 5195.9604492

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.0303613
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5730118, upper bound: 3613.6047026
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -2150.8366699, 1870.3956299, -2391.8261719, 2050.3837891, -4201.2207031, 4262.2216797
1: -1726.7579346, 1834.4456787, -1926.9498291, 2008.6407471, -3735.3986816, 3761.3947754
2: -2531.5119629, 1986.2237549, -2846.4287109, 2178.8039551, -4710.3144531, 4832.6523438
3: -979.4531250, 2568.5395508, -1076.8668213, 2849.2890625, -3828.7421875, 3645.4062500
4: -2787.1047363, 1934.2463379, -3131.3664551, 2121.8266602, -4908.9316406, 5065.6127930

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5335303
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5341053
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -2086.4152832, 1786.7084961, -1945.4641113, 1692.4307861, -3778.8457031, 3732.1726074
1: -1680.3137207, 1747.5551758, -1561.5529785, 1659.3345947, -3339.6479492, 3309.1081543
2: -2483.3825684, 1897.6339111, -2290.3945312, 1797.9466553, -4281.3291016, 4188.0283203
3: -936.0081787, 2477.0488281, -886.5736694, 2323.7197266, -3259.7275391, 3363.6220703
4: -2731.4265137, 1849.0607910, -2522.0895996, 1751.0017090, -4482.4282227, 4371.1499023

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7421097, upper bound: 3611.8517568
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6279643, upper bound: 3612.2683613
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2289235, upper bound: 3612.2683513
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2124.3757324, 1820.7863770, -2069.0957031, 1803.6733398, -3928.0488281, 3889.8820801
1: -1710.8575439, 1781.4042969, -1661.0068359, 1769.8718262, -3480.7290039, 3442.4108887
2: -2528.8212891, 1934.0708008, -2435.4772949, 1916.4532471, -4445.2744141, 4369.5478516
3: -953.7944946, 2523.5544434, -943.4463501, 2473.7744141, -3427.5688477, 3467.0004883
4: -2781.1245117, 1884.5233154, -2681.1193848, 1865.8546143, -4646.9790039, 4565.6420898

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7428817, upper bound: 3611.8421590
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6279948, upper bound: 3612.5956089
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2289376, upper bound: 3612.5955984
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -2345.6462402, 2013.4422607, -1931.3188477, 1677.3739014, -4023.0200195, 3944.7602539
1: -1889.2795410, 1971.6561279, -1549.8355713, 1644.7679443, -3534.0468750, 3521.4916992
2: -2789.1218262, 2138.9489746, -2272.2954102, 1782.4145508, -4571.5361328, 4411.2441406
3: -1054.8847656, 2792.9724121, -878.4545288, 2300.4899902, -3355.3747559, 3671.4265137
4: -3068.8076172, 2084.3842773, -2500.9545898, 1734.1303711, -4802.9375000, 4585.3383789

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.8947430, upper bound: 3612.7874078
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5036760, upper bound: 3612.7874060
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -2345.6164551, 2013.4104004, -2122.8266602, 1849.2138672, -4194.8300781, 4136.2368164
1: -1889.2519531, 1971.6265869, -1704.2628174, 1813.7387695, -3702.9907227, 3675.8891602
2: -2789.0683594, 2138.9160156, -2498.4062500, 1963.6425781, -4752.7109375, 4637.3222656
3: -1054.8717041, 2792.9277344, -967.1054077, 2536.7055664, -3591.5771484, 3760.0332031
4: -3068.7497559, 2084.3496094, -2750.7316895, 1912.6314697, -4981.3813477, 4835.0810547

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0667814, upper bound: 3612.5725118
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6658570, upper bound: 3612.2683744
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1495941, upper bound: 3612.5955495
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -2485.3222656, 2134.3266602, -2159.2155762, 1880.2059326, -4365.5283203, 4293.5419922
1: -2001.1912842, 2090.3337402, -1733.4216309, 1844.2053223, -3845.3964844, 3823.7553711
2: -2953.8076172, 2266.0375977, -2541.3471680, 1996.6114502, -4950.4189453, 4807.3847656
3: -1117.3603516, 2960.7731934, -983.7364502, 2579.6550293, -3697.0153809, 3944.5095215
4: -3251.3383789, 2208.3642578, -2797.9587402, 1944.6219482, -5195.9604492, 5006.3227539

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0303595, upper bound: 3612.5730209
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.6047026, upper bound: 3612.5730107
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -2391.8259277, 2050.3835449, -2150.8366699, 1870.3956299, -4262.2216797, 4201.2202148
1: -1926.9499512, 2008.6405029, -1726.7579346, 1834.4456787, -3761.3947754, 3735.3984375
2: -2846.4287109, 2178.8037109, -2531.5119629, 1986.2237549, -4832.6523438, 4710.3149414
3: -1076.8668213, 2849.2888184, -979.4531250, 2568.5395508, -3645.4062500, 3828.7419434
4: -3131.3664551, 2121.8266602, -2787.1047363, 1934.2463379, -5065.6127930, 4908.9316406

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5335260, upper bound: 3612.5729861
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.5335260, upper bound: 3612.5729861
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2032.5021973, 1767.8198242, -4046.8234863, 3624.2609863, -5509.8520508, 5814.6435547
1: -1637.2110596, 1736.3991699, -3263.7990723, 3566.7893066, -5081.6040039, 5000.1982422
2: -2434.3435059, 1878.1309814, -4909.4448242, 3836.3618164, -6138.0000000, 6787.5756836
3: -919.7875977, 2453.9553223, -1807.9580078, 4934.6020508, -5827.6923828, 4228.4921875
4: -2675.7617188, 1831.0811768, -5366.2963867, 3727.2829590, -6265.0683594, 7197.3774414

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8672840, upper bound: 3613.2902451
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9200132, upper bound: 3613.2903748
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3858.8288574, 3476.9750977, -4055.4401855, 3631.8959961, -7265.7617188, 7301.8989258
1: -3110.0395508, 3427.8459473, -3270.7504883, 3574.3640137, -6493.4716797, 6501.0273438
2: -4695.0322266, 3685.5327148, -4920.3764648, 3844.4882812, -8264.7695312, 8328.5986328
3: -1728.6866455, 4724.4501953, -1811.6080322, 4945.1596680, -6538.1000977, 6398.7958984
4: -5128.0468750, 3577.4528809, -5378.1831055, 3735.1677246, -8588.1787109, 8680.6074219

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209437, upper bound: 3612.9209597
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9209437, upper bound: 3613.0541488
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3842.9118652, 3492.4929199, -4046.6298828, 3624.4450684, -7232.2778320, 7282.9086914
1: -3097.1203613, 3444.2551270, -3263.8991699, 3566.6010742, -6462.0961914, 6488.6684570
2: -4690.9912109, 3699.9401855, -4909.8535156, 3836.0556641, -8231.2998047, 8308.8574219
3: -1723.5593262, 4738.4677734, -1807.2124023, 4934.3696289, -6516.5209961, 6382.4741211
4: -5127.6469727, 3592.6191406, -5366.8544922, 3727.3874512, -8557.3281250, 8658.2763672

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9538548, upper bound: 3612.9209431
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9538510, upper bound: 3613.0391644
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6016.1181641, 5522.2851562, -4046.6298828, 3624.4450684, -9278.3671875, 8939.5839844
1: -4844.8808594, 5454.9628906, -3263.8991699, 3566.6010742, -8102.5893555, 8181.6523438
2: -7361.7265625, 5863.3276367, -4909.8535156, 3836.0556641, -10687.2724609, 10083.7949219
3: -2705.0969238, 7430.4960938, -1807.2124023, 4934.3696289, -7345.7758789, 8915.3427734
4: -8030.8833008, 5678.8281250, -5366.8544922, 3727.3874512, -11243.7978516, 10356.2294922

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9538547, upper bound: 3612.9209495
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9538548, upper bound: 3613.3935581
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.52 seconds
IS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.2683613, upper bound: 3613.6279643
IS_A1_B2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.2683572, upper bound: 3613.2289236
IS_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5956089, upper bound: 3613.6279948
IS_A1_B2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5955984, upper bound: 3613.2289376
IS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.7874078, upper bound: 3613.8947430
IS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.7874060, upper bound: 3613.5036760
IS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.2683553, upper bound: 3614.1497614
IS_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5955495, upper bound: 3614.1498117
IS_A1_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.0303613
IS_A1_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5730118, upper bound: 3613.6047026
IS_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5335303
IS_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.5729872, upper bound: 3614.5341053
IS_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.6279643, upper bound: 3612.2683613
IS_A2_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.2289235, upper bound: 3612.2683513
IS_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.6279948, upper bound: 3612.5956089
IS_A2_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.2289376, upper bound: 3612.5955984
IS_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.8947430, upper bound: 3612.7874078
IS_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.5036760, upper bound: 3612.7874060
IS_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.6658570, upper bound: 3612.2683744
IS_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3614.1495941, upper bound: 3612.5955495
IS_A2_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3614.0303595, upper bound: 3612.5730209
IS_A2_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3613.6047026, upper bound: 3612.5730107
IS_A2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3614.5335260, upper bound: 3612.5729861
IS_A2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3614.5335260, upper bound: 3612.5729861
IS_A2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.8672840, upper bound: 3613.2902451
IS_A2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9200132, upper bound: 3613.2903748
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9209437, upper bound: 3612.9209597
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9209437, upper bound: 3613.0541488
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9538548, upper bound: 3612.9209431
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9538510, upper bound: 3613.0391644
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9538547, upper bound: 3612.9209495
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.52
Output dim: 0, lower bound: -3612.9538548, upper bound: 3613.3935581

## BFS IS instance: IS_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1945.4641113, 1692.4307861, -1872.9016113, 1613.1390381, -3558.6027832, 3565.3325195
1: -1561.5529785, 1659.3345947, -1509.7269287, 1580.2980957, -3141.8508301, 3169.0615234
2: -2290.3945312, 1797.9466553, -2238.1496582, 1714.9681396, -4005.3625488, 4036.0961914
3: -886.5736694, 2323.7197266, -843.5189209, 2237.2312012, -3123.8049316, 3167.2382812
4: -2522.0895996, 1751.0017090, -2460.2197266, 1670.3427734, -4192.4321289, 4211.2216797

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8444678, upper bound: 3613.3550183
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2683613, upper bound: 3613.2943193
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2683613, upper bound: 3613.6279643
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -2069.0957031, 1803.6733398, -1911.2631836, 1647.5620117, -3716.6577148, 3714.9360352
1: -1661.0068359, 1769.8718262, -1540.7338867, 1614.5070801, -3275.5139160, 3310.6057129
2: -2435.4772949, 1916.4532471, -2284.4213867, 1751.6201172, -4187.0971680, 4200.8745117
3: -943.4463501, 2473.7744141, -861.4611816, 2284.5644531, -3228.0100098, 3335.2353516
4: -2681.1193848, 1865.8546143, -2510.8459473, 1706.2406006, -4387.3593750, 4376.6992188

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8421590, upper bound: 3613.3557211
time: 0.75 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5956089, upper bound: 3613.6277604
time: 0.79 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5956089, upper bound: 3613.6279948
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1931.3188477, 1677.3739014, -2127.7014160, 1837.3172607, -3768.6354980, 3805.0751953
1: -1549.8355713, 1644.7679443, -1715.0382080, 1801.6431885, -3351.4787598, 3359.8061523
2: -2272.2954102, 1782.4145508, -2538.8828125, 1953.1657715, -4225.4604492, 4321.2973633
3: -878.4545288, 2300.4899902, -960.6401367, 2548.2338867, -3426.6882324, 3261.1301270
4: -2500.9545898, 1734.1303711, -2792.4162598, 1903.1232910, -4404.0781250, 4526.5458984

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5719750, upper bound: 3613.6196650
time: 0.77 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2683744, upper bound: 3613.2981345
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955495, upper bound: 3613.6394748
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1898.4571533, 1650.6470947, -3812.6015625, 3451.1047363, -5155.8593750, 5447.3632812
1: -1523.5505371, 1618.6271973, -3074.3764648, 3396.5749512, -4763.1938477, 4667.2983398
2: -2234.0625000, 1753.9880371, -4637.6699219, 3652.8068848, -5708.5156250, 6301.3383789
3: -864.0112305, 2262.5676270, -1711.3515625, 4680.4345703, -5458.6142578, 3929.0583496
4: -2458.5839844, 1706.4154053, -5069.9877930, 3548.5615234, -5822.3691406, 6695.7416992

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5719737, upper bound: 3613.1215257
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5951870, upper bound: 3612.9823496
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5955447, upper bound: 3613.2714865
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -1922.3486328, 1672.6491699, -2267.0883789, 1945.4528809, -3867.8015137, 3939.7375488
1: -1542.9865723, 1640.1636963, -1825.5111084, 1904.4765625, -3447.4631348, 3465.6745605
2: -2263.1162109, 1777.0783691, -2695.8100586, 2066.6315918, -4329.7475586, 4472.8886719
3: -876.0330811, 2295.9829102, -1019.6576538, 2698.7067871, -3574.7395020, 3315.6406250
4: -2492.1264648, 1730.9958496, -2966.5590820, 2013.8104248, -4505.9365234, 4697.5546875

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.2683553, upper bound: 3613.6396340
time: 0.84 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.2683526, upper bound: 3613.2715960
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -2084.4929199, 1816.5424805, -2333.3608398, 2003.0670166, -4087.5598145, 4149.9033203
1: -1673.2673340, 1782.1115723, -1879.4733887, 1961.5216064, -3634.7890625, 3661.5849609
2: -2453.0874023, 1929.4256592, -2774.8730469, 2127.9924316, -4581.0795898, 4704.2988281
3: -950.0939941, 2492.1274414, -1049.4366455, 2778.7492676, -3728.8432617, 3541.5639648
4: -2701.0148926, 1878.8061523, -3053.1311035, 2073.6096191, -4774.6245117, 4931.9375000

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8421526, upper bound: 3613.7579926
time: 0.86 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5955495, upper bound: 3613.6396856
time: 0.76 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5955445, upper bound: 3613.2716304
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2159.2155762, 1880.2059326, -2263.0097656, 1954.7694092, -4113.9848633, 4143.2148438
1: -1733.4216309, 1844.2053223, -1823.3979492, 1916.5390625, -3649.9606934, 3667.6030273
2: -2541.3471680, 1996.6114502, -2698.6789551, 2076.2812500, -4617.6284180, 4695.2905273
3: -983.7364502, 2579.6550293, -1021.0083618, 2711.4616699, -3695.1979980, 3600.6628418
4: -2797.9587402, 1944.6219482, -2969.3437500, 2023.2138672, -4821.1723633, 4913.9658203

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.0303594
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5730220, upper bound: 3614.0303594
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -2126.2890625, 1853.3342285, -3951.6337891, 3569.5126953, -5511.2768555, 5792.2495117
1: -1707.0120850, 1818.1448975, -3185.5791016, 3512.5834961, -5069.9711914, 4981.7568359
2: -2502.9104004, 1968.1744385, -4801.5273438, 3776.2404785, -6109.3496094, 6686.9628906
3: -969.2182007, 2541.6306152, -1773.3939209, 4846.9873047, -5738.2363281, 4271.8652344
4: -2755.4567871, 1916.7713623, -5253.3466797, 3668.9943848, -6248.8461914, 7095.3720703

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5724458, upper bound: 3613.5656835
time: 0.99 seconds

## Relational analysis of IS_A1_B2_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5729905, upper bound: 3613.6047026
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2200.5234375, 1911.4327393, -2391.8234863, 2050.3798828, -4250.9033203, 4303.2563477
1: -1765.5008545, 1873.8521729, -1926.9478760, 2008.6367188, -3774.1376953, 3800.8000488
2: -2586.5478516, 2028.2543945, -2846.4262695, 2178.7995605, -4765.3471680, 4874.6806641
3: -999.8256836, 2624.3425293, -1076.8645020, 2849.2856445, -3849.1110840, 3701.2070312
4: -2849.4145508, 1975.7768555, -3131.3625488, 2121.8227539, -4971.2368164, 5107.1391602

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7863259, upper bound: 3614.1330190
time: 0.78 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8426051, upper bound: 3614.1330190
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -2112.9672852, 1833.0050049, -2391.7839355, 2050.3518066, -4163.3193359, 4224.7890625
1: -1696.4462891, 1797.5927734, -1926.9154053, 2008.6096191, -3705.0554199, 3724.5083008
2: -2487.0517578, 1946.5822754, -2846.3762207, 2178.7705078, -4665.8217773, 4792.9584961
3: -961.7343140, 2521.0749512, -1076.8499756, 2849.2399902, -3810.9743652, 3597.9245605
4: -2738.1621094, 1895.1810303, -3131.3083496, 2121.7934570, -4859.9555664, 5026.4892578

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.7863259, upper bound: 3614.1342931
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8426051, upper bound: 3614.1341647
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1872.9016113, 1613.1390381, -1945.4641113, 1692.4307861, -3565.3322754, 3558.6025391
1: -1509.7269287, 1580.2980957, -1561.5529785, 1659.3345947, -3169.0615234, 3141.8508301
2: -2238.1496582, 1714.9681396, -2290.3945312, 1797.9466553, -4036.0961914, 4005.3627930
3: -843.5189209, 2237.2312012, -886.5736694, 2323.7197266, -3167.2385254, 3123.8049316
4: -2460.2197266, 1670.3427734, -2522.0895996, 1751.0017090, -4211.2211914, 4192.4316406

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3550183, upper bound: 3611.8444678
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2943193, upper bound: 3612.2683613
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2943193, upper bound: 3612.2683613
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1911.2631836, 1647.5620117, -2069.0957031, 1803.6733398, -3714.9360352, 3716.6577148
1: -1540.7338867, 1614.5070801, -1661.0068359, 1769.8718262, -3310.6057129, 3275.5136719
2: -2284.4213867, 1751.6201172, -2435.4772949, 1916.4532471, -4200.8745117, 4187.0976562
3: -861.4611206, 2284.5644531, -943.4463501, 2473.7744141, -3335.2355957, 3228.0102539
4: -2510.8459473, 1706.2406006, -2681.1193848, 1865.8546143, -4376.6992188, 4387.3593750

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3557211, upper bound: 3611.8421590
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=4653.345703125
rel_dist={0: [-3614.75946433006, 3614.75946433006]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1119.77 seconds
