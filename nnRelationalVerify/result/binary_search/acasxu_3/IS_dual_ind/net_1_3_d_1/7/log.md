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
execution time: IAR + LP analysis = 1.72 + 2.05 = 3.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7618777, upper bound: 3614.7618777


# Binary Search by BASE starts (time budget: 1196.23 seconds, max iter: 100)

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
Binary search time: 77.68 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1118.55 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2025555, upper bound: 3613.7565309
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565204, upper bound: 3613.7565215
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3614.2025555, upper bound: 3613.7565309
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3613.7565204, upper bound: 3613.7565215

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2502.3757324, 2150.9697266, -4423.8598633, 4467.8574219
1: -1831.3944092, 1927.9595947, -2015.7012939, 2107.5605469, -3938.9548340, 3943.6608887
2: -2712.4650879, 2088.7180176, -2978.1022949, 2285.2739258, -4997.7377930, 5066.8188477
3: -1026.7778320, 2724.7028809, -1126.3112793, 2983.9003906, -4010.6782227, 3851.0141602
4: -2982.6787109, 2035.6074219, -3276.4265137, 2226.9489746, -5209.6279297, 5312.0336914

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7201630
time: 0.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7565215
time: 0.88 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4124.1777344, 3713.5939941, -2497.2712402, 2147.3784180, -6271.5561523, 6039.2402344
1: -3324.5681152, 3656.1025391, -2011.7054443, 2104.0603027, -5428.6274414, 5525.1630859
2: -5014.5859375, 3930.4497070, -2972.4140625, 2281.4770508, -7266.3945312, 6744.4169922
3: -1845.6674805, 5055.1464844, -1124.3886719, 2978.6611328, -4782.7880859, 6124.5219727
4: -5486.4814453, 3818.3872070, -3270.0480957, 2223.2463379, -7691.5468750, 6921.5634766

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7201630
time: 0.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7565215
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7201630
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7565215
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7201630
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7565215

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2272.8901367, 1965.4815674, -4238.3715820, 4238.3715820
1: -1831.3944092, 1927.9595947, -1831.3944092, 1927.9595947, -3759.3540039, 3759.3540039
2: -2712.4650879, 2088.7180176, -2712.4650879, 2088.7180176, -4801.1821289, 4801.1816406
3: -1026.7778320, 2724.7028809, -1026.7778320, 2724.7028809, -3751.4807129, 3751.4807129
4: -2982.6787109, 2035.6074219, -2982.6787109, 2035.6074219, -5018.2861328, 5018.2861328

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311280, upper bound: 3612.9658089
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657996, upper bound: 3612.9658089
time: 0.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -4122.6225586, 3711.9274902, -5815.6030273, 6088.1040039
1: -1831.3944092, 1927.9595947, -3323.3232422, 3654.4309082, -5345.2187500, 5251.2827148
2: -2712.4650879, 2088.7180176, -5012.6992188, 3928.6540527, -6487.0761719, 7077.5927734
3: -1026.7778320, 2724.7028809, -1844.8614502, 5053.0864258, -6027.1240234, 4530.7172852
4: -2982.6787109, 2035.6074219, -5484.4082031, 3816.6474609, -6636.9287109, 7507.5883789

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311280, upper bound: 3613.4908192
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657996, upper bound: 3613.4908192
time: 0.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4122.6225586, 3711.9274902, -2272.8901367, 1965.4815674, -6088.1040039, 5815.6035156
1: -3323.3232422, 3654.4309082, -1831.3944092, 1927.9595947, -5251.2827148, 5345.2182617
2: -5012.6992188, 3928.6540527, -2712.4650879, 2088.7180176, -7077.5927734, 6487.0756836
3: -1844.8614502, 5053.0864258, -1026.7778320, 2724.7028809, -4530.7172852, 6027.1235352
4: -5484.4082031, 3816.6474609, -2982.6787109, 2035.6074219, -7507.5883789, 6636.9291992

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295725, upper bound: 3612.9657827
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908072, upper bound: 3612.9657996
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4128.4018555, 3718.1166992, -4128.4018555, 3718.1166992, -7603.6601562, 7603.6606445
1: -3327.9492188, 3660.6428223, -3327.9492188, 3660.6428223, -6780.5097656, 6780.5092773
2: -5019.7089844, 3935.3261719, -5019.7089844, 3935.3261719, -8661.0214844, 8661.0224609
3: -1847.8551025, 5060.7402344, -1847.8551025, 5060.7402344, -6755.2485352, 6755.2485352
4: -5492.1123047, 3823.1091309, -5492.1123047, 3823.1091309, -9018.0908203, 9018.0898438

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295730, upper bound: 3613.1744115
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908076, upper bound: 3613.4896465
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.5311280, upper bound: 3612.9658089
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -3612.9657996, upper bound: 3612.9658089
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.5311280, upper bound: 3613.4908192
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -3612.9657996, upper bound: 3613.4908192
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.0295725, upper bound: 3612.9657827
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.4908072, upper bound: 3612.9657996
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.0295730, upper bound: 3613.1744115
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 0, lower bound: -3613.4908076, upper bound: 3613.4896465

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -2272.8901367, 1965.4815674, -3999.3693848, 4041.9062500
1: -1638.3045654, 1737.5650635, -1831.3944092, 1927.9595947, -3566.2641602, 3568.9594727
2: -2435.9470215, 1879.3935547, -2712.4650879, 2088.7180176, -4524.6650391, 4591.8583984
3: -920.3916626, 2455.5791016, -1026.7778320, 2724.7028809, -3645.0944824, 3482.3569336
4: -2677.5119629, 1832.2690430, -2982.6787109, 2035.6074219, -4713.1181641, 4814.9477539

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -4121.3862305, 3710.6052246, -5578.7456055, 5890.4033203
1: -1638.3045654, 1737.5650635, -3322.3349609, 3653.1037598, -5153.6406250, 5059.8999023
2: -2435.9470215, 1879.3935547, -5011.2006836, 3927.2285156, -6214.8603516, 6873.1196289
3: -920.3916626, 2455.5791016, -1844.2224121, 5051.4501953, -5921.9218750, 4264.3569336
4: -2677.5119629, 1832.2690430, -5482.7602539, 3815.2673340, -6336.0502930, 7309.1645508

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295819
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -4125.2910156, 3715.4091797, -7454.1420898, 7464.4877930
1: -3211.7971191, 3524.2065430, -3325.4504395, 3658.0114746, -6660.5903320, 6646.9760742
2: -4848.2338867, 3788.9609375, -5016.0517578, 3932.4782715, -8484.7998047, 8514.4414062
3: -1780.8483887, 4870.8652344, -1846.4433594, 5057.0297852, -6680.4995117, 6577.0322266
4: -5298.1914062, 3677.4733887, -5488.0747070, 3820.3156738, -8818.0302734, 8875.5986328

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295814
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908130
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6103.8901367, 5568.2299805, -2264.6857910, 1958.7513428, -7950.0581055, 7293.1826172
1: -4911.7827148, 5501.2548828, -1824.8594971, 1921.3773193, -6730.6142578, 6869.3081055
2: -7462.5488281, 5912.2846680, -2702.9743652, 2081.6154785, -9288.9414062, 8073.6503906
3: -2735.5029297, 7516.6303711, -1023.2143555, 2715.3166504, -5256.7763672, 8324.7998047
4: -8146.9394531, 5726.9809570, -2972.1008301, 2028.6511230, -9924.9052734, 8149.2919922

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -4125.2910156, 3715.4091797, -9552.0107422, 9178.3339844
1: -5010.5698242, 5614.1787109, -3325.4504395, 3658.0114746, -8339.1435547, 8401.5888672
2: -7611.5888672, 6033.4814453, -5016.0517578, 3932.4782715, -11008.7968750, 10352.1542969
3: -2789.6572266, 7670.8759766, -1846.4433594, 5057.0297852, -7533.2338867, 9192.4150391
4: -8309.2480469, 5844.3500977, -5488.0747070, 3820.3156738, -11584.2900391, 10632.1796875

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295301
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.4896402
time: 0.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295819
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295814
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908130
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295301
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.4896402

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -6079.8549805, 5548.7236328, -7048.5986328, 7746.1054688
1: -1638.3045654, 1737.5650635, -4892.2514648, 5482.0126953, -6668.2084961, 6533.0117188
2: -2435.9470215, 1879.3935547, -7433.3466797, 5891.0068359, -7793.2504883, 9065.9023438
3: -920.3916626, 2455.5791016, -2725.6188965, 7488.2944336, -8197.8466797, 4991.3706055
4: -2677.5119629, 1832.2690430, -8114.9824219, 5706.0473633, -7841.6572266, 9705.2041016

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057546, upper bound: 3613.4907878
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311070, upper bound: 3613.4907898
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -6221.7968750, 5678.8237305, -9031.7128906, 9415.4287109
1: -3211.7971191, 3524.2065430, -5007.7504883, 5610.3808594, -8283.7460938, 8208.1796875
2: -4848.2338867, 3788.9609375, -7607.3647461, 6029.3945312, -10179.2890625, 10864.9121094
3: -1780.8483887, 4870.8652344, -2787.8225098, 7666.1835938, -9118.5800781, 7358.7138672
4: -5298.1914062, 3677.4733887, -8304.6015625, 5840.4184570, -10435.9970703, 11444.4169922

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657724, upper bound: 3613.1309497
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9265740, upper bound: 3613.0462884
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6079.8549805, 5548.7241211, -2033.8878174, 1769.0169678, -7746.1054688, 7048.5981445
1: -4892.2514648, 5482.0126953, -1638.3045654, 1737.5650635, -6533.0117188, 6668.2084961
2: -7433.3457031, 5891.0068359, -2435.9470215, 1879.3935547, -9065.9023438, 7793.2504883
3: -2725.6188965, 7488.2944336, -920.3916626, 2455.5791016, -4991.3710938, 8197.8466797
4: -8114.9824219, 5706.0478516, -2677.5119629, 1832.2690430, -9705.2041016, 7841.6572266

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907981, upper bound: 3612.9266012
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941720, upper bound: 3612.9265836
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6221.7968750, 5678.8237305, -3987.9494629, 3574.2734375, -9415.4287109, 9031.7138672
1: -5007.7504883, 5610.3808594, -3211.7971191, 3524.2065430, -8208.1806641, 8283.7451172
2: -7607.3647461, 6029.3945312, -4848.2338867, 3788.9609375, -10864.9130859, 10179.2880859
3: -2787.8225098, 7666.1835938, -1780.8483887, 4870.8652344, -7358.7133789, 9118.5810547
4: -8304.6015625, 5840.4184570, -5298.1914062, 3677.4733887, -11444.4169922, 10435.9960938

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907984, upper bound: 3612.9266012
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941722, upper bound: 3612.9265836
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -3904.8264160, 3541.1750488, -9358.8994141, 8945.6386719
1: -5010.5698242, 5614.1787109, -3146.5612793, 3492.1206055, -8155.9272461, 8213.0507812
2: -7611.5888672, 6033.4814453, -4764.1665039, 3751.3303223, -10806.4443359, 10082.0341797
3: -2789.6572266, 7670.8759766, -1749.9495850, 4810.4135742, -7274.6840820, 9085.5927734
4: -8309.2480469, 5844.3500977, -5208.8427734, 3642.6633301, -11386.6064453, 10332.4433594

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907764, upper bound: 3612.9552912
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941522, upper bound: 3612.9552731
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -6225.2827148, 5682.6333008, -11132.6572266, 11132.6582031
1: -5010.5698242, 5614.1787109, -5010.5698242, 5614.1787109, -9965.4804688, 9965.4804688
2: -7611.5888672, 6033.4814453, -7611.5888672, 6033.4814453, -12706.6269531, 12706.6269531
3: -2789.6572266, 7670.8759766, -2789.6572266, 7670.8759766, -9975.6748047, 9975.6748047
4: -8309.2480469, 5844.3500977, -8309.2480469, 5844.3500977, -13205.4160156, 13205.4160156

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907786, upper bound: 3613.2934108
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941524, upper bound: 3613.2932943
time: 0.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.40 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.4057546, upper bound: 3613.4907878
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.5311070, upper bound: 3613.4907898
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3612.9657724, upper bound: 3613.1309497
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3612.9265740, upper bound: 3613.0462884
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.4907981, upper bound: 3612.9266012
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.2941720, upper bound: 3612.9265836
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.4907984, upper bound: 3612.9266012
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.2941722, upper bound: 3612.9265836
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.4907764, upper bound: 3612.9552912
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.2941522, upper bound: 3612.9552731
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.4907786, upper bound: 3613.2934108
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.40
Output dim: 0, lower bound: -3613.2941524, upper bound: 3613.2932943

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -6053.9716797, 5529.0927734, -6787.2143555, 7507.4677734
1: -1442.5327148, 1525.0668945, -4871.2143555, 5462.3798828, -6455.1254883, 6300.4257812
2: -2150.3056641, 1650.0454102, -7401.8427734, 5869.1176758, -7488.8027344, 8806.9550781
3: -806.7396851, 2160.4394531, -2715.3159180, 7458.5175781, -8056.2958984, 4687.6923828
4: -2361.8547363, 1606.7463379, -8080.2578125, 5684.1567383, -7507.2768555, 9446.6015625

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057573, upper bound: 3613.4907675
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057485, upper bound: 3613.2941418
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6075.5498047, 5545.0888672, -7001.6513672, 7703.1542969
1: -1604.6556396, 1701.4029541, -4888.7314453, 5478.4458008, -6629.7705078, 6491.1801758
2: -2386.9431152, 1840.0159912, -7428.1166992, 5887.0859375, -7738.4941406, 9019.8242188
3: -900.5989380, 2405.9113770, -2723.7751465, 7483.0825195, -8173.6591797, 4937.1601562
4: -2623.7451172, 1794.3088379, -8109.2993164, 5702.1958008, -7782.1455078, 9659.4130859

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311097, upper bound: 3613.4907773
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311009, upper bound: 3613.2941516
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6082.2338867, 5546.9345703, -2033.8878174, 1769.0169678, -7749.7133789, 7044.2089844
1: -4893.3759766, 5479.8559570, -1638.3045654, 1737.5650635, -6535.0043945, 6663.6005859
2: -7433.8115234, 5887.6699219, -2435.9470215, 1879.3935547, -9067.6689453, 7786.7919922
3: -2725.1577148, 7488.3901367, -920.3916626, 2455.5791016, -4989.3027344, 8199.2763672
4: -8117.4887695, 5704.0190430, -2677.5119629, 1832.2690430, -9708.4697266, 7835.9477539

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907675, upper bound: 3613.4057573
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907773, upper bound: 3613.5311097
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6233.8764648, 5686.0698242, -3987.9494629, 3574.2734375, -9428.2490234, 9035.0839844
1: -5016.8271484, 5617.0932617, -3211.7971191, 3524.2065430, -8217.6513672, 8286.9580078
2: -7619.8056641, 6035.7329102, -4848.2338867, 3788.9609375, -10877.7158203, 10181.3300781
3: -2791.9553223, 7678.4619141, -1780.8483887, 4870.8652344, -7360.8989258, 9131.5195312
4: -8320.0849609, 5847.6738281, -5298.1914062, 3677.4733887, -11459.7939453, 10438.3378906

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941705, upper bound: 3612.9265836
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941705, upper bound: 3612.9265836
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -3904.8264160, 3541.1750488, -9371.9589844, 8949.2119141
1: -5019.8442383, 5621.1367188, -3146.5612793, 3492.1206055, -8165.5893555, 8216.4707031
2: -7624.3247070, 6040.0839844, -4764.1665039, 3751.3303223, -10819.5341797, 10084.2929688
3: -2793.9113770, 7683.4697266, -1749.9495850, 4810.4135742, -7276.9716797, 9098.8281250
4: -8325.0566406, 5851.8598633, -5208.8427734, 3642.6633301, -11402.2871094, 10334.9931641

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4898785, upper bound: 3612.5086647
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907756, upper bound: 3612.9552727
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -6225.2827148, 5682.6333008, -11145.7167969, 11136.2314453
1: -5019.8442383, 5621.1367188, -5010.5698242, 5614.1787109, -9975.1435547, 9968.9003906
2: -7624.3247070, 6040.0839844, -7611.5888672, 6033.4814453, -12719.7158203, 12708.8876953
3: -2793.9113770, 7683.4697266, -2789.6572266, 7670.8759766, -9977.9628906, 9988.9111328
4: -8325.0566406, 5851.8598633, -8309.2480469, 5844.3500977, -13221.0966797, 13207.9658203

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941603, upper bound: 3613.2932926
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941603, upper bound: 3613.2932926
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.47 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4057573, upper bound: 3613.4907675
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4057485, upper bound: 3613.2941418
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.5311097, upper bound: 3613.4907773
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.5311009, upper bound: 3613.2941516
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4907675, upper bound: 3613.4057573
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4907773, upper bound: 3613.5311097
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.2941705, upper bound: 3612.9265836
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.2941705, upper bound: 3612.9265836
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4898785, upper bound: 3612.5086647
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.4907756, upper bound: 3612.9552727
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.2941603, upper bound: 3613.2932926
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 0, lower bound: -3613.2941603, upper bound: 3613.2932926

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -6055.7119141, 5526.8359375, -6782.4311523, 7510.5244141
1: -1442.5327148, 1525.0668945, -4871.8417969, 5459.7622070, -6450.1083984, 6301.9873047
2: -2150.3056641, 1650.0454102, -7401.5039062, 5865.2817383, -7481.9072266, 8808.0410156
3: -806.7396851, 2160.4394531, -2714.5974121, 7457.8686523, -8057.0249023, 4685.4018555
4: -2361.8547363, 1606.7463379, -8081.8896484, 5681.6533203, -7501.1528320, 9449.1230469

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1663892, upper bound: 3613.4849670
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2977484, upper bound: 3613.4906850
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -5980.8188477, 5460.6489258, -6719.5224609, 7430.8764648
1: -1442.5327148, 1525.0668945, -4812.5136719, 5394.9311523, -6388.3354492, 6238.5131836
2: -2150.3056641, 1650.0454102, -7316.6718750, 5796.4248047, -7416.9482422, 8715.4160156
3: -806.7396851, 2160.4394531, -2680.6933594, 7369.2075195, -7964.6840820, 4652.8530273
4: -2361.8547363, 1606.7463379, -7985.9404297, 5612.3662109, -7436.9106445, 9345.0634766

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1663884, upper bound: 3613.2883412
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2977475, upper bound: 3613.2940593
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6077.8432617, 5543.2167969, -6997.1918945, 7706.6826172
1: -1604.6556396, 1701.4029541, -4889.7871094, 5476.2080078, -6625.0917969, 6493.1093750
2: -2386.9431152, 1840.0159912, -7428.4785156, 5883.6601562, -7731.9609375, 9021.4990234
3: -900.5989380, 2405.9113770, -2723.2734375, 7483.0717773, -8174.9868164, 4935.0571289
4: -2623.7451172, 1794.3088379, -8111.6938477, 5700.0820312, -7776.3637695, 9662.5751953

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505376, upper bound: 3613.4898848
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669531, upper bound: 3613.4906933
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6002.6767578, 5477.0166016, -6934.2509766, 7626.8173828
1: -1604.6556396, 1701.4029541, -4830.2675781, 5411.3964844, -6563.2973633, 6429.4912109
2: -2386.9431152, 1840.0159912, -7343.3027344, 5814.7944336, -7666.9428711, 8928.6132812
3: -900.5989380, 2405.9113770, -2689.2775879, 7394.2895508, -8082.5312500, 4902.4306641
4: -2623.7451172, 1794.3088379, -8015.3442383, 5630.8066406, -7712.0786133, 9558.2041016

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1663884, upper bound: 3613.2932591
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669523, upper bound: 3613.2940676
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6055.7124023, 5526.8359375, -1789.6997070, 1554.3094482, -7510.5249023, 6782.4311523
1: -4871.8422852, 5459.7622070, -1442.5327148, 1525.0668945, -6301.9873047, 6450.1088867
2: -7401.5039062, 5865.2822266, -2150.3056641, 1650.0454102, -8808.0410156, 7481.9072266
3: -2714.5974121, 7457.8686523, -806.7396851, 2160.4394531, -4685.4018555, 8057.0244141
4: -8081.8896484, 5681.6528320, -2361.8547363, 1606.7463379, -9449.1230469, 7501.1533203

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8999913, upper bound: 3613.2970405
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906850, upper bound: 3613.2977487
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6077.8432617, 5543.2167969, -1991.7487793, 1732.1446533, -7706.6826172, 6997.1918945
1: -4889.7871094, 5476.2080078, -1604.6556396, 1701.4029541, -6493.1093750, 6625.0917969
2: -7428.4785156, 5883.6601562, -2386.9431152, 1840.0159912, -9021.4990234, 7731.9609375
3: -2723.2734375, 7483.0717773, -900.5989380, 2405.9113770, -4935.0571289, 8174.9863281
4: -8111.6938477, 5700.0820312, -2623.7451172, 1794.3088379, -9662.5751953, 7776.3637695

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8999996, upper bound: 3613.4662450
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906933, upper bound: 3613.4669532
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -3664.4638672, 3330.0925293, -9148.6132812, 8701.7822266
1: -5019.8442383, 5621.1367188, -2953.4743652, 3285.4580078, -7947.7929688, 8017.8134766
2: -7624.3247070, 6040.0839844, -4484.3339844, 3529.7871094, -10584.4599609, 9794.0937500
3: -2793.9113770, 7683.4697266, -1640.4355469, 4520.1694336, -6979.9189453, 8983.2011719
4: -8325.0566406, 5851.8598633, -4898.6835938, 3424.3659668, -11172.1406250, 10013.4726562

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8991692, upper bound: 3612.4577413
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4898636, upper bound: 3612.4584494
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -3860.2880859, 3503.7695312, -9325.8027344, 8901.6035156
1: -5019.8442383, 5621.1367188, -3110.7521973, 3455.8679199, -8121.6240234, 8177.6230469
2: -7624.3247070, 6040.0839844, -4712.5708008, 3712.0446777, -10771.7568359, 10027.4296875
3: -2793.9113770, 7683.4697266, -1729.5795898, 4758.1948242, -7218.9931641, 9076.6445312
4: -8325.0566406, 5851.8598633, -5151.9331055, 3604.4179688, -11354.9306641, 10272.5390625

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8999955, upper bound: 3612.9325003
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915
time: 0.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.51 seconds
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.1663892, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.2977484, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.1663884, upper bound: 3613.2883412
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.2977475, upper bound: 3613.2940593
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4505376, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4669531, upper bound: 3613.4906933
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.1663884, upper bound: 3613.2932591
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4669523, upper bound: 3613.2940676
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3611.8999913, upper bound: 3613.2970405
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4906850, upper bound: 3613.2977487
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3611.8999996, upper bound: 3613.4662450
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4906933, upper bound: 3613.4669532
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3611.8991692, upper bound: 3612.4577413
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4898636, upper bound: 3612.4584494
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.51
Output dim: 0, lower bound: -3611.8999955, upper bound: 3612.9325003
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.51
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1722.8291016, 1497.3557129, -6045.0351562, 5517.6455078, -6708.2001953, 7444.1865234
1: -1388.8266602, 1467.8127441, -4863.1870117, 5450.5712891, -6388.8193359, 6239.3808594
2: -2072.7800293, 1587.4006348, -7388.5122070, 5855.2622070, -7396.7758789, 8736.9052734
3: -775.4130249, 2082.1958008, -2709.9519043, 7444.9770508, -8013.9418945, 4603.7592773
4: -2276.3806152, 1545.3090820, -8067.6816406, 5671.7792969, -7408.2592773, 9376.0068359

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1656810, upper bound: 3611.8942726
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1744.4897461, 1521.2305908, -6056.1547852, 5528.1718750, -6739.1264648, 7472.4404297
1: -1407.2333984, 1489.6105957, -4872.2744141, 5460.8520508, -6416.0034180, 6265.7075195
2: -2100.1879883, 1605.6900635, -7402.2441406, 5866.3901367, -7433.1191406, 8768.1796875
3: -783.8977661, 2111.2338867, -2715.0876465, 7459.3310547, -8035.9233398, 4635.4267578
4: -2305.2700195, 1564.3320312, -8082.5507812, 5682.6206055, -7446.5898438, 9407.8544922

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2970405, upper bound: 3611.8999913
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1920.2984619, 1669.9824219, -6068.9189453, 5535.7060547, -6919.6850586, 7638.5307617
1: -1547.3994141, 1641.2923584, -4882.5102539, 5468.8354492, -6561.6391602, 6426.9384766
2: -2304.2888184, 1774.5805664, -7417.5351562, 5875.5371094, -7642.9155273, 8946.8642578
3: -867.9794312, 2322.4943848, -2719.4304199, 7472.2568359, -8132.4580078, 4848.7578125
4: -2532.7153320, 1730.3000488, -8099.7402344, 5692.0458984, -7679.1015625, 9588.4140625

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991910
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -6079.3022461, 5545.6503906, -6948.7265625, 7664.1181641
1: -1564.4899902, 1655.1600342, -4891.0244141, 5478.5981445, -6587.1542969, 6450.3144531
2: -2329.3627930, 1788.5822754, -7430.3344727, 5886.0927734, -7676.7338867, 8975.2353516
3: -875.6460571, 2347.2778320, -2724.2631836, 7485.6674805, -8152.9028320, 4876.6523438
4: -2559.2373047, 1741.9539795, -8113.5795898, 5702.3359375, -7714.6186523, 9617.2041016

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662450, upper bound: 3611.8999990
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -6004.0410156, 5479.3520508, -6885.6918945, 7584.1586914
1: -1564.4899902, 1655.1600342, -4831.4228516, 5413.6953125, -6525.2705078, 6386.6132812
2: -2329.3627930, 1788.5822754, -7345.0390625, 5817.1264648, -7611.6191406, 8882.2314453
3: -875.6460571, 2347.2778320, -2690.2272949, 7396.7397461, -8060.3061523, 4843.9833984
4: -2559.2373047, 1741.9539795, -8017.1088867, 5632.9638672, -7650.2416992, 9512.7070312

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660952, upper bound: 3611.7109288
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2940676
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6014.6508789, 5501.8559570, -1789.6997070, 1554.3094482, -7465.8515625, 6743.0341797
1: -4838.9477539, 5434.8125000, -1442.5327148, 1525.0668945, -6266.0747070, 6412.8051758
2: -7356.0581055, 5838.8579102, -2150.3056641, 1650.0454102, -8755.9238281, 7440.0742188
3: -2698.1750488, 7414.9916992, -806.7396851, 2160.4394531, -4663.3906250, 8007.9687500
4: -8030.1000977, 5654.5083008, -2361.8547363, 1606.7463379, -9390.9697266, 7460.2255859

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4849670, upper bound: 3613.1663892
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4849670, upper bound: 3613.2977484
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6013.4746094, 5493.0332031, -1991.7487793, 1732.1446533, -7637.0292969, 6936.6875000
1: -4837.7236328, 5427.5483398, -1604.6556396, 1701.4029541, -6436.8471680, 6567.3286133
2: -7354.9052734, 5830.8779297, -2386.9431152, 1840.0159912, -8939.7617188, 7668.4633789
3: -2695.6022949, 7410.1484375, -900.5989380, 2405.9113770, -4903.2548828, 8096.0576172
4: -8029.8110352, 5647.5683594, -2623.7451172, 1794.3088379, -9571.9482422, 7713.6445312

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8991910, upper bound: 3613.4498293
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8991910, upper bound: 3613.4662450
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6037.3144531, 5518.8598633, -1991.7487793, 1732.1446533, -7662.5429688, 6958.3442383
1: -4857.3134766, 5451.9462891, -1604.6556396, 1701.4029541, -6457.6279297, 6588.3818359
2: -7383.6406250, 5857.9501953, -2386.9431152, 1840.0159912, -8969.9941406, 7690.7216797
3: -2707.1384277, 7440.9462891, -900.5989380, 2405.9113770, -4913.3320312, 8126.6850586
4: -8060.5771484, 5673.6303711, -2623.7451172, 1794.3088379, -9605.1015625, 7736.0019531

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4898848, upper bound: 3613.4505376
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4898848, upper bound: 3613.4669532
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3664.4638672, 3330.0925293, -9107.5566406, 8664.8554688
1: -4990.1635742, 5599.2011719, -2953.4743652, 3285.4580078, -7914.9160156, 7982.8183594
2: -7583.7001953, 6017.1689453, -4484.3339844, 3529.7871094, -10536.6972656, 9754.9160156
3: -2778.9992676, 7645.6254883, -1640.4355469, 4520.1694336, -6959.2163086, 8938.7939453
4: -8278.4130859, 5828.1499023, -4898.6835938, 3424.3659668, -11118.6250000, 9975.2900391

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4747414, upper bound: 3612.1118777
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1638291, upper bound: 3612.3128517
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4897836, upper bound: 3612.4583663
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 15

Time for candidate selection: 11.33 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4840426, upper bound: 3612.4016717
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4840460, upper bound: 3612.4584501
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3860.2880859, 3503.7695312, -9284.7451172, 8864.6777344
1: -4990.1635742, 5599.2011719, -3110.7521973, 3455.8679199, -8088.7475586, 8142.6279297
2: -7583.7001953, 6017.1689453, -4712.5708008, 3712.0446777, -10723.9921875, 9988.2519531
3: -2778.9992676, 7645.6254883, -1729.5795898, 4758.1948242, -7198.2910156, 9032.2382812
4: -8278.4130859, 5828.1499023, -5151.9331055, 3604.4179688, -11301.4140625, 10234.3574219

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915
time: 0.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.57 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.1656810, upper bound: 3611.8942726
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.2970405, upper bound: 3611.8999913
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991910
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4662450, upper bound: 3611.8999990
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4660952, upper bound: 3611.7109288
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2940676
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4849670, upper bound: 3613.1663892
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4849670, upper bound: 3613.2977484
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3611.8991910, upper bound: 3613.4498293
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3611.8991910, upper bound: 3613.4662450
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4898848, upper bound: 3613.4505376
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4898848, upper bound: 3613.4669532
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4840426, upper bound: 3612.4016717
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4840460, upper bound: 3612.4584501
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.57
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331915

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1722.8291016, 1497.3557129, -6003.8715820, 5492.4843750, -6668.6914062, 7399.4287109
1: -1388.8266602, 1467.8127441, -4830.2045898, 5425.4467773, -6351.3984375, 6203.3984375
2: -2072.7800293, 1587.4006348, -7342.9165039, 5828.6499023, -7354.8222656, 8684.6728516
3: -775.4130249, 2082.1958008, -2693.4523926, 7401.9233398, -7964.7407227, 4581.6889648
4: -2276.3806152, 1545.3090820, -8015.7231445, 5644.4448242, -7367.2094727, 9317.7304688

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1744.4897461, 1521.2305908, -6014.7856445, 5502.9272461, -6699.4794922, 7427.4643555
1: -1407.2333984, 1489.6105957, -4839.1225586, 5435.6381836, -6378.4462891, 6229.5444336
2: -2100.1879883, 1605.6900635, -7356.4199219, 5839.6791992, -7391.0146484, 8715.6953125
3: -783.8977661, 2111.2338867, -2698.5139160, 7416.0742188, -7986.4936523, 4613.2709961
4: -2305.2700195, 1564.3320312, -8030.3564453, 5655.1875000, -7405.3964844, 9349.3076172

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1920.2984619, 1669.9824219, -6004.2807617, 5485.4443359, -6859.0659180, 7568.6333008
1: -1547.3994141, 1641.2923584, -4830.2622070, 5420.0947266, -6503.7641602, 6370.4970703
2: -2304.2888184, 1774.5805664, -7343.6542969, 5822.6630859, -7579.2890625, 8864.8427734
3: -867.9794312, 2322.4943848, -2691.6535645, 7399.0703125, -8053.2519531, 4816.8422852
4: -2532.7153320, 1730.3000488, -8017.5244141, 5639.4550781, -7616.2675781, 9497.4746094

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991905
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991905
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1920.2984619, 1669.9824219, -6028.1640625, 5511.1376953, -6880.6772461, 7594.1875000
1: -1547.3994141, 1641.2923584, -4849.8530273, 5444.3637695, -6524.7636719, 6391.2895508
2: -2304.2888184, 1774.5805664, -7372.4033203, 5849.5966797, -7601.4995117, 8895.1054688
3: -867.9794312, 2322.4943848, -2703.1772461, 7429.7875977, -8083.8520508, 4826.9272461
4: -2532.7153320, 1730.3000488, -8048.2963867, 5665.3769531, -7638.5747070, 9530.6533203

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -6014.0917969, 5494.8344727, -6887.6333008, 7593.6669922
1: -1564.4899902, 1655.1600342, -4838.3066406, 5429.3125000, -6528.8022461, 6393.4223633
2: -2329.3627930, 1788.5822754, -7355.7504883, 5832.6489258, -7612.6230469, 8892.5488281
3: -875.6460571, 2347.2778320, -2696.2585449, 7411.7636719, -8073.0146484, 4844.5527344
4: -2559.2373047, 1741.9539795, -8030.6083984, 5649.1870117, -7651.3149414, 9525.5410156

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662449, upper bound: 3611.8999990
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662449, upper bound: 3611.8999996
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -6038.4687500, 5521.0268555, -6909.6313477, 7619.6782227
1: -1564.4899902, 1655.1600342, -4858.2929688, 5454.0683594, -6550.1923828, 6414.5810547
2: -2329.3627930, 1788.5822754, -7385.1196289, 5860.0937500, -7635.2285156, 8923.3671875
3: -875.6460571, 2347.2778320, -2707.9909668, 7443.1547852, -8104.2221680, 4854.7968750
4: -2559.2373047, 1741.9539795, -8062.0629883, 5675.5917969, -7673.9956055, 9559.3398438

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -5938.0761719, 5427.8056641, -6823.4150391, 7512.6660156
1: -1564.4899902, 1655.1600342, -4778.1049805, 5363.6552734, -6465.7553711, 6328.8784180
2: -2329.3627930, 1788.5822754, -7269.7797852, 5762.8315430, -7546.1625977, 8798.3750000
3: -875.6460571, 2347.2778320, -2661.6201172, 7322.5683594, -7979.6782227, 4811.1796875
4: -2559.2373047, 1741.9539795, -7933.3129883, 5579.0449219, -7585.7285156, 9419.6845703

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4496796, upper bound: 3611.7109288
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660951, upper bound: 3611.7109288
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -5959.8579102, 5451.5322266, -6842.8779297, 7536.0546875
1: -1564.4899902, 1655.1600342, -4796.0776367, 5386.0708008, -6484.7539062, 6347.9824219
2: -2329.3627930, 1788.5822754, -7296.1811523, 5787.7260742, -7566.1372070, 8826.1279297
3: -875.6460571, 2347.2778320, -2672.3415527, 7350.8774414, -8007.7255859, 4820.0102539
4: -2559.2373047, 1741.9539795, -7961.4692383, 5602.7944336, -7605.7265625, 9450.0839844

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2939960
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2940032
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6003.8715820, 5492.4843750, -1722.8291016, 1497.3557129, -7399.4282227, 6668.6914062
1: -4830.2045898, 5425.4467773, -1388.8266602, 1467.8127441, -6203.3984375, 6351.3984375
2: -7342.9160156, 5828.6494141, -2072.7800293, 1587.4006348, -8684.6728516, 7354.8222656
3: -2693.4523926, 7401.9233398, -775.4130249, 2082.1958008, -4581.6889648, 7964.7402344
4: -8015.7241211, 5644.4448242, -2276.3806152, 1545.3090820, -9317.7304688, 7367.2089844

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.1663774
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.1663892
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6014.7851562, 5502.9272461, -1744.4897461, 1521.2305908, -7427.4643555, 6699.4794922
1: -4839.1225586, 5435.6376953, -1407.2333984, 1489.6105957, -6229.5439453, 6378.4458008
2: -7356.4199219, 5839.6791992, -2100.1879883, 1605.6900635, -8715.6953125, 7391.0151367
3: -2698.5139160, 7416.0742188, -783.8977661, 2111.2338867, -4613.2709961, 7986.4936523
4: -8030.3564453, 5655.1875000, -2305.2700195, 1564.3320312, -9349.3076172, 7405.3964844

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.2977369
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.2977487
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6004.2807617, 5485.4443359, -1920.2984619, 1669.9824219, -7568.6333008, 6859.0654297
1: -4830.2622070, 5420.0947266, -1547.3994141, 1641.2923584, -6370.4965820, 6503.7641602
2: -7343.6542969, 5822.6635742, -2304.2888184, 1774.5805664, -8864.8437500, 7579.2900391
3: -2691.6533203, 7399.0698242, -867.9794312, 2322.4943848, -4816.8422852, 8053.2519531
4: -8017.5244141, 5639.4550781, -2532.7153320, 1730.3000488, -9497.4755859, 7616.2680664

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2478966, upper bound: 3613.1739073
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8961813, upper bound: 3613.4141261
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8567883, upper bound: 3612.9315382
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6014.0917969, 5494.8344727, -1940.5773926, 1689.8077393, -7593.6665039, 6887.6337891
1: -4838.3066406, 5429.3125000, -1564.4899902, 1655.1600342, -6393.4228516, 6528.8022461
2: -7355.7504883, 5832.6489258, -2329.3627930, 1788.5822754, -8892.5498047, 7612.6230469
3: -2696.2585449, 7411.7636719, -875.6460571, 2347.2778320, -4844.5522461, 8073.0146484
4: -8030.6083984, 5649.1870117, -2559.2373047, 1741.9539795, -9525.5400391, 7651.3149414

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2478966, upper bound: 3613.1984042
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8961813, upper bound: 3613.4141261
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8567883, upper bound: 3612.9315382
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4660730
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4661695
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6028.1645508, 5511.1376953, -1920.2984619, 1669.9824219, -7594.1879883, 6880.6777344
1: -4849.8530273, 5444.3637695, -1547.3994141, 1641.2923584, -6391.2895508, 6524.7631836
2: -7372.4033203, 5849.5971680, -2304.2888184, 1774.5805664, -8895.1054688, 7601.5000000
3: -2703.1772461, 7429.7875977, -867.9794312, 2322.4943848, -4826.9277344, 8083.8525391
4: -8048.2968750, 5665.3769531, -2532.7153320, 1730.3000488, -9530.6542969, 7638.5747070

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505258
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6038.4687500, 5521.0273438, -1940.5773926, 1689.8077393, -7619.6772461, 6909.6313477
1: -4858.2929688, 5454.0683594, -1564.4899902, 1655.1600342, -6414.5805664, 6550.1918945
2: -7385.1206055, 5860.0937500, -2329.3627930, 1788.5822754, -8923.3671875, 7635.2285156
3: -2707.9909668, 7443.1547852, -875.6460571, 2347.2778320, -4854.7968750, 8104.2216797
4: -8062.0629883, 5675.5917969, -2559.2373047, 1741.9539795, -9559.3388672, 7673.9956055

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4669414
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4668300
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3420.8581543, 3095.4916992, -8875.0996094, 8421.6533203
1: -4990.1635742, 5599.2011719, -2759.9699707, 3050.4467773, -7682.3862305, 7791.6103516
2: -7583.7001953, 6017.1689453, -4186.4858398, 3280.9074707, -10290.3769531, 9461.3300781
3: -2778.9992676, 7645.6254883, -1524.3773193, 4204.2319336, -6647.5834961, 8825.8554688
4: -8278.4130859, 5828.1499023, -4567.4165039, 3181.7402344, -10879.0097656, 9647.8134766

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2824237, upper bound: 3612.4016601
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2824231, upper bound: 3612.4016720
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3621.6245117, 3293.9077148, -9067.9375000, 8620.2138672
1: -4990.1635742, 5599.2011719, -2918.8852539, 3249.9467773, -7876.4355469, 7946.0698242
2: -7583.7001953, 6017.1689453, -4433.8842773, 3491.0607910, -10494.1796875, 9700.6796875
3: -2778.9992676, 7645.6254883, -1620.9511719, 4469.9775391, -6906.6235352, 8916.9853516
4: -8278.4130859, 5828.1499023, -4843.2011719, 3387.4592285, -11077.5966797, 9915.6308594

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2824271, upper bound: 3612.4584383
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2824271, upper bound: 3612.4584501
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3900.7976074, 3535.5217285, -9314.7587891, 8907.3398438
1: -4990.1635742, 5599.2011719, -3143.2265625, 3485.4291992, -8116.8842773, 8176.7578125
2: -7583.7001953, 6017.1689453, -4758.0590820, 3743.1376953, -10753.4296875, 10036.6621094
3: -2778.9992676, 7645.6254883, -1746.4667969, 4804.0815430, -7245.8447266, 9048.1347656
4: -8278.4130859, 5828.1499023, -5203.6137695, 3635.8081055, -11330.2578125, 10288.9638672

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6306154, upper bound: 3612.2880265
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1646543, upper bound: 3612.2880340
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3791.6030273, 3438.4899902, -9222.5781250, 8794.5058594
1: -4990.1635742, 5599.2011719, -3056.1794434, 3391.3686523, -8027.0307617, 8086.3286133
2: -7583.7001953, 6017.1689453, -4632.7915039, 3642.6462402, -10657.8730469, 9905.0390625
3: -2778.9992676, 7645.6254883, -1696.6635742, 4674.0625000, -7113.4536133, 9000.3964844
4: -8278.4130859, 5828.1499023, -5063.8461914, 3536.7946777, -11237.3652344, 10142.3769531

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6306154, upper bound: 3612.3014622
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1646543, upper bound: 3612.3014697
time: 0.94 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.00 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.1656810, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2970405, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991905
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4498293, upper bound: 3611.8991905
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4498293, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4662449, upper bound: 3611.8999990
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4662449, upper bound: 3611.8999996
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4662450, upper bound: 3613.4906933
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4496796, upper bound: 3611.7109288
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4660951, upper bound: 3611.7109288
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2939960
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.4660952, upper bound: 3613.2940032
IS_A2_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.1663774
IS_A2_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.1663892
IS_A2_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.2977369
IS_A2_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.2977487
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4660730
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4661695
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505258
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4669414
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4668300
IS_A2_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2824237, upper bound: 3612.4016601
IS_A2_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2824231, upper bound: 3612.4016720
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2824271, upper bound: 3612.4584383
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.2824271, upper bound: 3612.4584501
IS_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3612.6306154, upper bound: 3612.2880265
IS_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.1646543, upper bound: 3612.2880340
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3612.6306154, upper bound: 3612.3014622
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.00
Output dim: 0, lower bound: -3613.1646543, upper bound: 3612.3014697

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1716.4951172, 1490.1773682, -5994.9316406, 5483.9711914, -6653.6518555, 7375.5742188
1: -1383.7639160, 1459.6079102, -4823.0478516, 5416.8784180, -6338.0664062, 6181.3256836
2: -2064.3198242, 1578.0806885, -7332.2827148, 5819.4853516, -7336.8447266, 8657.3134766
3: -771.3589478, 2073.4702148, -2689.2993164, 7391.1210938, -7946.5986328, 4567.3256836
4: -2268.2358398, 1536.5035400, -8004.1225586, 5635.4033203, -7349.5200195, 9288.9423828

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.2775367
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.4849670
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1685.1627197, 1460.3262939, -5994.0859375, 5483.5009766, -6623.2241211, 7356.5761719
1: -1359.2292480, 1430.2169189, -4822.3364258, 5416.3784180, -6313.7202148, 6162.0991211
2: -2029.3806152, 1546.2307129, -7331.2524414, 5818.7802734, -7302.5434570, 8637.5771484
3: -756.2575073, 2035.6834717, -2688.8532715, 7390.1210938, -7935.6181641, 4531.6391602
4: -2228.4836426, 1504.5333252, -8002.8623047, 5634.5092773, -7310.1787109, 9270.3515625

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.2775367
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.4849670
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1738.0455322, 1514.7304688, -6006.6376953, 5495.3383789, -6685.3193359, 7405.6674805
1: -1401.9702148, 1482.1727295, -4832.6240234, 5427.9511719, -6365.7758789, 6209.1762695
2: -2091.4250488, 1597.7427979, -7346.7944336, 5831.4833984, -7373.7016602, 8690.7021484
3: -780.3956299, 2102.7360840, -2694.8569336, 7406.3691406, -7970.1318359, 4599.6420898
4: -2296.6911621, 1556.6961670, -8019.8422852, 5647.1440430, -7388.3510742, 9322.7832031

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.2832547
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.4906850
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1706.8536377, 1484.5152588, -6005.2646484, 5494.3823242, -6654.4575195, 7385.2050781
1: -1377.6455078, 1453.4088135, -4831.4692383, 5426.9511719, -6341.1484375, 6188.5317383
2: -2056.8505859, 1566.4759521, -7345.0795898, 5830.2412109, -7339.2216797, 8669.1875000
3: -765.0651245, 2065.1086426, -2694.1982422, 7404.6845703, -7957.8330078, 4563.8432617
4: -2257.4387207, 1525.4367676, -8017.8305664, 5645.7539062, -7348.9140625, 9302.2880859

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.2832547
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.4906850
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -6000.5595703, 5481.5000000, -6891.8803711, 7588.2314453
1: -1577.3613281, 1670.7596436, -4827.2788086, 5416.1362305, -6529.7163086, 6389.4584961
2: -2346.6181641, 1806.2084961, -7339.1059570, 5818.5561523, -7616.9516602, 8882.6562500
3: -884.1437378, 2366.0300293, -2689.8771973, 7394.3012695, -8061.0180664, 4856.9970703
4: -2580.5910645, 1761.1538086, -8012.5571289, 5635.5048828, -7659.2924805, 9513.1796875

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9298068, upper bound: 3611.7205202
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0249802, upper bound: 3611.2478966
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4384829, upper bound: 3611.8943457
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9318343, upper bound: 3611.8519475
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3524528, upper bound: 3611.7623059
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4496573, upper bound: 3611.8593534
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497539, upper bound: 3611.8990446
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5993.7197266, 5476.6391602, -6787.4916992, 7500.4282227
1: -1496.7210693, 1578.8179932, -4821.8071289, 5411.2221680, -6445.0981445, 6303.6601562
2: -2230.0136719, 1707.1827393, -7331.0727539, 5812.9565430, -7496.2285156, 8789.8779297
3: -836.8752441, 2243.5935059, -2687.0917969, 7386.6718750, -8011.6079102, 4734.3095703
4: -2450.8525391, 1663.9034424, -8003.6357422, 5629.8364258, -7525.4628906, 9422.7470703

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.9298068, upper bound: 3611.7205209
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0249802, upper bound: 3611.2478965
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4384829, upper bound: 3611.8962237
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9318343, upper bound: 3611.8588196
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3524528, upper bound: 3611.7623059
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4370370, upper bound: 3611.8694387
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1542655, upper bound: 3611.8965085
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -6023.9780273, 5506.6738281, -6913.0585938, 7613.3247070
1: -1577.3613281, 1670.7596436, -4846.4980469, 5439.8500977, -6550.2490234, 6409.8808594
2: -2346.6181641, 1806.2084961, -7367.3061523, 5844.9184570, -7638.6967773, 8912.3750000
3: -884.1437378, 2366.0300293, -2701.1899414, 7424.4257812, -8091.0307617, 4866.9003906
4: -2580.5910645, 1761.1538086, -8042.7275391, 5660.8779297, -7681.1582031, 9545.7607422

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -6017.1884766, 5501.8466797, -6808.7119141, 7525.5766602
1: -1496.7210693, 1578.8179932, -4841.0747070, 5434.9667969, -6465.6616211, 6324.1372070
2: -2230.0136719, 1707.1827393, -7359.3339844, 5839.3637695, -7518.0219727, 8819.6699219
3: -836.8752441, 2243.5935059, -2698.4396973, 7416.8476562, -8041.6884766, 4744.2480469
4: -2450.8525391, 1663.9034424, -8033.8613281, 5655.2504883, -7547.3735352, 9455.3984375

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -6010.5014648, 5490.9716797, -6919.0048828, 7612.3110352
1: -1593.0754395, 1682.9068604, -4835.4340820, 5425.3837891, -6553.4111328, 6410.9189453
2: -2369.6191406, 1820.1040039, -7351.3925781, 5828.5864258, -7648.2553711, 8909.3818359
3: -891.3933716, 2388.5690918, -2694.5236816, 7407.1889648, -8080.5825195, 4882.5571289
4: -2604.9167480, 1773.1571045, -8025.8393555, 5645.2719727, -7692.3125000, 9540.5566406

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1984040, upper bound: 3611.2506227
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3524653, upper bound: 3611.8916143
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602214
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -6003.5390625, 5485.9443359, -6818.1972656, 7527.4980469
1: -1515.4637451, 1596.6520996, -4829.8476562, 5420.2929688, -6471.6816406, 6328.7089844
2: -2257.5788574, 1724.0441895, -7343.1748047, 5822.8090820, -7531.9536133, 8819.9306641
3: -845.8417969, 2271.2236328, -2691.6811523, 7399.3408203, -8032.2934570, 4764.9047852
4: -2480.0874023, 1678.2492676, -8016.7182617, 5639.4394531, -7563.1660156, 9452.9726562

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1984040, upper bound: 3611.2506227
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3524653, upper bound: 3611.8961807
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602219
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -6034.5048828, 5516.7587891, -6940.6611328, 7637.9526367
1: -1593.0754395, 1682.9068604, -4855.1269531, 5449.7060547, -6574.4331055, 6431.7841797
2: -2369.6191406, 1820.1040039, -7380.3242188, 5855.5810547, -7670.4882812, 8939.7675781
3: -891.3933716, 2388.5690918, -2706.0734863, 7438.1098633, -8111.3232422, 4892.6386719
4: -2604.9167480, 1773.1571045, -8056.8110352, 5671.2495117, -7714.6435547, 9573.8740234

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4534007, upper bound: 3613.4906431
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -6027.7373047, 5511.8564453, -6839.9775391, 7553.3281250
1: -1515.4637451, 1596.6520996, -4849.6899414, 5444.7412109, -6492.8242188, 6349.7260742
2: -2257.5788574, 1724.0441895, -7372.3383789, 5849.9409180, -7554.3222656, 8850.5439453
3: -845.8417969, 2271.2236328, -2703.3117676, 7430.4497070, -8063.2421875, 4775.0600586
4: -2480.0874023, 1678.2492676, -8047.9316406, 5665.5463867, -7585.6220703, 9486.5283203

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.4906431
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -5934.4106445, 5423.8842773, -6854.7026367, 7531.2216797
1: -1593.0754395, 1682.9068604, -4775.1572266, 5359.6376953, -6490.2646484, 6346.2919922
2: -2369.6191406, 1820.1040039, -7265.3222656, 5758.6669922, -7581.6889648, 8815.0917969
3: -891.3933716, 2388.5690918, -2659.8466797, 7317.9116211, -7987.1274414, 4849.1381836
4: -2604.9167480, 1773.1571045, -7928.4458008, 5575.0239258, -7626.6225586, 9434.5830078

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1983944, upper bound: 3611.2312796
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3523009, upper bound: 3611.7025441
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5927.9892578, 5419.2636719, -6754.2646484, 7446.9252930
1: -1515.4637451, 1596.6520996, -4769.9975586, 5354.9589844, -6408.9140625, 6264.4916992
2: -2257.5788574, 1724.0441895, -7257.7465820, 5753.3334961, -7465.7895508, 8726.2470703
3: -845.8417969, 2271.2236328, -2657.2333984, 7310.6386719, -7939.4062500, 4731.6884766
4: -2480.0874023, 1678.2492676, -7920.0268555, 5569.6269531, -7497.8637695, 9347.6640625

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1983944, upper bound: 3611.2298226
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3523009, upper bound: 3611.7071105
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -5956.0566406, 5447.4697266, -6874.0532227, 7554.4809570
1: -1593.0754395, 1682.9068604, -4793.0249023, 5381.9082031, -6509.1430664, 6365.2939453
2: -2369.6191406, 1820.1040039, -7291.5649414, 5783.4199219, -7601.5517578, 8842.6962891
3: -891.3933716, 2388.5690918, -2670.5202637, 7346.0468750, -8015.0107422, 4857.9345703
4: -2604.9167480, 1773.1571045, -7956.4228516, 5598.6425781, -7646.5146484, 9464.8115234

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2147160
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2733852
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5949.8598633, 5442.9956055, -6773.7568359, 7470.3925781
1: -1515.4637451, 1596.6520996, -4788.0234375, 5377.3828125, -6427.9404297, 6283.6464844
2: -2257.5788574, 1724.0441895, -7284.2465820, 5778.2500000, -7485.8120117, 8754.0917969
3: -845.8417969, 2271.2236328, -2668.0097656, 7339.0083008, -7967.5273438, 4740.5791016
4: -2480.0874023, 1678.2492676, -7948.2934570, 5593.3930664, -7517.9013672, 9378.1679688

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2501438
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2938694
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5937.9794922, 5427.9677734, -1920.2984619, 1669.9824219, -7498.6772461, 6795.2480469
1: -4777.6103516, 5363.5043945, -1547.3994141, 1641.2923584, -6314.9853516, 6441.6044922
2: -7266.6977539, 5760.4379883, -2304.2888184, 1774.5805664, -8782.2158203, 7510.3808594
3: -2659.7119141, 7321.0390625, -867.9794312, 2322.4943848, -4781.9521484, 7971.2392578
4: -7932.8330078, 5579.5351562, -2532.7153320, 1730.3000488, -9406.5566406, 7549.9814453

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5974.5874023, 5460.7929688, -1920.2984619, 1669.9824219, -7537.4843750, 6831.7036133
1: -4806.5922852, 5395.8237305, -1547.3994141, 1641.2923584, -6345.5722656, 6477.0356445
2: -7309.2700195, 5796.4418945, -2304.2888184, 1774.5805664, -8828.1582031, 7550.1831055
3: -2678.0915527, 7364.0517578, -867.9794312, 2322.4943848, -4802.4101562, 8016.5048828
4: -7979.6005859, 5613.9013672, -2532.7153320, 1730.3000488, -9456.9257812, 7588.0742188

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5947.6391602, 5437.2260742, -1940.5773926, 1689.8077393, -7523.5625000, 6823.6933594
1: -4785.5268555, 5372.5966797, -1564.4899902, 1655.1600342, -6337.7875977, 6466.5239258
2: -7278.6059570, 5770.2841797, -2329.3627930, 1788.5822754, -8809.7382812, 7543.5810547
3: -2664.2475586, 7333.5717773, -875.6460571, 2347.2778320, -4809.5947266, 7990.8398438
4: -7945.7182617, 5589.1264648, -2559.2373047, 1741.9539795, -9434.4257812, 7584.8999023

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5984.3276367, 5470.1235352, -1940.5773926, 1689.8077393, -7562.4487305, 6860.2138672
1: -4814.5820312, 5404.9848633, -1564.4899902, 1655.1600342, -6368.4448242, 6502.0175781
2: -7321.2846680, 5806.3652344, -2329.3627930, 1788.5822754, -8855.7812500, 7583.4560547
3: -2682.6687012, 7376.6948242, -875.6460571, 2347.2778320, -4830.0947266, 8036.2109375
4: -7992.5986328, 5623.5712891, -2559.2373047, 1741.9539795, -9484.9082031, 7623.0615234

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5744.9360352, 5274.9921875, -1920.2984619, 1669.9824219, -7294.8398438, 6612.7734375
1: -4623.5776367, 5211.1718750, -1547.3994141, 1641.2923584, -6151.3608398, 6262.4843750
2: -7045.5664062, 5598.2724609, -2304.2888184, 1774.5805664, -8542.8300781, 7316.5747070
3: -2576.0681152, 7097.7070312, -867.9794312, 2322.4943848, -4688.2568359, 7733.1425781
4: -7685.2802734, 5419.4414062, -2532.7153320, 1730.3000488, -9140.1259766, 7360.9077148

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505257
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505258
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5992.5185547, 5482.6782227, -1920.2984619, 1669.9824219, -7555.5805664, 6843.3188477
1: -4821.0917969, 5416.4809570, -1547.3994141, 1641.2923584, -6359.4438477, 6488.8237305
2: -7330.9321289, 5819.2934570, -2304.2888184, 1774.5805664, -8848.2607422, 7562.6708984
3: -2687.7221680, 7388.8339844, -867.9794312, 2322.4943848, -4809.8051758, 8036.8818359
4: -8002.2309570, 5635.3720703, -2532.7153320, 1730.3000488, -9478.9052734, 7599.6425781

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5755.0493164, 5284.6347656, -1940.5773926, 1689.8077393, -7320.1235352, 6641.5258789
1: -4631.8828125, 5220.5981445, -1564.4899902, 1655.1600342, -6174.5053711, 6287.6733398
2: -7058.0566406, 5608.4848633, -2329.3627930, 1788.5822754, -8570.8486328, 7350.0610352
3: -2580.7822266, 7110.8579102, -875.6460571, 2347.2778320, -4716.0385742, 7753.2802734
4: -7698.7900391, 5429.3847656, -2559.2373047, 1741.9539795, -9168.5283203, 7396.0976562

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6002.6977539, 5492.3476562, -1940.5773926, 1689.8077393, -7580.9301758, 6872.0781250
1: -4829.4252930, 5425.9453125, -1564.4899902, 1655.1600342, -6382.6176758, 6514.0454102
2: -7343.5122070, 5829.5498047, -2329.3627930, 1788.5822754, -8876.3623047, 7596.1977539
3: -2692.4621582, 7402.0239258, -875.6460571, 2347.2778320, -4837.6098633, 8057.0683594
4: -8015.8398438, 5645.3471680, -2559.2373047, 1741.9539795, -9507.4062500, 7634.8657227

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299
time: 0.77 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.25 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.2775367
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.2775367
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.1663774, upper bound: 3613.4849670
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.2832547
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.2832547
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4496573, upper bound: 3611.8593534
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4497539, upper bound: 3611.8990446
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4370370, upper bound: 3611.8694387
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.1542655, upper bound: 3611.8965085
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602214
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602219
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4534007, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2147160
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2733852
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2501438
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2938694
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505257
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505258
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.25
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1716.4951172, 1490.1773682, -5957.3764648, 5453.4472656, -6614.5551758, 7335.1318359
1: -1383.7639160, 1459.6079102, -4792.7373047, 5386.9121094, -6300.3759766, 6147.9897461
2: -2064.3198242, 1578.0806885, -7288.5107422, 5786.9916992, -7296.1938477, 8608.2861328
3: -771.3589478, 2073.4702148, -2672.9016113, 7347.4550781, -7897.0605469, 4549.3491211
4: -2268.2358398, 1536.5035400, -7955.5571289, 5603.1674805, -7308.7651367, 9234.8105469

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7310230, upper bound: 3613.4700707
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8827089, upper bound: 3613.4723822
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1685.1627197, 1460.3262939, -5956.3535156, 5452.8354492, -6583.9912109, 7315.9575195
1: -1359.2292480, 1430.2169189, -4791.8896484, 5386.2739258, -6275.8955078, 6128.6259766
2: -2029.3806152, 1546.2307129, -7287.2758789, 5786.1381836, -7261.7470703, 8588.3447266
3: -756.2575073, 2035.6834717, -2672.3710938, 7346.1948242, -7885.8310547, 4513.5756836
4: -2228.4836426, 1504.5333252, -7954.0683594, 5602.1162109, -7269.2739258, 9215.9892578

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7495960, upper bound: 3613.4698448
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0279435, upper bound: 3613.4829162
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1663055, upper bound: 3613.4849533
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1738.0455322, 1514.7304688, -5969.1591797, 5464.8798828, -6646.2778320, 7365.2944336
1: -1401.9702148, 1482.1727295, -4802.3740234, 5398.0434570, -6328.1313477, 6175.8940430
2: -2091.4250488, 1597.7427979, -7303.1166992, 5799.0708008, -7333.1196289, 8641.7558594
3: -780.3956299, 2102.7360840, -2678.5488281, 7362.8481445, -7920.7158203, 4581.7543945
4: -2296.6911621, 1556.6961670, -7971.3842773, 5614.9941406, -7347.6699219, 9268.7500000

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8018440, upper bound: 3613.4755648
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9193476, upper bound: 3613.4755667
time: 0.89 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 4.62 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3612.7310230, upper bound: 3613.4700707
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3612.8827089, upper bound: 3613.4723822
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3613.0279435, upper bound: 3613.4829162
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3613.1663055, upper bound: 3613.4849533
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3612.8018440, upper bound: 3613.4755648
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.62
Output dim: 0, lower bound: -3612.9193476, upper bound: 3613.4755667
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2972051, upper bound: 3613.4906850
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4496573, upper bound: 3611.8593534
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4497539, upper bound: 3611.8990446
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4370370, upper bound: 3611.8694387
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4505258, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602214
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4660729, upper bound: 3611.8602219
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4661694, upper bound: 3611.8999131
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4534007, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.2832631
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4534009, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4659912, upper bound: 3611.7108423
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.4659268, upper bound: 3611.6707964
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2147160
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2733852
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2501438
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.3430883, upper bound: 3613.2938694
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8593539, upper bound: 3613.4496573
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8990451, upper bound: 3613.4497539
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8602219, upper bound: 3613.4660729
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3611.8999131, upper bound: 3613.4661694
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505257
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4505258
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2775367, upper bound: 3613.4503360
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4669413
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.62
Output dim: 0, lower bound: -3613.2831954, upper bound: 3613.4668299
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=4653.345703125
rel_dist={0: [-3614.761877702603, 3614.7618777026037]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2022982, upper bound: 3613.7565309
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565204, upper bound: 3613.7565215
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -3614.2022982, upper bound: 3613.7565309
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -3613.7565204, upper bound: 3613.7565215

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2502.3757324, 2150.9697266, -4423.8598633, 4467.8574219
1: -1831.3944092, 1927.9595947, -2015.7012939, 2107.5605469, -3938.9548340, 3943.6608887
2: -2712.4650879, 2088.7180176, -2978.1022949, 2285.2739258, -4997.7377930, 5066.8188477
3: -1026.7778320, 2724.7028809, -1126.3112793, 2983.9003906, -4010.6782227, 3851.0141602
4: -2982.6787109, 2035.6074219, -3276.4265137, 2226.9489746, -5209.6279297, 5312.0336914

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7201630
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7565215
time: 0.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4122.1186523, 3711.3884277, -2488.1984863, 2140.8833008, -6263.0019531, 6028.3676758
1: -3322.9208984, 3653.8891602, -2004.5759277, 2097.7246094, -5420.6450195, 5516.1621094
2: -5012.0878906, 3928.0732422, -2962.2211914, 2274.6040039, -7257.1293945, 6732.2988281
3: -1844.6010742, 5052.4194336, -1120.9125977, 2969.2133789, -4772.4760742, 6118.5239258
4: -5483.7353516, 3816.0854492, -3258.6638184, 2216.5571289, -7682.1782227, 6908.3496094

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7201630
time: 0.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7565215
time: 0.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.13 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7201630
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -3613.7201630, upper bound: 3613.7565215
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7201630
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.13
Output dim: 0, lower bound: -3613.7565215, upper bound: 3613.7565215

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2272.8901367, 1965.4815674, -4238.3715820, 4238.3715820
1: -1831.3944092, 1927.9595947, -1831.3944092, 1927.9595947, -3759.3540039, 3759.3540039
2: -2712.4650879, 2088.7180176, -2712.4650879, 2088.7180176, -4801.1821289, 4801.1816406
3: -1026.7778320, 2724.7028809, -1026.7778320, 2724.7028809, -3751.4807129, 3751.4807129
4: -2982.6787109, 2035.6074219, -2982.6787109, 2035.6074219, -5018.2861328, 5018.2861328

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311068, upper bound: 3612.9658089
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657996, upper bound: 3612.9658089
time: 0.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -4120.3857422, 3709.5341797, -5813.6098633, 6085.8671875
1: -1831.3944092, 1927.9595947, -3321.5349121, 3652.0288086, -5343.1538086, 5249.4946289
2: -2712.4650879, 2088.7180176, -5009.9873047, 3926.0749512, -6484.9042969, 7074.9960938
3: -1026.7778320, 2724.7028809, -1843.7044678, 5050.1254883, -6024.3237305, 4529.7241211
4: -2982.6787109, 2035.6074219, -5481.4267578, 3814.1491699, -6634.8647461, 7504.7265625

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5311068, upper bound: 3613.4908192
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657996, upper bound: 3613.4908192
time: 0.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4120.3857422, 3709.5341797, -2272.8901367, 1965.4815674, -6085.8671875, 5813.6093750
1: -3321.5349121, 3652.0288086, -1831.3944092, 1927.9595947, -5249.4946289, 5343.1538086
2: -5009.9873047, 3926.0749512, -2712.4650879, 2088.7180176, -7074.9960938, 6484.9038086
3: -1843.7044678, 5050.1254883, -1026.7778320, 2724.7028809, -4529.7236328, 6024.3237305
4: -5481.4267578, 3814.1491699, -2982.6787109, 2035.6074219, -7504.7265625, 6634.8647461

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295729, upper bound: 3612.9657827
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908036, upper bound: 3612.9657996
time: 0.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4128.4018555, 3718.1166992, -4128.4018555, 3718.1166992, -7603.6601562, 7603.6606445
1: -3327.9492188, 3660.6428223, -3327.9492188, 3660.6428223, -6780.5097656, 6780.5092773
2: -5019.7089844, 3935.3261719, -5019.7089844, 3935.3261719, -8661.0214844, 8661.0224609
3: -1847.8551025, 5060.7402344, -1847.8551025, 5060.7402344, -6755.2485352, 6755.2485352
4: -5492.1123047, 3823.1091309, -5492.1123047, 3823.1091309, -9018.0908203, 9018.0898438

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295730, upper bound: 3613.1525273
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908076, upper bound: 3613.4896465
time: 0.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.5311068, upper bound: 3612.9658089
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -3612.9657996, upper bound: 3612.9658089
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.5311068, upper bound: 3613.4908192
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3612.9657996, upper bound: 3613.4908192
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.0295729, upper bound: 3612.9657827
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.4908036, upper bound: 3612.9657996
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.0295730, upper bound: 3613.1525273
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.32
Output dim: 0, lower bound: -3613.4908076, upper bound: 3613.4896465

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -2272.8901367, 1965.4815674, -3999.3693848, 4041.9062500
1: -1638.3045654, 1737.5650635, -1831.3944092, 1927.9595947, -3566.2641602, 3568.9594727
2: -2435.9470215, 1879.3935547, -2712.4650879, 2088.7180176, -4524.6650391, 4591.8583984
3: -920.3916626, 2455.5791016, -1026.7778320, 2724.7028809, -3645.0944824, 3482.3569336
4: -2677.5119629, 1832.2690430, -2982.6787109, 2035.6074219, -4713.1181641, 4814.9477539

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -4119.0043945, 3708.0573730, -5576.6225586, 5888.0214844
1: -1638.3045654, 1737.5650635, -3320.4311523, 3650.5466309, -5151.4418945, 5057.9960938
2: -2435.9470215, 1879.3935547, -5008.3129883, 3924.4824219, -6212.5473633, 6870.3544922
3: -920.3916626, 2455.5791016, -1842.9903564, 5048.2968750, -5918.9414062, 4263.2993164
4: -2677.5119629, 1832.2690430, -5479.5859375, 3812.6076660, -6333.8500977, 7306.1171875

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295819
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -4120.4204102, 3711.1000977, -7449.7148438, 7459.5146484
1: -3211.7971191, 3524.2065430, -3321.5488281, 3653.8020020, -6656.2724609, 6643.0014648
2: -4848.2338867, 3788.9609375, -5010.3110352, 3927.9245605, -8480.1093750, 8508.5693359
3: -1780.8483887, 4870.8652344, -1844.2166748, 5051.1689453, -6674.5473633, 6574.6728516
4: -5298.1914062, 3677.4733887, -5481.7441406, 3815.8579102, -8813.4433594, 8869.1152344

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295814
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6083.4877930, 5547.8891602, -2250.3500977, 1947.1267090, -7918.6787109, 7261.7397461
1: -4895.1855469, 5481.2202148, -1813.4694824, 1909.9956055, -6703.2719727, 6840.6655273
2: -7437.6269531, 5890.8720703, -2686.4743652, 2069.2993164, -9253.0634766, 8038.9516602
3: -2726.1181641, 7490.3676758, -1017.0298462, 2699.0520020, -5232.0273438, 8293.9052734
4: -8120.0102539, 5706.4619141, -2953.7026367, 2016.5915527, -9887.2207031, 8113.5161133

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -4120.4204102, 3711.1000977, -9547.5849609, 9173.3593750
1: -5010.5698242, 5614.1787109, -3321.5488281, 3653.8020020, -8334.8261719, 8397.6123047
2: -7611.5888672, 6033.4814453, -5010.3110352, 3927.9245605, -11004.1044922, 10346.2832031
3: -2789.6572266, 7670.8759766, -1844.2166748, 5051.1689453, -7527.2822266, 9190.0566406
4: -8309.2480469, 5844.3500977, -5481.7441406, 3815.8579102, -11579.7050781, 10625.6962891

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295301
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.4896424
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.59 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9658089, upper bound: 3612.9658089
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295819
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295814
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908152
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3613.4908152, upper bound: 3612.9657996
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295301
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.4896424

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -6059.6582031, 5528.2924805, -7031.3588867, 7726.5590820
1: -1638.3045654, 1737.5650635, -4875.8427734, 5461.8833008, -6650.8359375, 6517.2197266
2: -2435.9470215, 1879.3935547, -7408.7856445, 5869.5312500, -7774.8994141, 9042.6601562
3: -920.3916626, 2455.5791016, -2716.2336426, 7462.2182617, -8173.2329102, 4982.8125000
4: -2677.5119629, 1832.2690430, -8088.4770508, 5685.5073242, -7824.1801758, 9679.9404297

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057508, upper bound: 3613.4907875
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310877, upper bound: 3613.4907886
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -6220.5375977, 5677.4472656, -9030.6015625, 9414.2050781
1: -3211.7971191, 3524.2065430, -5006.7314453, 5609.0083008, -8282.5947266, 8207.2080078
2: -4848.2338867, 3788.9609375, -7605.8393555, 6027.9179688, -10178.0791016, 10863.4648438
3: -1780.8483887, 4870.8652344, -2787.1591797, 7664.4882812, -9117.0068359, 7358.1440430
4: -5298.1914062, 3677.4733887, -8302.9218750, 5838.9990234, -10434.8554688, 11442.8203125

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657724, upper bound: 3613.1028309
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9265725, upper bound: 3613.0232128
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6059.6582031, 5528.2924805, -2033.8878174, 1769.0169678, -7726.5590820, 7031.3588867
1: -4875.8427734, 5461.8833008, -1638.3045654, 1737.5650635, -6517.2202148, 6650.8359375
2: -7408.7856445, 5869.5312500, -2435.9470215, 1879.3935547, -9042.6601562, 7774.9008789
3: -2716.2338867, 7462.2187500, -920.3916626, 2455.5791016, -4982.8129883, 8173.2333984
4: -8088.4775391, 5685.5073242, -2677.5119629, 1832.2690430, -9679.9414062, 7824.1811523

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907812, upper bound: 3612.9266012
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941500, upper bound: 3612.9265725
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6220.5375977, 5677.4472656, -3987.9494629, 3574.2734375, -9414.2050781, 9030.6025391
1: -5006.7314453, 5609.0083008, -3211.7971191, 3524.2065430, -8207.2089844, 8282.5947266
2: -7605.8393555, 6027.9179688, -4848.2338867, 3788.9609375, -10863.4658203, 10178.0791016
3: -2787.1591797, 7664.4882812, -1780.8483887, 4870.8652344, -7358.1435547, 9117.0068359
4: -8302.9218750, 5838.9990234, -5298.1914062, 3677.4733887, -11442.8203125, 10434.8554688

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907817, upper bound: 3612.9266012
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941504, upper bound: 3612.9265725
time: 4.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -3904.8264160, 3541.1750488, -9358.8994141, 8945.6386719
1: -5010.5698242, 5614.1787109, -3146.5612793, 3492.1206055, -8155.9272461, 8213.0507812
2: -7611.5888672, 6033.4814453, -4764.1665039, 3751.3303223, -10806.4443359, 10082.0341797
3: -2789.6572266, 7670.8759766, -1749.9495850, 4810.4135742, -7274.6840820, 9085.5927734
4: -8309.2480469, 5844.3500977, -5208.8427734, 3642.6633301, -11386.6064453, 10332.4433594

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907764, upper bound: 3612.9552912
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941484, upper bound: 3612.9552648
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -6225.2827148, 5682.6333008, -11132.6572266, 11132.6582031
1: -5010.5698242, 5614.1787109, -5010.5698242, 5614.1787109, -9965.4804688, 9965.4804688
2: -7611.5888672, 6033.4814453, -7611.5888672, 6033.4814453, -12706.6269531, 12706.6269531
3: -2789.6572266, 7670.8759766, -2789.6572266, 7670.8759766, -9975.6748047, 9975.6748047
4: -8309.2480469, 5844.3500977, -8309.2480469, 5844.3500977, -13205.4160156, 13205.4160156

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907786, upper bound: 3613.2934108
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941502, upper bound: 3613.2932841
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.62 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.4057508, upper bound: 3613.4907875
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.5310877, upper bound: 3613.4907886
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3612.9657724, upper bound: 3613.1028309
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3612.9265725, upper bound: 3613.0232128
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.4907812, upper bound: 3612.9266012
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.2941500, upper bound: 3612.9265725
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.4907817, upper bound: 3612.9266012
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.2941504, upper bound: 3612.9265725
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.4907764, upper bound: 3612.9552912
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.2941484, upper bound: 3612.9552648
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.4907786, upper bound: 3613.2934108
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 0, lower bound: -3613.2941502, upper bound: 3613.2932841

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -6018.4794922, 5495.8173828, -6756.2319336, 7472.0146484
1: -1442.5327148, 1525.0668945, -4842.5541992, 5429.5756836, -6424.2260742, 6271.8627930
2: -2150.3056641, 1650.0454102, -7359.5351562, 5834.0302734, -7455.8144531, 8765.0400391
3: -806.7396851, 2160.4394531, -2699.0346680, 7414.4360352, -8012.9853516, 4671.7768555
4: -2361.8547363, 1606.7463379, -8034.0791016, 5650.2719727, -7475.5673828, 9400.6640625

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057508, upper bound: 3613.4856053
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4057318, upper bound: 3613.2941184
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6053.9223633, 5523.0703125, -6983.1567383, 7682.2324219
1: -1604.6556396, 1701.4029541, -4871.2006836, 5456.7231445, -6611.0908203, 6474.2998047
2: -2386.9431152, 1840.0159912, -7401.8569336, 5863.9545898, -7718.8100586, 8994.9589844
3: -900.5989380, 2405.9113770, -2713.6914062, 7455.0996094, -8147.2641602, 4928.0102539
4: -2623.7451172, 1794.3088379, -8080.9213867, 5680.0712891, -7763.3930664, 9632.3583984

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310902, upper bound: 3613.4907737
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310724, upper bound: 3613.2941468
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6060.2988281, 5524.8271484, -2031.1785889, 1766.6568604, -7724.7685547, 7022.7778320
1: -4875.5463867, 5458.0791016, -1636.1564941, 1735.2275391, -6514.3388672, 6642.6157227
2: -7407.1459961, 5864.4165039, -2432.7812500, 1876.8747559, -9038.5361328, 7763.6977539
3: -2714.9853516, 7460.1118164, -919.1136475, 2452.2871094, -4976.4697266, 8170.7329102
4: -8088.7265625, 5681.7841797, -2674.0334473, 1829.8524170, -9677.0908203, 7813.4931641

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4856053, upper bound: 3613.4057508
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907737, upper bound: 3613.5310902
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6232.5551758, 5684.6313477, -3984.0727539, 3570.8161621, -9422.1728516, 9030.0058594
1: -5015.7568359, 5615.6601562, -3208.7543945, 3520.7617188, -8212.0029297, 8282.6689453
2: -7618.2021484, 6034.1894531, -4843.7485352, 3785.2382812, -10871.0634766, 10175.5136719
3: -2791.2629395, 7676.6879883, -1779.0634766, 4866.2109375, -7355.4252930, 9127.5000000
4: -8318.3222656, 5846.1894531, -5293.2880859, 3673.9799805, -11453.1132812, 10432.1542969

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941486, upper bound: 3612.9265725
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941486, upper bound: 3612.9265725
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -3900.9096680, 3537.6669922, -9367.0009766, 8945.1982422
1: -5019.8442383, 5621.1367188, -3143.4506836, 3488.6738281, -8160.8374023, 8213.2910156
2: -7624.3247070, 6040.0839844, -4759.6416016, 3747.6030273, -10814.2363281, 10079.6445312
3: -2793.9113770, 7683.4697266, -1748.0971680, 4805.7324219, -7271.9804688, 9096.3466797
4: -8325.0566406, 5851.8598633, -5203.8710938, 3639.0812988, -11397.0791016, 10329.8759766

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3794465, upper bound: 3612.5086628
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907732, upper bound: 3612.9552724
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -6221.2314453, 5679.0273438, -11140.4423828, 11132.0546875
1: -5019.8442383, 5621.1367188, -5007.3735352, 5610.6137695, -9970.0839844, 9965.5791016
2: -7624.3247070, 6040.0839844, -7606.9096680, 6029.6420898, -12714.0996094, 12703.9951172
3: -2793.9113770, 7683.4697266, -2787.7727051, 7666.0498047, -9972.7490234, 9986.3271484
4: -8325.0566406, 5851.8598633, -8304.1103516, 5840.6464844, -13215.5615234, 13202.5625000

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941484, upper bound: 3613.2932824
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941484, upper bound: 3613.2932824
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.51 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.4057508, upper bound: 3613.4856053
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.4057318, upper bound: 3613.2941184
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.5310902, upper bound: 3613.4907737
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.5310724, upper bound: 3613.2941468
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.4856053, upper bound: 3613.4057508
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.4907737, upper bound: 3613.5310902
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.2941486, upper bound: 3612.9265725
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.2941486, upper bound: 3612.9265725
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.3794465, upper bound: 3612.5086628
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.4907732, upper bound: 3612.9552724
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.2941484, upper bound: 3613.2932824
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -3613.2941484, upper bound: 3613.2932824

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1785.0306396, 1550.0185547, -6015.1093750, 5489.3457031, -6742.3945312, 7464.2929688
1: -1438.8216553, 1520.9581299, -4839.0937500, 5422.8095703, -6411.2250977, 6263.9091797
2: -2144.8640137, 1645.6396484, -7353.1679688, 5825.7153320, -7438.8134766, 8754.1591797
3: -804.6100464, 2154.7856445, -2696.1132812, 7407.7651367, -8004.9096680, 4661.2753906
4: -2355.8813477, 1602.4992676, -8029.1367188, 5643.4570312, -7458.9921875, 9390.5849609

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1240670, upper bound: 3613.4809622
time: 2.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2977380, upper bound: 3613.4855005
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -5943.3735352, 5425.2851562, -6686.9067383, 7393.5444336
1: -1442.5327148, 1525.0668945, -4782.3530273, 5360.0043945, -6355.6918945, 6208.4848633
2: -2150.3056641, 1650.0454102, -7272.0727539, 5759.1455078, -7382.2006836, 8671.3037109
3: -806.7396851, 2160.4394531, -2663.4238281, 7322.6123047, -7918.9956055, 4636.1591797
4: -2361.8547363, 1606.7463379, -7937.2475586, 5576.3969727, -7403.5332031, 9296.7050781

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1237995, upper bound: 3613.2883167
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2977349, upper bound: 3613.2940356
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1989.0161133, 1729.7738037, -6054.3081055, 5519.4052734, -6974.3701172, 7680.1372070
1: -1602.4916992, 1699.0515137, -4870.7109375, 5452.7192383, -6602.6723633, 6471.1875000
2: -2383.7546387, 1837.4832764, -7399.9116211, 5858.6293945, -7707.3940430, 8990.4785156
3: -899.3161621, 2402.5927734, -2712.3417969, 7452.7006836, -8144.4570312, 4921.5317383
4: -2620.2399902, 1791.8824463, -8080.8398438, 5676.1499023, -7752.4931641, 9629.1318359

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4505143, upper bound: 3613.4898848
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669373, upper bound: 3613.4906909
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -5979.4877930, 5453.2133789, -6914.4311523, 7604.4047852
1: -1604.6556396, 1701.4029541, -4811.5244141, 5387.8735352, -6543.1962891, 6411.4492188
2: -2386.9431152, 1840.0159912, -7315.1958008, 5789.7998047, -7645.8276367, 8902.0146484
3: -900.5989380, 2405.9113770, -2678.3981934, 7364.1850586, -8054.1870117, 4892.6591797
4: -2623.7451172, 1794.3088379, -7984.9213867, 5606.8911133, -7691.9501953, 9529.2255859

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1237995, upper bound: 3613.2932583
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669362, upper bound: 3613.2940614
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6015.1098633, 5489.3457031, -1785.0306396, 1550.0185547, -7464.2939453, 6742.3935547
1: -4839.0937500, 5422.8095703, -1438.8216553, 1520.9581299, -6263.9086914, 6411.2246094
2: -7353.1679688, 5825.7158203, -2144.8640137, 1645.6396484, -8754.1591797, 7438.8139648
3: -2696.1135254, 7407.7651367, -804.6100464, 2154.7856445, -4661.2753906, 8004.9096680
4: -8029.1367188, 5643.4570312, -2355.8813477, 1602.4992676, -9390.5849609, 7458.9921875

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2377336, upper bound: 3613.2965021
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6054.3081055, 5519.4052734, -1989.0161133, 1729.7738037, -7680.1372070, 6974.3701172
1: -4870.7109375, 5452.7192383, -1602.4916992, 1699.0515137, -6471.1875000, 6602.6723633
2: -7399.9116211, 5858.6293945, -2383.7546387, 1837.4832764, -8990.4785156, 7707.3940430
3: -2712.3417969, 7452.7006836, -899.3161621, 2402.5927734, -4921.5317383, 8144.4565430
4: -8080.8398438, 5676.1499023, -2620.2399902, 1791.8824463, -9629.1318359, 7752.4926758

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8999895, upper bound: 3613.4662252
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6220.8188477, 5676.2934570, -3658.4387207, 3324.6960449, -9124.0361328, 8680.2568359
1: -5006.3930664, 5607.5483398, -2948.6689453, 3280.1208496, -7926.9077148, 7997.9462891
2: -7604.8759766, 6025.4760742, -4477.3779297, 3524.0493164, -10556.3535156, 9770.6689453
3: -2786.4316406, 7663.8798828, -1637.6434326, 4512.9521484, -6964.1718750, 8959.2001953
4: -8303.4892578, 5837.5219727, -4891.0341797, 3418.8139648, -11141.9208984, 9989.7197266

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2759168, upper bound: 3612.5086505
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2759155, upper bound: 3612.5086628
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -3856.0683594, 3500.0046387, -9320.5126953, 8897.2558594
1: -5019.8442383, 5621.1367188, -3107.3972168, 3452.1826172, -8116.5717773, 8174.1767578
2: -7624.3247070, 6040.0839844, -4707.7021484, 3708.0732422, -10766.1367188, 10022.3896484
3: -2793.9113770, 7683.4697266, -1727.5924072, 4753.1562500, -7213.5991211, 9073.9980469
4: -8325.0566406, 5851.8598633, -5146.5771484, 3600.5832520, -11349.3935547, 10266.9804688

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8999895, upper bound: 3612.9325003
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901
time: 1.70 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.45 seconds
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.1240670, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.2977380, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.1237995, upper bound: 3613.2883167
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.2977349, upper bound: 3613.2940356
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4505143, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4669373, upper bound: 3613.4906909
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.1237995, upper bound: 3613.2932583
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4669362, upper bound: 3613.2940614
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3611.2377336, upper bound: 3613.2965021
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3611.8999895, upper bound: 3613.4662252
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.2759168, upper bound: 3612.5086505
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.2759155, upper bound: 3612.5086628
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.45
Output dim: 0, lower bound: -3611.8999895, upper bound: 3612.9325003
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.45
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1718.1533203, 1493.0067139, -6003.9028320, 5479.5874023, -6667.6669922, 7397.3706055
1: -1385.1115723, 1463.6422119, -4830.0112305, 5413.0585938, -6349.4467773, 6200.8017578
2: -2067.3376465, 1582.9438477, -7339.5429688, 5815.1103516, -7353.1801758, 8682.3330078
3: -773.2760010, 2076.5085449, -2691.2211914, 7394.1567383, -7961.1240234, 4579.3618164
4: -2270.4079590, 1541.0151367, -8014.2392578, 5633.0078125, -7365.6103516, 9316.7421875

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1171765, upper bound: 3611.2377336
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1740.0734863, 1517.1613770, -6002.7373047, 5479.9399414, -6687.5732422, 7413.1455078
1: -1403.7215576, 1485.5802002, -4829.2695312, 5413.3271484, -6365.8203125, 6217.1684570
2: -2095.0383301, 1601.3913574, -7339.0961914, 5815.3686523, -7377.7714844, 8698.9423828
3: -781.8970337, 2105.8525391, -2690.7658691, 7394.1748047, -7968.2392578, 4605.3002930
4: -2299.6218262, 1560.1614990, -8013.4531250, 5633.2163086, -7392.5888672, 9332.3916016

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2965021, upper bound: 3611.2377336
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1917.5677490, 1667.5399170, -6044.3994141, 5511.0937500, -6896.1572266, 7611.0327148
1: -1545.2342529, 1638.9377441, -4862.6796875, 5444.5449219, -6538.4868164, 6404.2734375
2: -2301.1032715, 1772.0444336, -7387.8056641, 5849.6645508, -7617.5908203, 8914.7304688
3: -866.7033081, 2319.1787109, -2708.0883789, 7440.7314453, -8100.8051758, 4834.8608398
4: -2529.2170410, 1727.8784180, -8067.6157227, 5667.3183594, -7654.5209961, 9553.7509766

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991838
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1937.9083252, 1687.3262939, -6042.9682617, 5511.2705078, -6914.3203125, 7624.3642578
1: -1562.3759766, 1652.6922607, -4861.7202148, 5444.7177734, -6553.4277344, 6417.7719727
2: -2326.2473145, 1786.0650635, -7386.9619141, 5849.7783203, -7639.8237305, 8928.7060547
3: -874.3999023, 2344.0041504, -2707.5263672, 7440.3627930, -8106.8286133, 4856.9326172
4: -2555.8173828, 1739.5623779, -8066.3964844, 5667.3930664, -7678.8154297, 9566.6787109

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1940.5773926, 1689.8077393, -5967.8623047, 5444.6425781, -6853.8632812, 7548.2597656
1: -1564.4899902, 1655.1600342, -4802.2866211, 5379.4248047, -6493.4414062, 6357.7416992
2: -2329.3627930, 1788.5822754, -7301.9130859, 5780.4755859, -7577.6948242, 8839.7939453
3: -875.6460571, 2347.2778320, -2673.3745117, 7351.4765625, -8016.0957031, 4827.8071289
4: -2559.2373047, 1741.9539795, -7970.1196289, 5597.6801758, -7617.7089844, 9466.3046875

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2955875, upper bound: 3612.7880083
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3115210, upper bound: 3613.0449949
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5973.2011719, 5463.3686523, -1773.2364502, 1540.0426025, -7408.9667969, 6690.7094727
1: -4805.5219727, 5396.8671875, -1429.4506836, 1511.1704102, -6217.7119141, 6363.9433594
2: -7306.6943359, 5798.1796875, -2131.2707520, 1635.0765381, -8690.7509766, 7382.7944336
3: -2679.3227539, 7363.9204102, -799.3659058, 2140.9614258, -4625.3100586, 7949.6767578
4: -7976.1967773, 5615.2280273, -2340.8388672, 1592.2003174, -9321.2099609, 7402.4941406

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5990.4067383, 5469.9062500, -1989.0161133, 1729.7738037, -7610.9711914, 6914.3930664
1: -4819.0522461, 5404.7382812, -1602.4916992, 1699.0515137, -6415.3344727, 6545.4858398
2: -7326.8935547, 5806.5654297, -2383.7546387, 1837.4832764, -8909.3212891, 7644.4990234
3: -2684.9260254, 7380.4814453, -899.3161621, 2402.5927734, -4889.9575195, 8066.1977539
4: -7999.5576172, 5624.3369141, -2620.2399902, 1791.8824463, -9539.1259766, 7690.3525391

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8991838, upper bound: 3613.4497980
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8991838, upper bound: 3613.4662252
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6013.2250977, 5494.5561523, -1978.0841064, 1720.5938721, -7626.4165039, 6924.3837891
1: -4837.7924805, 5427.9750977, -1593.8105469, 1690.0715332, -6426.4038086, 6556.9956055
2: -7354.3730469, 5832.3632812, -2371.1567383, 1827.8018799, -8928.7890625, 7653.2709961
3: -2695.9714355, 7409.9199219, -894.4675293, 2389.7939453, -4886.9326172, 8090.7099609
4: -8028.9497070, 5649.1728516, -2606.2458496, 1782.4775391, -9561.6933594, 7697.9111328

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3843.8420410, 3489.7048340, -9268.5732422, 8847.7998047
1: -4990.1635742, 5599.2011719, -3097.5605469, 3442.1884766, -8073.1494141, 8129.0869141
2: -7583.7001953, 6017.1689453, -4693.4838867, 3697.3427734, -10706.9921875, 9968.5009766
3: -2778.9992676, 7645.6254883, -1722.1224365, 4738.7392578, -7178.1005859, 9023.8642578
4: -8278.4130859, 5828.1499023, -5130.8554688, 3590.0625000, -11284.7734375, 10212.5566406

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901
time: 0.89 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.60 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.1171765, upper bound: 3611.2377336
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.2965021, upper bound: 3611.2377336
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991838
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.2955875, upper bound: 3612.7880083
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.3115210, upper bound: 3613.0449949
IS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
IS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4855005, upper bound: 3613.2977384
IS_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3611.8991838, upper bound: 3613.4497980
IS_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3611.8991838, upper bound: 3613.4662252
IS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
IS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4906909, upper bound: 3613.4669374
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.60
Output dim: 0, lower bound: -3613.4906892, upper bound: 3612.9331901

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1718.1533203, 1493.0067139, -5962.9018555, 5454.1455078, -6628.1328125, 7352.7514648
1: -1385.1115723, 1463.6422119, -4797.1723633, 5387.6796875, -6312.0019531, 6164.9365234
2: -2067.3376465, 1582.9438477, -7294.1650391, 5788.2211914, -7311.2153320, 8630.2744141
3: -773.2760010, 2076.5085449, -2674.7836914, 7351.2304688, -7912.0273438, 4557.4458008
4: -2270.4079590, 1541.0151367, -7962.4995117, 5605.3920898, -7324.5454102, 9258.6367188

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1740.0734863, 1517.1613770, -5974.7592773, 5465.6391602, -6660.0678711, 7382.0029297
1: -1403.7215576, 1485.5802002, -4806.8510742, 5398.9091797, -6340.1440430, 6192.1181641
2: -2095.0383301, 1601.3913574, -7308.7998047, 5800.3393555, -7348.6303711, 8662.6992188
3: -781.8970337, 2105.8525391, -2680.2924805, 7366.6943359, -7935.1660156, 4589.7568359
4: -2299.6218262, 1560.1614990, -7978.3593750, 5617.2026367, -7363.9477539, 9291.6992188

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1917.5677490, 1667.5399170, -5980.7900391, 5461.8881836, -6836.4267578, 7542.1474609
1: -1545.2342529, 1638.9377441, -4811.2548828, 5396.8574219, -6481.5551758, 6348.6430664
2: -2301.1032715, 1772.0444336, -7315.1401367, 5797.8974609, -7554.9453125, 8833.9052734
3: -866.7033081, 2319.1787109, -2680.7770996, 7368.8618164, -8022.8759766, 4803.3642578
4: -2529.2170410, 1727.8784180, -7986.7231445, 5615.7910156, -7592.6186523, 9464.1191406

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991833
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991838
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1917.5677490, 1667.5399170, -6003.8701172, 5486.5385742, -6857.2841797, 7566.9140625
1: -1545.2342529, 1638.9377441, -4830.2075195, 5420.0678711, -6501.7353516, 6368.8056641
2: -2301.1032715, 1772.0444336, -7342.9389648, 5823.7172852, -7576.3193359, 8863.2402344
3: -866.7033081, 2319.1787109, -2691.9150391, 7398.5151367, -8052.4487305, 4813.1684570
4: -2529.2170410, 1727.8784180, -8016.4599609, 5640.6679688, -7614.1367188, 9496.2851562

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1937.9083252, 1687.3262939, -5759.0585938, 5271.9370117, -6645.7128906, 7324.8759766
1: -1562.3759766, 1652.6922607, -4633.7290039, 5208.2861328, -6290.3823242, 6176.9741211
2: -2326.2473145, 1786.0650635, -7058.9804688, 5595.8818359, -7354.3300781, 8576.1230469
3: -874.3999023, 2344.0041504, -2579.9072266, 7103.9067383, -7754.0292969, 4717.9438477
4: -2555.8173828, 1739.5623779, -7702.3764648, 5417.1079102, -7400.2397461, 9176.0712891

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1937.9083252, 1687.3262939, -6009.2695312, 5484.7441406, -6879.1889648, 7587.9204102
1: -1562.3759766, 1652.6922607, -4834.5073242, 5418.7299805, -6519.6879883, 6387.6674805
2: -2326.2473145, 1786.0650635, -7347.6621094, 5821.4819336, -7603.3461914, 8884.3964844
3: -874.3999023, 2344.0041504, -2692.8964844, 7401.5190430, -8062.2968750, 4840.7617188
4: -2555.8173828, 1739.5623779, -8022.8041992, 5639.4096680, -7642.1679688, 9517.7958984

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5965.8027344, 5456.6152344, -1769.7990723, 1536.1658936, -7391.5078125, 6680.2260742
1: -4799.6293945, 5390.0004883, -1426.6289062, 1506.1417236, -6201.6044922, 6354.3730469
2: -7297.9609375, 5790.8789062, -2126.0754395, 1629.1384277, -8670.4072266, 7369.8037109
3: -2676.0402832, 7355.1206055, -796.7026978, 2136.0278320, -4615.8886719, 7935.6235352
4: -7966.6318359, 5608.1030273, -2336.3110352, 1586.4744873, -9297.5957031, 7390.2314453

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977271
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977384
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5964.2739258, 5455.5253906, -1738.8707275, 1506.3010254, -7370.2377930, 6649.5205078
1: -4798.3588867, 5388.8706055, -1402.4498291, 1476.2041016, -6180.2192383, 6329.7709961
2: -7296.0888672, 5789.5239258, -2091.6687012, 1597.3557129, -8648.1669922, 7335.3339844
3: -2675.3889160, 7353.2407227, -781.8993530, 2098.5158691, -4579.8759766, 7923.3081055
4: -7964.4726562, 5606.6396484, -2297.1281738, 1554.6807861, -9277.3925781, 7350.7680664

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977271
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977384
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5980.7900391, 5461.8881836, -1917.5677490, 1667.5399170, -7542.1474609, 6836.4272461
1: -4811.2548828, 5396.8574219, -1545.2342529, 1638.9377441, -6348.6425781, 6481.5546875
2: -7315.1401367, 5797.8974609, -2301.1032715, 1772.0444336, -8833.9052734, 7554.9453125
3: -2680.7770996, 7368.8618164, -866.7033081, 2319.1787109, -4803.3642578, 8022.8759766
4: -7986.7231445, 5615.7910156, -2529.2170410, 1727.8784180, -9464.1181641, 7592.6181641

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2478499, upper bound: 3613.1738855
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8958758, upper bound: 3613.4141261
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5991.2548828, 5471.9399414, -1937.9083252, 1687.3262939, -7567.8701172, 6865.5815430
1: -4819.8227539, 5406.7119141, -1562.3759766, 1652.6922607, -6372.0961914, 6507.1889648
2: -7328.0249023, 5808.5625000, -2326.2473145, 1786.0650635, -8862.4091797, 7588.9277344
3: -2685.6877441, 7382.4121094, -874.3999023, 2344.0041504, -4831.3969727, 8043.4770508
4: -8000.6591797, 5626.1665039, -2555.8173828, 1739.5623779, -9493.0439453, 7628.2973633

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.2478499, upper bound: 3613.1983966
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8958758, upper bound: 3613.4141261
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4660573
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4661560
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6010.3779297, 5491.4736328, -2014.9360352, 1751.3082275, -7648.0800781, 6957.5966797
1: -4835.5356445, 5424.8867188, -1623.1793213, 1719.2806396, -6447.0937500, 6583.1645508
2: -7350.8974609, 5829.1943359, -2412.6357422, 1859.3305664, -8948.9941406, 7690.9785156
3: -2694.6108398, 7406.2529297, -910.2250977, 2432.6159668, -4927.1088867, 8099.8403320
4: -8025.1254883, 5646.1225586, -2653.2941895, 1812.6442871, -9579.6074219, 7741.0605469

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4669226
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4668181
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6002.6494141, 5485.6284180, -1915.7650146, 1658.8310547, -7559.7050781, 6854.3452148
1: -4829.3398438, 5418.9902344, -1544.3862305, 1629.0655518, -6360.8691406, 6499.5126953
2: -7341.7680664, 5822.5727539, -2298.7004395, 1761.9990234, -8855.5019531, 7571.9643555
3: -2691.4187012, 7397.4003906, -864.4005737, 2312.7797852, -4806.3173828, 8049.8823242
4: -8015.0239258, 5639.4902344, -2526.4140625, 1717.6270752, -9488.5839844, 7609.1191406

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4669226
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4668181
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6200.3579102, 5668.2075195, -3888.2846680, 3524.9892578, -9303.5849609, 8894.4882812
1: -4990.1225586, 5599.1586914, -3133.1621094, 3475.1997070, -8106.0537109, 8166.3940430
2: -7583.6416016, 6017.1235352, -4743.5263672, 3732.1315918, -10741.6992188, 10021.6005859
3: -2778.9755859, 7645.5649414, -1740.8699951, 4789.3388672, -7230.6992188, 9042.2089844
4: -8278.3486328, 5828.1064453, -5187.5239258, 3625.0412598, -11318.8076172, 10272.3212891

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6303736, upper bound: 3612.2880265
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1640631, upper bound: 3612.2880322
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6200.4106445, 5668.2495117, -3779.1152344, 3427.9399414, -9209.9150391, 8781.6943359
1: -4990.1635742, 5599.2011719, -3046.1433105, 3381.1396484, -8014.9003906, 8076.0180664
2: -7583.7001953, 6017.1689453, -4618.2841797, 3631.6606445, -10644.6025391, 9890.0107422
3: -2778.9992676, 7645.6254883, -1691.0637207, 4659.3134766, -7098.0722656, 8993.8828125
4: -8278.4130859, 5828.1499023, -5047.7875977, 3526.0205078, -11224.2500000, 10125.7744141

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6303736, upper bound: 3612.3014622
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1640631, upper bound: 3612.3014674
time: 0.97 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.10 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.1171765, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2965021, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991833
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4497980, upper bound: 3611.8991838
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4497980, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.2832582
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.4669225, upper bound: 3613.4906431
IS_A2_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977271
IS_A2_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977384
IS_A2_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977271
IS_A2_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.2977384
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4660573
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4661560
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4669226
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4668181
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4669226
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.2832336, upper bound: 3613.4668181
IS_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3612.6303736, upper bound: 3612.2880265
IS_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.1640631, upper bound: 3612.2880322
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3612.6303736, upper bound: 3612.3014622
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.10
Output dim: 0, lower bound: -3613.1640631, upper bound: 3612.3014674

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1716.4951172, 1490.1773682, -5955.0258789, 5446.5776367, -6618.6757812, 7335.6054688
1: -1383.7639160, 1459.6079102, -4790.8828125, 5380.0297852, -6303.1982422, 6149.1596680
2: -2064.3198242, 1578.0806885, -7284.8398438, 5780.0605469, -7299.5815430, 8610.0986328
3: -771.3589478, 2073.4702148, -2671.0954590, 7341.7187500, -7897.8574219, 4549.4951172
4: -2268.2358398, 1536.5035400, -7952.3037109, 5597.3242188, -7313.7202148, 9237.1562500

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.2775200
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.4809622
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1685.1627197, 1460.3262939, -5953.3740234, 5445.3374023, -6587.5317383, 7314.5048828
1: -1359.2292480, 1430.2169189, -4789.5126953, 5378.7763672, -6278.1870117, 6128.1669922
2: -2029.3806152, 1546.2307129, -7282.8198242, 5778.5493164, -7264.5708008, 8588.0673828
3: -756.2575073, 2035.6834717, -2670.2968750, 7339.6943359, -7885.3681641, 4513.2172852
4: -2228.4836426, 1504.5333252, -7949.9892578, 5595.6679688, -7273.7080078, 9216.0429688

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.2775200
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.4809622
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1738.0455322, 1514.7304688, -5967.5585938, 5458.8291016, -6651.0771484, 7366.5014648
1: -1401.9702148, 1482.1727295, -4801.1230469, 5391.9804688, -6331.6562500, 6177.6547852
2: -2091.4250488, 1597.7427979, -7300.3393555, 5792.9794922, -7337.2177734, 8644.4335938
3: -780.3956299, 2102.7360840, -2677.0292969, 7358.0976562, -7922.4472656, 4582.1562500
4: -2296.6911621, 1556.6961670, -7969.0966797, 5609.9472656, -7353.2709961, 9272.0283203

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.2832336
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.4855005
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1706.8536377, 1484.5152588, -5965.3569336, 5457.0625000, -6619.4658203, 7343.9389648
1: -1377.6455078, 1453.4088135, -4799.2939453, 5390.1835938, -6306.3217773, 6155.2480469
2: -2056.8505859, 1566.4759521, -7297.6166992, 5790.8945312, -7301.9902344, 8620.6269531
3: -765.0651245, 2065.1086426, -2676.0239258, 7355.3203125, -7908.5952148, 4545.7729492
4: -2257.4387207, 1525.4367676, -7966.0078125, 5607.7548828, -7313.1254883, 9249.0078125

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.2832336
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.4855005
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5977.5903320, 5458.2900391, -6872.3979492, 7566.0253906
1: -1577.3613281, 1670.7596436, -4808.7016602, 5393.1787109, -6509.9702148, 6371.5698242
2: -2346.6181641, 1806.2084961, -7311.2553711, 5794.1196289, -7596.1572266, 8856.2929688
3: -884.1437378, 2366.0300293, -2679.1831055, 7364.7153320, -8033.0654297, 4847.2978516
4: -2580.5910645, 1761.1538086, -7982.4467773, 5612.1416016, -7639.4912109, 9484.4785156

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1738855, upper bound: 3611.2478499
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4384829, upper bound: 3611.8942794
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4496325, upper bound: 3611.8593426
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4497322, upper bound: 3611.8990344
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5969.7143555, 5452.4165039, -6767.0903320, 7475.9111328
1: -1496.7210693, 1578.8179932, -4802.3940430, 5387.2685547, -6424.4760742, 6283.8500977
2: -2230.0136719, 1707.1827393, -7301.9619141, 5787.4619141, -7474.5087891, 8760.9833984
3: -836.8752441, 2243.5935059, -2675.9321289, 7355.7734375, -7981.8828125, 4723.9169922
4: -2450.8525391, 1663.9034424, -7972.1728516, 5605.4692383, -7504.7841797, 9391.2480469

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1738855, upper bound: 3611.2478498
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4384829, upper bound: 3611.8960876
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4370191, upper bound: 3611.8694323
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4370631, upper bound: 3611.8965018
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -6000.3911133, 5482.6455078, -6893.0097656, 7590.5190430
1: -1577.3613281, 1670.7596436, -4827.4335938, 5416.0776367, -6529.8867188, 6391.5141602
2: -2346.6181641, 1806.2084961, -7338.7255859, 5819.6230469, -7617.2729492, 8885.3056641
3: -884.1437378, 2366.0300293, -2690.2050781, 7394.0156250, -8062.2929688, 4857.0034180
4: -2580.5910645, 1761.1538086, -8011.8242188, 5636.7177734, -7660.7646484, 9516.2939453

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.2824545
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.4898848
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5992.5249023, 5476.7583008, -6787.7031250, 7500.4204102
1: -1496.7210693, 1578.8179932, -4821.1391602, 5410.1538086, -6444.3881836, 6303.8144531
2: -2230.0136719, 1707.1827393, -7329.4467773, 5812.9594727, -7495.6328125, 8790.0195312
3: -836.8752441, 2243.5935059, -2686.9714355, 7385.0649414, -8011.1196289, 4733.6455078
4: -2450.8525391, 1663.9034424, -8001.5532227, 5630.0390625, -7526.0664062, 9423.0781250

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.2824545
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.4898848
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -5756.1220703, 5268.5576172, -6680.2915039, 7347.8789062
1: -1593.0754395, 1682.9068604, -4631.3994141, 5204.7973633, -6317.5522461, 6198.5273438
2: -2369.6191406, 1820.1040039, -7055.4423828, 5592.3125000, -7393.5937500, 8597.6328125
3: -891.3933716, 2388.5690918, -2578.4084473, 7100.1044922, -7764.0747070, 4759.7309570
4: -2604.9167480, 1773.1571045, -7698.4750977, 5413.6435547, -7445.1420898, 9195.8779297

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2955083, upper bound: 3612.7357783
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3115036, upper bound: 3613.1519526
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5748.2744141, 5262.6499023, -6578.6743164, 7260.9248047
1: -1515.4637451, 1596.6520996, -4625.0556641, 5198.8583984, -6235.0795898, 6114.4936523
2: -2257.5788574, 1724.0441895, -7046.1235352, 5585.6210938, -7276.5000000, 8505.8349609
3: -845.8417969, 2271.2236328, -2575.1853027, 7091.0488281, -7714.1684570, 4641.4653320
4: -2480.0874023, 1678.2492676, -7688.1855469, 5406.9291992, -7315.2392578, 9105.7050781

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2955083, upper bound: 3612.7357783
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3115036, upper bound: 3613.1519526
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -6006.1860352, 5481.2519531, -6913.6772461, 7610.7744141
1: -1593.0754395, 1682.9068604, -4832.0576172, 5415.1196289, -6546.7402344, 6409.1005859
2: -2369.6191406, 1820.1040039, -7343.9453125, 5817.7915039, -7642.4902344, 8905.7343750
3: -891.3933716, 2388.5690918, -2691.3601074, 7397.5512695, -8072.2392578, 4882.5161133
4: -2604.9167480, 1773.1571045, -8018.7070312, 5635.8422852, -7686.9672852, 9537.4062500

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4631895, upper bound: 3611.8996825
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4631896, upper bound: 3613.4906431
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5998.1401367, 5475.1757812, -6811.9277344, 7523.6191406
1: -1515.4637451, 1596.6520996, -4825.6005859, 5408.9926758, -6464.1040039, 6324.9545898
2: -2257.5788574, 1724.0441895, -7334.4257812, 5810.9130859, -7525.2353516, 8813.7363281
3: -845.8417969, 2271.2236328, -2688.0505371, 7388.3808594, -8022.1855469, 4764.1811523
4: -2480.0874023, 1678.2492676, -8008.1694336, 5628.9584961, -7556.9116211, 9446.9853516

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4631895, upper bound: 3611.8996825
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4631896, upper bound: 3613.4906431
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5913.9482422, 5404.0043945, -1912.1146240, 1662.5505371, -7466.1791992, 6766.7768555
1: -4758.1596680, 5339.8686523, -1540.8786621, 1634.0223389, -6287.3300781, 6414.7050781
2: -7237.4980469, 5735.2343750, -2294.6564941, 1766.7187500, -8744.7509766, 7479.1860352
3: -2648.6115723, 7290.1840820, -864.1401367, 2312.5270996, -4761.6064453, 7937.3935547
4: -7901.2866211, 5555.4306641, -2522.1616211, 1722.7070312, -9366.7421875, 7518.8920898

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
time: 8.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5950.4916992, 5436.6914062, -1917.5677490, 1667.5399170, -7510.4179688, 6808.5839844
1: -4787.0927734, 5372.0605469, -1545.2342529, 1638.9377441, -6323.2553711, 6454.3764648
2: -7280.0014648, 5771.1005859, -2301.1032715, 1772.0444336, -8796.5205078, 7525.3554688
3: -2666.9323730, 7333.0976562, -866.7033081, 2319.1787109, -4788.7270508, 7985.4296875
4: -7947.9921875, 5589.6826172, -2529.2170410, 1727.8784180, -9422.8154297, 7563.9648438

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5924.2119141, 5413.8588867, -1932.4864502, 1682.3754883, -7491.7314453, 6795.8105469
1: -4766.5610352, 5349.5219727, -1558.0471191, 1647.8212891, -6310.6479492, 6440.1777344
2: -7250.1376953, 5745.6821289, -2319.8364258, 1780.8179932, -8773.0380859, 7513.0073242
3: -2653.4245605, 7303.5087891, -871.8530273, 2337.3813477, -4789.5737305, 7957.7812500
4: -7914.9580078, 5565.5917969, -2548.8132324, 1734.4686279, -9395.4316406, 7554.4331055

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8602036, upper bound: 3613.4660572
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8602036, upper bound: 3613.4660572
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5960.9770508, 5446.7885742, -1937.9083252, 1687.3262939, -7536.1635742, 6837.7944336
1: -4795.6791992, 5381.9643555, -1562.3759766, 1652.6922607, -6346.7270508, 6480.0449219
2: -7292.9106445, 5781.8198242, -2326.2473145, 1786.0650635, -8825.0517578, 7559.3754883
3: -2671.8608398, 7346.7094727, -874.3999023, 2344.0041504, -4816.7685547, 8006.0854492
4: -7961.9560547, 5600.1088867, -2555.8173828, 1739.5623779, -9451.7695312, 7599.6767578

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8998948, upper bound: 3613.4661559
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.8998948, upper bound: 3613.4661559
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5727.5791016, 5255.8701172, -2014.9360352, 1751.3082275, -7349.1289062, 6690.0683594
1: -4609.5380859, 5192.2236328, -1623.1793213, 1719.2806396, -6207.4580078, 6321.3427734
2: -7024.5039062, 5578.4096680, -2412.6357422, 1859.3305664, -8597.1513672, 7406.5195312
3: -2567.7092285, 7074.6943359, -910.2250977, 2432.6159668, -4788.6083984, 7749.5668945
4: -7662.6064453, 5400.6713867, -2653.2941895, 1812.6442871, -9189.5625000, 7463.8237305

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4504971
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4669226
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5975.3598633, 5463.7597656, -2014.9360352, 1751.3082275, -7610.0566406, 6920.7846680
1: -4807.2456055, 5397.7387695, -1623.1793213, 1719.2806396, -6415.7016602, 6547.8129883
2: -7310.1538086, 5799.6406250, -2412.6357422, 1859.3305664, -8902.8281250, 7652.7373047
3: -2679.4545898, 7366.1777344, -910.2250977, 2432.6159668, -4910.2182617, 8053.6669922
4: -7979.8789062, 5616.8422852, -2653.2941895, 1812.6442871, -9528.6240234, 7702.7055664

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4503174
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4668181
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5720.0283203, 5250.1655273, -1915.7650146, 1658.8310547, -7260.9394531, 6586.9267578
1: -4603.4379883, 5186.4892578, -1544.3862305, 1629.0655518, -6121.3364258, 6237.8344727
2: -7015.5551758, 5571.9365234, -2298.7004395, 1761.9990234, -8503.8476562, 7287.6372070
3: -2564.5854492, 7065.9848633, -864.4005737, 2312.7797852, -4667.8793945, 7699.7739258
4: -7652.7338867, 5394.1655273, -2526.4140625, 1717.6270752, -9098.7822266, 7332.0053711

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4504971
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4669226
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5967.5493164, 5457.8969727, -1915.7650146, 1658.8310547, -7521.6054688, 6817.5146484
1: -4800.9780273, 5391.8291016, -1544.3862305, 1629.0655518, -6329.4067383, 6464.1459961
2: -7300.9184570, 5792.9882812, -2298.7004395, 1761.9990234, -8809.2353516, 7533.6879883
3: -2676.2282715, 7357.2299805, -864.4005737, 2312.7797852, -4789.3881836, 8003.6088867
4: -7969.6708984, 5610.1757812, -2526.4140625, 1717.6270752, -9437.4990234, 7570.7294922

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4503174
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4668181
time: 0.96 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.32 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.2775200
IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.2775200
IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.1238967, upper bound: 3613.4809622
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.2832336
IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.2832336
IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2972007, upper bound: 3613.4855005
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4496325, upper bound: 3611.8593426
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4497322, upper bound: 3611.8990344
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4370191, upper bound: 3611.8694323
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4370631, upper bound: 3611.8965018
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.2824545
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4504971, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2955083, upper bound: 3612.7357783
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.3115036, upper bound: 3613.1519526
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2955083, upper bound: 3612.7357783
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.3115036, upper bound: 3613.1519526
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4631895, upper bound: 3611.8996825
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4631896, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4631895, upper bound: 3611.8996825
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.4631896, upper bound: 3613.4906431
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
IS_A2_B1_A2_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8593426, upper bound: 3613.4496325
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
IS_A2_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8990344, upper bound: 3613.4497322
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8602036, upper bound: 3613.4660572
IS_A2_B1_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8602036, upper bound: 3613.4660572
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8998948, upper bound: 3613.4661559
IS_A2_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3611.8998948, upper bound: 3613.4661559
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4504971
IS_A2_B1_A2_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4669226
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4503174
IS_A2_B1_A2_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2800674, upper bound: 3613.4668181
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4504971
IS_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4669226
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4503174
IS_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.32
Output dim: 0, lower bound: -3613.2775200, upper bound: 3613.4668181

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1716.4951172, 1490.1773682, -5934.0341797, 5429.8041992, -6594.7407227, 7312.5419922
1: -1383.7639160, 1459.6079102, -4773.8408203, 5363.5356445, -6280.2724609, 6129.7841797
2: -2064.3198242, 1578.0806885, -7260.2050781, 5762.1103516, -7275.0170898, 8581.4658203
3: -771.3589478, 2073.4702148, -2662.1042480, 7317.3823242, -7868.6098633, 4539.5454102
4: -2268.2358398, 1536.5035400, -7924.9853516, 5579.3828125, -7288.6157227, 9205.6376953

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7305101, upper bound: 3613.4423956
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8827042, upper bound: 3613.4723822
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1685.1627197, 1460.3262939, -5932.3183594, 5428.5053711, -6563.5488281, 7291.3798828
1: -1359.2292480, 1430.2169189, -4772.4208984, 5362.2255859, -6255.2119141, 6108.7451172
2: -2029.3806152, 1546.2307129, -7258.1152344, 5760.5341797, -7239.9511719, 8559.3671875
3: -756.2575073, 2035.6834717, -2661.2766113, 7315.2607422, -7856.0346680, 4503.2382812
4: -2228.4836426, 1504.5333252, -7922.5883789, 5577.6596680, -7248.5473633, 9184.4472656

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.7306066, upper bound: 3613.4128649
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9702635, upper bound: 3613.2690256
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1238515, upper bound: 3613.4809505
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1738.0455322, 1514.7304688, -5946.6386719, 5442.0913086, -6627.1645508, 7343.5034180
1: -1401.9702148, 1482.1727295, -4784.1386719, 5375.5126953, -6308.7470703, 6158.3305664
2: -2091.4250488, 1597.7427979, -7275.7993164, 5775.0883789, -7312.7031250, 8615.8808594
3: -780.3956299, 2102.7360840, -2668.1354980, 7333.8457031, -7893.2758789, 4572.2963867
4: -2296.6911621, 1556.6961670, -7941.8803711, 5592.0629883, -7328.2226562, 9240.6074219

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8018350, upper bound: 3613.4292652
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9193476, upper bound: 3613.4755646
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1706.8536377, 1484.5152588, -5944.3105469, 5440.2109375, -6595.4565430, 7320.8183594
1: -1377.6455078, 1453.4088135, -4782.2080078, 5373.6083984, -6283.3212891, 6135.8256836
2: -2056.8505859, 1566.4759521, -7272.9228516, 5772.8725586, -7277.3652344, 8591.9287109
3: -765.0651245, 2065.1086426, -2667.0371094, 7330.9125977, -7879.2729492, 4535.8344727
4: -2257.4387207, 1525.4367676, -7938.6235352, 5589.7373047, -7287.9653320, 9217.4199219

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8406396, upper bound: 3613.4126597
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.5305887, upper bound: 3613.4122254
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1952.6938477, 1696.4223633, -5910.7753906, 5400.4135742, -6803.0864258, 7490.4257812
1: -1573.2641602, 1666.1597900, -4755.6328125, 5336.1943359, -6443.3828125, 6310.6723633
2: -2340.5456543, 1801.2883301, -7233.6494141, 5731.4633789, -7520.7841797, 8767.5644531
3: -881.7428589, 2359.7683105, -2647.0268555, 7286.0571289, -7947.7788086, 4805.9545898
4: -2573.9462891, 1756.3043213, -7897.0507812, 5551.7856445, -7566.1865234, 9387.4960938

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7470844, upper bound: 3611.6092108
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3666523, upper bound: 3611.8531498
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5947.4033203, 5433.2177734, -6844.6616211, 7534.4096680
1: -1577.3613281, 1670.7596436, -4784.6308594, 5368.5175781, -6482.9111328, 6346.2749023
2: -2346.6181641, 1806.2084961, -7276.2456055, 5767.4648438, -7566.6904297, 8819.0400391
3: -884.1437378, 2366.0300293, -2665.3947754, 7329.0922852, -7995.7622070, 4832.7050781
4: -2580.5910645, 1761.1538086, -7943.8608398, 5586.1728516, -7610.9560547, 9443.3242188

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7471530, upper bound: 3611.6285789
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3667526, upper bound: 3611.8921847
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1853.3447266, 1605.8117676, -5989.6279297, 5476.4028320, -6777.6967773, 7491.1835938
1: -1494.2474365, 1576.3221436, -4818.7080078, 5410.5375977, -6436.5639648, 6295.4018555
2: -2226.4201660, 1704.4774170, -7327.7919922, 5813.2080078, -7485.6162109, 8779.6416016
3: -835.5184937, 2239.9782715, -2685.9082031, 7382.0439453, -8003.3911133, 4727.1293945
4: -2446.9025879, 1661.2800293, -7999.6787109, 5630.1064453, -7514.5732422, 9412.1074219

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8042700, upper bound: 3611.3637845
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3527425, upper bound: 3611.8610845
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5965.0922852, 5448.6425781, -6761.8637695, 7470.7275391
1: -1496.7210693, 1578.8179932, -4798.6665039, 5383.5756836, -6419.5429688, 6279.6728516
2: -2230.0136719, 1707.1827393, -7296.6230469, 5783.4907227, -7468.9877930, 8754.7548828
3: -836.8752441, 2243.5935059, -2673.9440918, 7350.3764648, -7975.8647461, 4721.2832031
4: -2450.8525391, 1663.9034424, -7966.2592773, 5601.5527344, -7499.3471680, 9384.4140625

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8042680, upper bound: 3611.6533808
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3527835, upper bound: 3611.8893898
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5717.8823242, 5247.2690430, -6625.6542969, 7291.8554688
1: -1577.3613281, 1670.7596436, -4601.6425781, 5183.6625977, -6268.2934570, 6152.0839844
2: -2346.6181641, 1806.2084961, -7012.6723633, 5569.0751953, -7333.0317383, 8533.7988281
3: -884.1437378, 2366.0300293, -2563.3923340, 7062.7441406, -7712.3066406, 4718.5776367
4: -2580.5910645, 1761.1538086, -7649.6977539, 5391.4755859, -7383.7309570, 9126.6386719

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1563237, upper bound: 3613.1541787
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1750046, upper bound: 3613.1536607
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5965.3486328, 5454.9526367, -6856.2202148, 7552.4750977
1: -1577.3613281, 1670.7596436, -4799.1225586, 5388.9541016, -6494.5595703, 6360.1044922
2: -2346.6181641, 1806.2084961, -7297.9443359, 5790.0761719, -7579.0356445, 8839.1103516
3: -884.1437378, 2366.0300293, -2675.0190430, 7353.9179688, -8016.0937500, 4840.0771484
4: -2580.5910645, 1761.1538086, -7966.5454102, 5607.4443359, -7622.4189453, 9465.2832031

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1563237, upper bound: 3613.1635117
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1750046, upper bound: 3613.1629936
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5710.1542969, 5241.4990234, -6520.4340820, 7201.9067383
1: -1496.7210693, 1578.8179932, -4595.4047852, 5177.8745117, -6182.9160156, 6064.4497070
2: -2230.0136719, 1707.1827393, -7003.5209961, 5562.5312500, -7211.4965820, 8438.6582031
3: -836.8752441, 2243.5935059, -2560.2165527, 7053.8647461, -7661.2426758, 4595.2705078
4: -2450.8525391, 1663.9034424, -7639.6015625, 5384.8945312, -7249.1279297, 9033.6162109

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8172449, upper bound: 3613.1509149
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3675154, upper bound: 3613.1515631
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5957.3862305, 5449.0463867, -6750.8901367, 7462.2866211
1: -1496.7210693, 1578.8179932, -4792.7426758, 5383.0126953, -6409.0400391, 6272.3173828
2: -2230.0136719, 1707.1827393, -7288.5410156, 5783.3735352, -7457.3530273, 8743.7089844
3: -836.8752441, 2243.5935059, -2671.7497559, 7344.8398438, -7964.7934570, 4716.6782227
4: -2450.8525391, 1663.9034424, -7956.1503906, 5600.7260742, -7487.6777344, 9371.9511719

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8172449, upper bound: 3613.4441959
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3675154, upper bound: 3613.4747626
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -5956.0922852, 5442.9711914, -6866.5488281, 7556.0336914
1: -1593.0754395, 1682.9068604, -4791.5078125, 5378.0776367, -6501.9423828, 6364.8398438
2: -2369.6191406, 1820.1040039, -7286.9111328, 5777.7050781, -7593.2612305, 8841.4902344
3: -891.3933716, 2388.5690918, -2670.2763672, 7341.5400391, -8010.9887695, 4857.9057617
4: -2604.9167480, 1773.1571045, -7955.0859375, 5595.7900391, -7638.1572266, 9466.0488281

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5605397, upper bound: 3611.7041403
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4580211, upper bound: 3611.8963910
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1976.4196777, 1719.2852783, -5976.8090820, 5465.9121094, -6884.5244141, 7578.0019531
1: -1593.0754395, 1682.9068604, -4808.4926758, 5399.6767578, -6519.4877930, 6382.7202148
2: -2369.6191406, 1820.1040039, -7312.0839844, 5801.6757812, -7611.6284180, 8867.5605469
3: -891.3933716, 2388.5690918, -2680.3535156, 7368.7719727, -8037.4965820, 4866.2290039
4: -2604.9167480, 1773.1571045, -7981.8422852, 5618.7001953, -7656.6557617, 9494.5507812

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5605366, upper bound: 3613.2430406
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4580211, upper bound: 3613.3839494
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5948.0590820, 5436.9321289, -6764.8139648, 7468.8857422
1: -1515.4637451, 1596.6520996, -4785.0502930, 5371.9931641, -6419.3344727, 6280.6826172
2: -2257.5788574, 1724.0441895, -7277.3920898, 5770.8603516, -7476.0219727, 8749.4843750
3: -845.8417969, 2271.2236328, -2666.9609375, 7332.3115234, -7960.8750000, 4739.5532227
4: -2480.0874023, 1678.2492676, -7944.5678711, 5588.9404297, -7508.1201172, 9375.6347656

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7591363, upper bound: 3611.5865555
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3108016, upper bound: 3611.8926007
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1878.8485107, 1630.7659912, -5968.7485352, 5459.8593750, -6782.7924805, 7490.8325195
1: -1515.4637451, 1596.6520996, -4802.0151367, 5393.5810547, -6436.8745117, 6298.5468750
2: -2257.5788574, 1724.0441895, -7302.5273438, 5794.8144531, -7494.3813477, 8775.5253906
3: -845.8417969, 2271.2236328, -2677.0344238, 7359.5146484, -7987.3613281, 4747.8720703
4: -2480.0874023, 1678.2492676, -7971.2846680, 5611.8398438, -7526.6127930, 9404.1044922

Time for backsubstitution: 1.88 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=4653.345703125
rel_dist={0: [-3614.7615518272232, 3614.7615518272232]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.1319917, upper bound: 3613.7564481
time: 0.84 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7564402, upper bound: 3613.7564413
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -3614.1319917, upper bound: 3613.7564481
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -3613.7564402, upper bound: 3613.7564413

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2502.3757324, 2150.9697266, -4423.8598633, 4467.8574219
1: -1831.3944092, 1927.9595947, -2015.7012939, 2107.5605469, -3938.9548340, 3943.6608887
2: -2712.4650879, 2088.7180176, -2978.1022949, 2285.2739258, -4997.7377930, 5066.8188477
3: -1026.7778320, 2724.7028809, -1126.3112793, 2983.9003906, -4010.6782227, 3851.0141602
4: -2982.6787109, 2035.6074219, -3276.4265137, 2226.9489746, -5209.6279297, 5312.0336914

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201061, upper bound: 3613.7201061
time: 0.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201061, upper bound: 3613.7564413
time: 1.11 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4121.0024414, 3710.1943359, -2474.4423828, 2130.8649902, -6251.8671875, 6013.6752930
1: -3322.0283203, 3652.6914062, -1993.7117920, 2087.9550781, -5409.9833984, 5504.3183594
2: -5010.7353516, 3926.7858887, -2946.6174316, 2263.9978027, -7245.2080078, 6715.7416992
3: -1844.0239258, 5050.9423828, -1115.5541992, 2954.6892090, -4757.5468750, 6111.8530273
4: -5482.2490234, 3814.8386230, -3241.3146973, 2206.2404785, -7670.3486328, 6890.0903320

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907409, upper bound: 3613.0294508
time: 0.79 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907475, upper bound: 3613.4907493
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 0, lower bound: -3613.7201061, upper bound: 3613.7201061
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 0, lower bound: -3613.7201061, upper bound: 3613.7564413
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 0, lower bound: -3613.4907409, upper bound: 3613.0294508
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 0, lower bound: -3613.4907475, upper bound: 3613.4907493

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2272.8901367, 1965.4815674, -4238.3715820, 4238.3715820
1: -1831.3944092, 1927.9595947, -1831.3944092, 1927.9595947, -3759.3540039, 3759.3540039
2: -2712.4650879, 2088.7180176, -2712.4650879, 2088.7180176, -4801.1821289, 4801.1816406
3: -1026.7778320, 2724.7028809, -1026.7778320, 2724.7028809, -3751.4807129, 3751.4807129
4: -2982.6787109, 2035.6074219, -2982.6787109, 2035.6074219, -5018.2861328, 5018.2861328

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4931978, upper bound: 3612.9657265
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657180, upper bound: 3612.9657265
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -4119.1708984, 3708.2338867, -5812.5273438, 6084.6523438
1: -1831.3944092, 1927.9595947, -3320.5634766, 3650.7246094, -5342.0312500, 5248.5229492
2: -2712.4650879, 2088.7180176, -5008.5136719, 3924.6728516, -6483.7221680, 7073.5854492
3: -1026.7778320, 2724.7028809, -1843.0756836, 5048.5166016, -6022.8037109, 4529.1845703
4: -2982.6787109, 2035.6074219, -5479.8066406, 3812.7922363, -6633.7426758, 7503.1713867

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4931978, upper bound: 3613.4907587
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657180, upper bound: 3613.4907587
time: 0.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4119.6625977, 3708.7607422, -2227.8205566, 1930.2016602, -6049.8642578, 5768.1074219
1: -3320.9572754, 3651.2529297, -1795.3686523, 1893.3834229, -5214.3398438, 5306.6762695
2: -5009.1118164, 3925.2412109, -2660.6184082, 2051.4250488, -7036.5341797, 6432.9238281
3: -1843.3304443, 5049.1679688, -1007.6977539, 2675.2614746, -4480.0561523, 6004.6416016
4: -5480.4633789, 3813.3420410, -2925.3925781, 1998.4335938, -7466.3950195, 6577.4306641

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0294111, upper bound: 3613.0294115
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0294111, upper bound: 3613.0294508
time: 0.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4112.9409180, 3704.4726562, -4177.0981445, 3722.7863770, -7615.3710938, 7641.5454102
1: -3315.5607910, 3647.3198242, -3366.3637695, 3663.9316406, -6790.2021484, 6811.3505859
2: -5001.4960938, 3920.9116211, -5063.9184570, 3940.6691895, -8667.7919922, 8703.1025391
3: -1840.7945557, 5042.1601562, -1859.7589111, 5081.9282227, -6791.4682617, 6752.0151367
4: -5472.0249023, 3808.9978027, -5538.6865234, 3828.4196777, -9026.0839844, 9062.9003906

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0294504, upper bound: 3613.4907449
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0294504, upper bound: 3613.4907515
time: 0.92 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.60 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.4931978, upper bound: 3612.9657265
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.60
Output dim: 0, lower bound: -3612.9657180, upper bound: 3612.9657265
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.4931978, upper bound: 3613.4907587
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -3612.9657180, upper bound: 3613.4907587
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.0294111, upper bound: 3613.0294115
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.0294111, upper bound: 3613.0294508
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.0294504, upper bound: 3613.4907449
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.60
Output dim: 0, lower bound: -3613.0294504, upper bound: 3613.4907515

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -2272.8901367, 1965.4815674, -3999.3693848, 4041.9062500
1: -1638.3045654, 1737.5650635, -1831.3944092, 1927.9595947, -3566.2641602, 3568.9594727
2: -2435.9470215, 1879.3935547, -2712.4650879, 2088.7180176, -4524.6650391, 4591.8583984
3: -920.3916626, 2455.5791016, -1026.7778320, 2724.7028809, -3645.0944824, 3482.3569336
4: -2677.5119629, 1832.2690430, -2982.6787109, 2035.6074219, -4713.1181641, 4814.9477539

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657265, upper bound: 3612.9657265
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657265, upper bound: 3612.9657265
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -4117.7084961, 3706.6694336, -5575.4663086, 5886.7255859
1: -1638.3045654, 1737.5650635, -3319.3942871, 3649.1538086, -5150.2436523, 5056.9584961
2: -2435.9470215, 1879.3935547, -5006.7392578, 3922.9868164, -6211.2875977, 6868.8476562
3: -920.3916626, 2455.5791016, -1842.3189697, 5046.5795898, -5917.3168945, 4262.7231445
4: -2677.5119629, 1832.2690430, -5477.8569336, 3811.1589355, -6332.6523438, 7304.4565430

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.0294596
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.4907547
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -4112.9409180, 3704.4726562, -7442.9116211, 7451.8837891
1: -3211.7971191, 3524.2065430, -3315.5607910, 3647.3198242, -6649.6279297, 6636.9052734
2: -4848.2338867, 3788.9609375, -5001.4960938, 3920.9116211, -8472.8906250, 8499.5556641
3: -1780.8483887, 4870.8652344, -1840.7945557, 5042.1601562, -6665.4018555, 6571.0488281
4: -5298.1914062, 3677.4733887, -5472.0249023, 3808.9978027, -8806.3837891, 8859.1699219

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.0294596
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.4907547
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3904.8264160, 3541.1750488, -4177.0981445, 3722.7863770, -7395.2783203, 7459.6640625
1: -3146.5612793, 3492.1206055, -3366.3637695, 3663.9316406, -6611.7338867, 6639.0961914
2: -4764.1665039, 3751.3303223, -5063.9184570, 3940.6691895, -8412.5576172, 8512.6611328
3: -1749.9495850, 4810.4135742, -1859.7589111, 5081.9282227, -6690.6254883, 6508.5629883
4: -5208.8427734, 3642.6633301, -5538.6865234, 3828.4196777, -8742.7763672, 8876.8623047

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0294113, upper bound: 3612.9657107
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0294115, upper bound: 3613.1092122
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6220.8808594, 5677.8222656, -4177.0981445, 3722.7863770, -9578.0244141, 9229.5371094
1: -5007.0087891, 5609.3833008, -3366.3637695, 3663.9316406, -8360.7695312, 8444.6308594
2: -7606.2534180, 6028.3188477, -5063.9184570, 3940.6691895, -11032.0947266, 10408.6240234
3: -2787.3400879, 7664.9511719, -1859.7589111, 5081.9282227, -7578.7197266, 9204.0458984
4: -8303.3789062, 5839.3857422, -5538.6865234, 3828.4196777, -11610.1660156, 10691.6835938

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0294113, upper bound: 3612.9657107
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0294115, upper bound: 3613.4896201
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657265, upper bound: 3612.9657265
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657265, upper bound: 3612.9657265
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.0294596
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.4907547
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.0294596
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3612.9657107, upper bound: 3613.4907547
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.0294113, upper bound: 3612.9657107
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.0294115, upper bound: 3613.1092122
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.0294113, upper bound: 3612.9657107
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.51
Output dim: 0, lower bound: -3613.0294115, upper bound: 3613.4896201

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -6048.7617188, 5517.2426758, -7022.0698242, 7716.0126953
1: -1638.3045654, 1737.5650635, -4866.9892578, 5450.9921875, -6641.4360352, 6508.6972656
2: -2435.9470215, 1879.3935547, -7395.5346680, 5857.9091797, -7764.9677734, 9030.1162109
3: -920.3916626, 2455.5791016, -2711.1582031, 7448.1323242, -8159.9350586, 4978.1801758
4: -2677.5119629, 1832.2690430, -8074.1787109, 5674.4003906, -7814.7275391, 9666.3076172

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3279278, upper bound: 3613.4906522
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4931893, upper bound: 3613.4907293
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -6219.8725586, 5676.7207031, -9030.0146484, 9413.5576172
1: -3211.7971191, 3524.2065430, -5006.1933594, 5608.2841797, -8281.9882812, 8206.6953125
2: -4848.2338867, 3788.9609375, -7605.0322266, 6027.1381836, -10177.4423828, 10862.7011719
3: -1780.8483887, 4870.8652344, -2786.8093262, 7663.5932617, -9116.1738281, 7357.8437500
4: -5298.1914062, 3677.4733887, -8302.0351562, 5838.2500000, -10434.2519531, 11441.9765625

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9656337, upper bound: 3613.0272119
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9264259, upper bound: 3612.9868361
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -6225.2827148, 5682.6333008, -11132.6572266, 11132.6582031
1: -5010.5698242, 5614.1787109, -5010.5698242, 5614.1787109, -9965.4804688, 9965.4804688
2: -7611.5888672, 6033.4814453, -7611.5888672, 6033.4814453, -12706.6269531, 12706.6269531
3: -2789.6572266, 7670.8759766, -2789.6572266, 7670.8759766, -9975.6748047, 9975.6748047
4: -8309.2480469, 5844.3500977, -8309.2480469, 5844.3500977, -13205.4160156, 13205.4160156

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4537245, upper bound: 3613.2933170
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2940275, upper bound: 3613.2931248
time: 0.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.59 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 0, lower bound: -3613.3279278, upper bound: 3613.4906522
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 0, lower bound: -3613.4931893, upper bound: 3613.4907293
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 0, lower bound: -3612.9656337, upper bound: 3613.0272119
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 0, lower bound: -3612.9264259, upper bound: 3612.9868361
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.59
Output dim: 0, lower bound: -3613.4537245, upper bound: 3613.2933170
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.59
Output dim: 0, lower bound: -3613.2940275, upper bound: 3613.2931248

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -5953.7685547, 5440.2006836, -6698.4287109, 7405.3403320
1: -1442.5327148, 1525.0668945, -4790.6376953, 5374.9150391, -6367.4311523, 6218.2812500
2: -2150.3056641, 1650.0454102, -7284.0781250, 5775.3110352, -7394.4326172, 8686.5712891
3: -806.7396851, 2160.4394531, -2669.7526855, 7337.3676758, -7933.9897461, 4641.0820312
4: -2361.8547363, 1606.7463379, -7950.6562500, 5592.9213867, -7416.0292969, 9314.0644531

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3279278, upper bound: 3613.3977616
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3277066, upper bound: 3613.2939660
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6022.5009766, 5495.8222656, -6952.9257812, 7649.5659180
1: -1604.6556396, 1701.4029541, -4845.7646484, 5429.9980469, -6581.5766602, 6447.6079102
2: -2386.9431152, 1840.0159912, -7364.6484375, 5835.0712891, -7686.9892578, 8955.6640625
3: -900.5989380, 2405.9113770, -2699.6628418, 7417.3393555, -8107.2919922, 4913.4345703
4: -2623.7451172, 1794.3088379, -8040.1069336, 5651.9272461, -7732.0913086, 9589.2578125

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4931893, upper bound: 3613.4738810
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4929412, upper bound: 3613.2940290
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -6209.4194336, 5668.4228516, -11126.4863281, 11119.8837891
1: -5019.8442383, 5621.1367188, -4998.0268555, 5600.1225586, -9956.5683594, 9955.8808594
2: -7624.3247070, 6040.0839844, -7593.2348633, 6018.3378906, -12699.1923828, 12689.6953125
3: -2793.9113770, 7683.4697266, -2782.2307129, 7651.9389648, -9957.7500000, 9979.3564453
4: -8325.0566406, 5851.8598633, -8289.0908203, 5829.7543945, -13200.9921875, 13186.7871094

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4601148, upper bound: 3611.7102933
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4624077, upper bound: 3613.2932895
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.50 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.3279278, upper bound: 3613.3977616
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.3277066, upper bound: 3613.2939660
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.4931893, upper bound: 3613.4738810
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.4929412, upper bound: 3613.2940290
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.4601148, upper bound: 3611.7102933
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.50
Output dim: 0, lower bound: -3613.4624077, upper bound: 3613.2932895

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1772.1903076, 1538.2601318, -5937.2573242, 5423.1367188, -6659.9843750, 7369.6250000
1: -1428.6251221, 1509.6762695, -4776.5483398, 5357.7827148, -6332.9277344, 6185.4584961
2: -2129.9387207, 1633.5253906, -7262.2172852, 5755.7402344, -7350.0517578, 8644.5585938
3: -798.7545166, 2139.2978516, -2661.2167969, 7315.3251953, -7903.0297852, 4608.5615234
4: -2339.5097656, 1590.8367920, -7928.7753906, 5575.1391602, -7371.0581055, 9271.1982422

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8283850, upper bound: 3613.1728474
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6055422, upper bound: 3613.1720104
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1774.2343750, 1539.3814697, -5874.5732422, 5366.0415039, -6610.4824219, 7309.2089844
1: -1430.3736572, 1509.7462158, -4727.2114258, 5301.6108398, -6283.3828125, 6137.9150391
2: -2132.4255371, 1633.5198975, -7191.7089844, 5696.4477539, -7299.3095703, 8573.5058594
3: -798.9464111, 2141.3232422, -2632.2153320, 7240.9516602, -7828.2451172, 4584.9091797
4: -2342.1535645, 1590.4686279, -7848.5576172, 5515.1796875, -7320.6826172, 9190.3408203

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8283685, upper bound: 3613.0448798
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6054510, upper bound: 3613.0448930
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1980.5772705, 1722.4606934, -6026.7255859, 5494.7226562, -6939.7153320, 7641.7988281
1: -1595.8057861, 1691.8012695, -4848.3999023, 5428.4750977, -6570.3447266, 6438.4409180
2: -2373.8984375, 1829.6684570, -7366.9892578, 5832.5200195, -7670.1215820, 8945.7714844
3: -895.3614502, 2392.3549805, -2700.0356445, 7419.0039062, -8104.2294922, 4898.5117188
4: -2609.4040527, 1784.3922119, -8044.8725586, 5650.8232422, -7714.7084961, 9581.2187500

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1951963, upper bound: 3613.4731360
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3887705, upper bound: 3613.4736695
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1967.2442627, 1708.1020508, -5944.8330078, 5423.0170898, -6857.4951172, 7546.4228516
1: -1585.2321777, 1677.5906982, -4783.5063477, 5358.1669922, -6491.8793945, 6359.7182617
2: -2358.4953613, 1814.3497314, -7274.1152344, 5757.6743164, -7582.9404297, 8835.0273438
3: -888.6257324, 2375.7312012, -2662.9113770, 7322.5561523, -7999.3125000, 4846.9096680
4: -2592.4125977, 1769.1459961, -7939.8422852, 5575.5585938, -7626.6298828, 9458.7500000

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1852474, upper bound: 3613.2926877
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3882944, upper bound: 3613.2939574
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6221.5620117, 5677.2485352, -6138.6474609, 5611.6015625, -11042.5166016, 11028.6689453
1: -5006.8872070, 5608.6162109, -4940.9189453, 5544.8579102, -9878.5917969, 9880.0253906
2: -7605.9331055, 6026.5625000, -7512.1791992, 5958.6572266, -12608.8564453, 12584.5009766
3: -2786.9270020, 7665.1157227, -2751.4711914, 7571.4541016, -9863.2548828, 9924.9072266
4: -8304.6171875, 5838.4462891, -8198.9501953, 5770.6440430, -13109.6064453, 13072.1083984

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8985975, upper bound: 3611.7095103
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8985974, upper bound: 3611.7102933
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6212.1191406, 5668.4755859, -6168.9492188, 5643.0415039, -11059.7705078, 11051.9863281
1: -4999.4418945, 5599.8237305, -4965.7646484, 5574.8359375, -9897.1787109, 9897.3203125
2: -7594.8027344, 6016.9907227, -7548.8286133, 5991.7377930, -12625.4462891, 12612.8447266
3: -2782.1948242, 7653.4121094, -2765.6816406, 7610.1870117, -9896.6826172, 9925.7031250
4: -8292.4980469, 5829.3525391, -8238.2490234, 5802.3715820, -13125.1562500, 13104.6259766

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2797678, upper bound: 3613.2931668
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4624057, upper bound: 3613.2932403
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.41 seconds
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3612.8283850, upper bound: 3613.1728474
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3612.6055422, upper bound: 3613.1720104
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3612.8283685, upper bound: 3613.0448798
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3612.6054510, upper bound: 3613.0448930
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.1951963, upper bound: 3613.4731360
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.3887705, upper bound: 3613.4736695
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.1852474, upper bound: 3613.2926877
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.3882944, upper bound: 3613.2939574
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3611.8985975, upper bound: 3611.7095103
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3611.8985974, upper bound: 3611.7102933
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.2797678, upper bound: 3613.2931668
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 0, lower bound: -3613.4624057, upper bound: 3613.2932403

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1909.0864258, 1660.1823730, -6001.5927734, 5474.5229492, -6847.2119141, 7556.1757812
1: -1538.5064697, 1631.6311035, -4828.0629883, 5408.7382812, -6492.4975586, 6358.1704102
2: -2291.2036133, 1764.1735840, -7337.4716797, 5811.0034180, -7565.2968750, 8850.6562500
3: -862.7436523, 2308.9113770, -2689.1767578, 7389.7827148, -8041.9111328, 4804.2402344
4: -2518.3461914, 1720.3592529, -8012.2705078, 5629.5126953, -7601.9101562, 9484.3935547

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1880272, upper bound: 3611.8991035
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1929.6755371, 1679.7036133, -6002.0942383, 5475.0996094, -6867.6425781, 7572.4443359
1: -1555.8516846, 1645.1103516, -4828.7641602, 5409.0952148, -6509.2236328, 6374.0390625
2: -2316.6340332, 1778.3107910, -7338.6347656, 5811.3637695, -7589.7084961, 8868.0283203
3: -870.5614014, 2333.9362793, -2689.0769043, 7390.9248047, -8050.4199219, 4827.5878906
4: -2545.2612305, 1732.1882324, -8013.4482422, 5630.0756836, -7628.6772461, 9501.1845703

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0604694, upper bound: 3612.6131312
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0605778, upper bound: 3613.0358956
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1915.7763672, 1666.3088379, -5918.6186523, 5401.6157227, -6783.1845703, 7475.0087891
1: -1544.8132324, 1631.8199463, -4762.6132812, 5336.9042969, -6428.5205078, 6293.8369141
2: -2300.5524902, 1762.9499512, -7243.9365234, 5734.5644531, -7500.0200195, 8755.0292969
3: -863.5355225, 2316.8605957, -2651.1103516, 7292.5864258, -7943.2070312, 4774.7441406
4: -2527.4992676, 1716.7758789, -7906.4028320, 5552.9238281, -7538.0664062, 9376.2011719

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0525148, upper bound: 3612.7879447
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2463250, upper bound: 3613.0448918
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6182.1967773, 5644.1416016, -6150.9941406, 5628.4082031, -11007.0937500, 10998.8789062
1: -4975.3505859, 5576.0278320, -4951.3085938, 5560.5253906, -9851.0253906, 9849.1865234
2: -7560.2133789, 5991.3784180, -7528.0839844, 5976.2646484, -12564.8027344, 12554.7011719
3: -2768.9711914, 7618.8452148, -2757.7529297, 7589.6079102, -9857.8837891, 9876.0039062
4: -8254.0771484, 5804.0957031, -8215.2509766, 5787.1323242, -13060.4082031, 13043.8144531

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1454028, upper bound: 3612.7868895
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3823848, upper bound: 3613.0442279
time: 0.86 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.95 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.1880272, upper bound: 3611.8991035
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.0604694, upper bound: 3612.6131312
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.0605778, upper bound: 3613.0358956
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.0525148, upper bound: 3612.7879447
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.2463250, upper bound: 3613.0448918
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.1454028, upper bound: 3612.7868895
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.95
Output dim: 0, lower bound: -3613.3823848, upper bound: 3613.0442279

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1909.0864258, 1660.1823730, -5975.3569336, 5460.6699219, -6821.5649414, 7527.5258789
1: -1538.5064697, 1631.6311035, -4807.1557617, 5394.5639648, -6468.2719727, 6335.2016602
2: -2291.2036133, 1764.1735840, -7308.9340820, 5796.2675781, -7537.8520508, 8817.3183594
3: -862.7436523, 2308.9113770, -2679.1284180, 7363.6977539, -8011.0078125, 4789.6762695
4: -2518.3461914, 1720.3592529, -7979.2978516, 5614.0605469, -7575.2636719, 9447.0615234

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6166.7778320, 5630.8593750, -6157.7187500, 5635.7431641, -10996.1718750, 10991.1308594
1: -4962.9897461, 5562.9833984, -4957.4663086, 5568.1835938, -9844.1884766, 9840.3603516
2: -7542.3886719, 5977.2192383, -7537.0581055, 5983.4145508, -12551.6992188, 12547.3330078
3: -2761.8461914, 7600.6406250, -2760.8061523, 7600.5327148, -9858.2988281, 9859.7763672
4: -8234.5263672, 5790.3916016, -8225.3759766, 5795.8872070, -13045.4541016, 13037.9941406

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.4924498, upper bound: 3612.2973976
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7974339, upper bound: 3612.2973987
time: 0.86 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.85 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.85
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.85
Output dim: 0, lower bound: -3613.1880272, upper bound: 3613.4731360
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.85
Output dim: 0, lower bound: -3612.4924498, upper bound: 3612.2973976
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.85
Output dim: 0, lower bound: -3612.7974339, upper bound: 3612.2973987

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5974.7495117, 5459.4179688, -6868.2500000, 7564.1694336
1: -1577.3613281, 1670.7596436, -4806.7124023, 5393.1640625, -6505.4912109, 6369.9423828
2: -2346.6181641, 1806.2084961, -7308.2109375, 5794.9702148, -7591.2553711, 8853.5078125
3: -884.1437378, 2366.0300293, -2678.6684570, 7362.7631836, -8029.4111328, 4845.4790039
4: -2580.5910645, 1761.1538086, -7978.4140625, 5612.7807617, -7635.2202148, 9481.4462891

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6496089, upper bound: 3612.6127432
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6505023, upper bound: 3613.0344536
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5964.4746094, 5451.2070312, -6760.8198242, 7469.0888672
1: -1496.7210693, 1578.8179932, -4798.4580078, 5384.9614258, -6417.9785156, 6278.0913086
2: -2230.0136719, 1707.1827393, -7296.0073242, 5785.8652344, -7467.4643555, 8752.7412109
3: -836.8752441, 2243.5935059, -2674.3657227, 7350.7548828, -7974.2563477, 4720.5908203
4: -2450.8525391, 1663.9034424, -7965.0136719, 5603.7885742, -7498.5029297, 9382.1855469

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6496089, upper bound: 3612.6127432
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1381955, upper bound: 3613.2791683
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.1381955, upper bound: 3613.4731360
time: 0.84 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 6.00 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 6.00
Output dim: 0, lower bound: -3612.6496089, upper bound: 3612.6127432
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 6.00
Output dim: 0, lower bound: -3612.6505023, upper bound: 3613.0344536
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 6.00
Output dim: 0, lower bound: -3613.1381955, upper bound: 3613.2791683
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 6.00
Output dim: 0, lower bound: -3613.1381955, upper bound: 3613.4731360

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1856.4094238, 1608.3928223, -5941.1967773, 5432.7519531, -6737.0888672, 7443.9667969
1: -1496.7210693, 1578.8179932, -4779.6455078, 5366.9199219, -6395.1713867, 6257.4296875
2: -2230.0136719, 1707.1827393, -7268.9130859, 5766.2202148, -7442.7128906, 8722.3896484
3: -836.8752441, 2243.5935059, -2664.2668457, 7324.0693359, -7944.0517578, 4709.3481445
4: -2450.8525391, 1663.9034424, -7934.9785156, 5584.3725586, -7473.7836914, 9348.6835938

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.7329831, upper bound: 3612.6776964
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0363927, upper bound: 3613.4028226
time: 0.95 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 4.46 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 4.46
Output dim: 0, lower bound: -3610.7329831, upper bound: 3612.6776964
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 4.46
Output dim: 0, lower bound: -3613.0363927, upper bound: 3613.4028226

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1872.6331787, 1623.9456787, -5928.4194336, 5421.5317383, -6741.0078125, 7445.4531250
1: -1509.8994141, 1593.9841309, -4769.3955078, 5355.9145508, -6396.0317383, 6262.1494141
2: -2249.5837402, 1723.5537109, -7254.2626953, 5754.2446289, -7448.6201172, 8723.5341797
3: -844.9586792, 2263.1669922, -2658.1801758, 7308.8940430, -7936.1489258, 4721.9047852
4: -2472.1391602, 1680.4029541, -7918.9594727, 5572.8476562, -7482.2734375, 9348.3466797

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.4353618, upper bound: 3612.5556845
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.4361114, upper bound: 3612.8293518
time: 0.95 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 4.19 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 4.19
Output dim: 0, lower bound: -3612.4353618, upper bound: 3612.5556845
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 4.19
Output dim: 0, lower bound: -3612.4361114, upper bound: 3612.8293518
Binary search (step 2): status=Status.VERIFIED, low=0.1250000, high=0.2500000, mid=0.1250000, abs_max=4653.345703125
rel_dist={0: [-3614.75946433006, 3614.75946433006]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.2003535, upper bound: 3613.7565307
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7565198, upper bound: 3613.7565209
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3614.2003535, upper bound: 3613.7565307
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 0, lower bound: -3613.7565198, upper bound: 3613.7565209

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2502.3757324, 2150.9697266, -4423.8598633, 4467.8574219
1: -1831.3944092, 1927.9595947, -2015.7012939, 2107.5605469, -3938.9548340, 3943.6608887
2: -2712.4650879, 2088.7180176, -2978.1022949, 2285.2739258, -4997.7377930, 5066.8188477
3: -1026.7778320, 2724.7028809, -1126.3112793, 2983.9003906, -4010.6782227, 3851.0141602
4: -2982.6787109, 2035.6074219, -3276.4265137, 2226.9489746, -5209.6279297, 5312.0336914

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201623, upper bound: 3613.7201623
time: 0.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7201623, upper bound: 3613.7565209
time: 0.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -4121.5683594, 3710.7995605, -2483.2746582, 2137.3198242, -6258.8881836, 6022.9697266
1: -3322.4802246, 3653.2983398, -2000.6915283, 2094.2460938, -5416.7260742, 5511.7841797
2: -5011.4208984, 3927.4387207, -2956.6376953, 2270.8276367, -7252.7060547, 6726.2236328
3: -1844.3164062, 5051.6909180, -1119.0009766, 2964.0288086, -4767.0771484, 6115.9516602
4: -5483.0029297, 3815.4704590, -3252.4533691, 2212.8876953, -7677.7724609, 6901.6689453

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295689
time: 2.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4908058, upper bound: 3613.4908076
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -3613.7201623, upper bound: 3613.7201623
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -3613.7201623, upper bound: 3613.7565209
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -3613.4907890, upper bound: 3613.0295689
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.73
Output dim: 0, lower bound: -3613.4908058, upper bound: 3613.4908076

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -2272.8901367, 1965.4815674, -4238.3715820, 4238.3715820
1: -1831.3944092, 1927.9595947, -1831.3944092, 1927.9595947, -3759.3540039, 3759.3540039
2: -2712.4650879, 2088.7180176, -2712.4650879, 2088.7180176, -4801.1821289, 4801.1816406
3: -1026.7778320, 2724.7028809, -1026.7778320, 2724.7028809, -3751.4807129, 3751.4807129
4: -2982.6787109, 2035.6074219, -2982.6787109, 2035.6074219, -5018.2861328, 5018.2861328

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310388, upper bound: 3612.9658088
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657992, upper bound: 3612.9658088
time: 0.83 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -2272.8901367, 1965.4815674, -4119.7871094, 3708.8930664, -5813.0761719, 6085.2685547
1: -1831.3944092, 1927.9595947, -3321.0563965, 3651.3857422, -5342.6010742, 5249.0161133
2: -2712.4650879, 2088.7180176, -5009.2607422, 3925.3830566, -6484.3217773, 7074.3007812
3: -1026.7778320, 2724.7028809, -1843.3944092, 5049.3334961, -6023.5751953, 4529.4575195
4: -2982.6787109, 2035.6074219, -5480.6293945, 3813.4809570, -6634.3105469, 7503.9604492

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310388, upper bound: 3613.4908192
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657992, upper bound: 3613.4908192
time: 0.75 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4120.2651367, 3709.4050293, -2236.0668945, 1936.5114746, -6056.7758789, 5776.8447266
1: -3321.4382324, 3651.8999023, -1801.9112549, 1899.5860596, -5221.0244141, 5313.7187500
2: -5009.8417969, 3925.9360352, -2670.0183105, 2058.1057129, -7043.9179688, 6442.7958984
3: -1843.6422119, 5049.9663086, -1011.0809937, 2684.3432617, -4489.3369141, 6008.7172852
4: -5481.2666016, 3814.0151367, -2935.8645020, 2004.9266357, -7473.6835938, 6588.3627930

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295479, upper bound: 3613.0295483
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295479, upper bound: 3613.0295684
time: 0.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -4117.6650391, 3708.6584473, -4179.7080078, 3724.9226074, -7622.2773438, 7648.4223633
1: -3319.3422852, 3651.4145508, -3368.3696289, 3666.1508789, -6796.2148438, 6817.5512695
2: -5007.0639648, 3925.3413086, -5067.0336914, 3943.1066895, -8675.8447266, 8710.7060547
3: -1842.9554443, 5047.8500977, -1861.0742188, 5085.0200195, -6796.8442383, 6759.0864258
4: -5478.1630859, 3813.3303223, -5542.0429688, 3830.6635742, -9034.5351562, 9070.6806641

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0295684, upper bound: 3613.4907930
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0295684, upper bound: 3613.4908099
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.5310388, upper bound: 3612.9658088
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.56
Output dim: 0, lower bound: -3612.9657992, upper bound: 3612.9658088
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.5310388, upper bound: 3613.4908192
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -3612.9657992, upper bound: 3613.4908192
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.0295479, upper bound: 3613.0295483
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.0295479, upper bound: 3613.0295684
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.0295684, upper bound: 3613.4907930
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -3613.0295684, upper bound: 3613.4908099

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -2272.8901367, 1965.4815674, -3999.3693848, 4041.9062500
1: -1638.3045654, 1737.5650635, -1831.3944092, 1927.9595947, -3566.2641602, 3568.9594727
2: -2435.9470215, 1879.3935547, -2712.4650879, 2088.7180176, -4524.6650391, 4591.8583984
3: -920.3916626, 2455.5791016, -1026.7778320, 2724.7028809, -3645.0944824, 3482.3569336
4: -2677.5119629, 1832.2690430, -2982.6787109, 2035.6074219, -4713.1181641, 4814.9477539

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658088, upper bound: 3612.9658088
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9658088, upper bound: 3612.9658088
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -4118.3662109, 3707.3737793, -5576.0532227, 5887.3833008
1: -1638.3045654, 1737.5650635, -3319.9199219, 3649.8605957, -5150.8520508, 5057.4843750
2: -2435.9470215, 1879.3935547, -5007.5371094, 3923.7456055, -6211.9272461, 6869.6118164
3: -920.3916626, 2455.5791016, -1842.6595459, 5047.4511719, -5918.1411133, 4263.0161133
4: -2677.5119629, 1832.2690430, -5478.7329102, 3811.8942871, -6333.2587891, 7305.2993164

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295777
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908151
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -4117.6650391, 3708.6584473, -7447.2080078, 7456.7021484
1: -3211.7971191, 3524.2065430, -3319.3422852, 3651.4145508, -6653.8247070, 6640.7534180
2: -4848.2338867, 3788.9609375, -5007.0639648, 3925.3413086, -8477.4501953, 8505.2470703
3: -1780.8483887, 4870.8652344, -1842.9554443, 5047.8500977, -6671.1782227, 6573.3349609
4: -5298.1914062, 3677.4733887, -5478.1630859, 3813.3303223, -8810.8408203, 8865.4492188

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295777
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908129
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3904.8264160, 3541.1750488, -4179.7080078, 3724.9226074, -7397.3657227, 7462.2446289
1: -3146.5612793, 3492.1206055, -3368.3696289, 3666.1508789, -6613.8989258, 6641.1010742
2: -4764.1665039, 3751.3303223, -5067.0336914, 3943.1066895, -8414.9189453, 8515.7050781
3: -1749.9495850, 4810.4135742, -1861.0742188, 5085.0200195, -6693.7148438, 6509.8574219
4: -5208.8427734, 3642.6633301, -5542.0429688, 3830.6635742, -8744.9482422, 8880.1855469

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295482, upper bound: 3612.9657827
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295483, upper bound: 3613.1324507
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6221.1977539, 5678.1694336, -4179.7080078, 3724.9226074, -9580.4199219, 9232.3974609
1: -5007.2651367, 5609.7280273, -3368.3696289, 3666.1508789, -8363.1777344, 8446.9248047
2: -7606.6391602, 6028.6923828, -5067.0336914, 3943.1066895, -11034.8173828, 10411.9716797
3: -2787.5065918, 7665.3764648, -1861.0742188, 5085.0200195, -7581.9516602, 9205.7402344
4: -8303.8027344, 5839.7431641, -5542.0429688, 3830.6635742, -11612.7402344, 10695.2919922

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0295482, upper bound: 3612.9657992
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0295483, upper bound: 3613.4896442
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.63 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9658088, upper bound: 3612.9658088
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9658088, upper bound: 3612.9658088
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295777
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908151
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.0295777
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 0, lower bound: -3612.9657827, upper bound: 3613.4908129
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3613.0295482, upper bound: 3612.9657827
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3613.0295483, upper bound: 3613.1324507
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 0, lower bound: -3613.0295482, upper bound: 3612.9657992
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 0, lower bound: -3613.0295483, upper bound: 3613.4896442

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2033.8878174, 1769.0169678, -6054.2851562, 5522.8408203, -7026.7763672, 7721.3598633
1: -1638.3045654, 1737.5650635, -4871.4775391, 5456.5097656, -6646.1982422, 6513.0175781
2: -2435.9470215, 1879.3935547, -7402.2500000, 5863.7973633, -7770.0009766, 9036.4736328
3: -920.3916626, 2455.5791016, -2713.7316895, 7455.2700195, -8166.6748047, 4980.5292969
4: -2677.5119629, 1832.2690430, -8081.4262695, 5680.0278320, -7819.5166016, 9673.2187500

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4056440, upper bound: 3613.4907664
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310159, upper bound: 3613.4907813
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3987.9494629, 3574.2734375, -6220.2084961, 5677.0874023, -9030.3105469, 9413.8837891
1: -3211.7971191, 3524.2065430, -5006.4658203, 5608.6503906, -8282.2949219, 8206.9550781
2: -4848.2338867, 3788.9609375, -7605.4389648, 6027.5322266, -10177.7636719, 10863.0869141
3: -1780.8483887, 4870.8652344, -2786.9855957, 7664.0449219, -9116.5957031, 7357.9960938
4: -5298.1914062, 3677.4733887, -8302.4833984, 5838.6269531, -10434.5585938, 11442.4023438

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657290, upper bound: 3613.0662419
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9265274, upper bound: 3613.0051433
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6225.2827148, 5682.6333008, -6225.2827148, 5682.6333008, -11132.6572266, 11132.6582031
1: -5010.5698242, 5614.1787109, -5010.5698242, 5614.1787109, -9965.4804688, 9965.4804688
2: -7611.5888672, 6033.4814453, -7611.5888672, 6033.4814453, -12706.6269531, 12706.6269531
3: -2789.6572266, 7670.8759766, -2789.6572266, 7670.8759766, -9975.6748047, 9975.6748047
4: -8309.2480469, 5844.3500977, -8309.2480469, 5844.3500977, -13205.4160156, 13205.4160156

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4851302, upper bound: 3613.2934083
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941068, upper bound: 3613.2932243
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.38 seconds
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 0, lower bound: -3613.4056440, upper bound: 3613.4907664
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 0, lower bound: -3613.5310159, upper bound: 3613.4907813
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 0, lower bound: -3612.9657290, upper bound: 3613.0662419
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 0, lower bound: -3612.9265274, upper bound: 3613.0051433
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.38
Output dim: 0, lower bound: -3613.4851302, upper bound: 3613.2934083
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 0, lower bound: -3613.2941068, upper bound: 3613.2932243

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1789.6997070, 1554.3094482, -5989.7011719, 5470.9951172, -6730.5766602, 7442.4160156
1: -1442.5327148, 1525.0668945, -4819.4584961, 5405.1748047, -6398.9975586, 6248.0751953
2: -2150.3056641, 1650.0454102, -7325.9213867, 5807.8232422, -7428.5664062, 8730.2041016
3: -806.7396851, 2160.4394531, -2686.0136719, 7380.0639648, -7977.8481445, 4658.1767578
4: -2361.8547363, 1606.7463379, -7996.9477539, 5624.6958008, -7449.1503906, 9362.2167969

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4056484, upper bound: 3613.4528233
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4055725, upper bound: 3613.2940652
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1991.7487793, 1732.1446533, -6039.8666992, 5510.7583008, -6969.6499023, 7667.6718750
1: -1604.6556396, 1701.4029541, -4859.8212891, 5444.6420898, -6597.8818359, 6462.3955078
2: -2386.9431152, 1840.0159912, -7385.1801758, 5850.9077148, -7704.5981445, 8977.4316406
3: -900.5989380, 2405.9113770, -2707.4050293, 7438.1298828, -8129.3613281, 4921.5444336
4: -2623.7451172, 1794.3088379, -8062.6464844, 5667.3745117, -7749.4165039, 9613.1464844

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5310195, upper bound: 3613.4907210
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.5309589, upper bound: 3613.2941041
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6237.6035156, 5690.1298828, -6215.9658203, 5674.3188477, -11134.0410156, 11126.6191406
1: -5019.8442383, 5621.1367188, -5003.2031250, 5605.9624023, -9963.9033203, 9961.2480469
2: -7624.3247070, 6040.0839844, -7600.8222656, 6024.6308594, -12707.2724609, 12697.6162109
3: -2793.9113770, 7683.4697266, -2785.3095703, 7659.7734375, -9966.0400391, 9983.1474609
4: -8325.0566406, 5851.8598633, -8297.4150391, 5835.8105469, -13208.8652344, 13195.5224609

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941034, upper bound: 3613.2932210
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2941034, upper bound: 3613.2932210
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.44 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.4056484, upper bound: 3613.4528233
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.4055725, upper bound: 3613.2940652
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.5310195, upper bound: 3613.4907210
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.5309589, upper bound: 3613.2941041
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.2941034, upper bound: 3613.2932210
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 0, lower bound: -3613.2941034, upper bound: 3613.2932210

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1779.3359375, 1544.8118896, -5980.5312500, 5459.8310547, -6705.8666992, 7422.1645508
1: -1434.2962646, 1515.9592285, -4811.3110352, 5393.8115234, -6376.4936523, 6228.9985352
2: -2138.2351074, 1640.2750244, -7312.7050781, 5794.5073242, -7399.4418945, 8705.4072266
3: -802.0129395, 2147.9094238, -2680.5759277, 7366.6284180, -7959.6142578, 4637.8105469
4: -2348.6105957, 1597.3273926, -7984.5434570, 5613.0200195, -7419.9926758, 9337.4707031

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.3401078, upper bound: 3613.4226259
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2950962, upper bound: 3613.4525732
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1782.6702881, 1547.5867920, -5912.6611328, 5398.7226562, -6652.7036133, 7355.8793945
1: -1437.0015869, 1518.1776123, -4757.7255859, 5333.8129883, -6323.3217773, 6176.9575195
2: -2142.1704102, 1642.6020508, -7236.1523438, 5731.0332031, -7345.0771484, 8627.6230469
3: -803.2163086, 2151.7790527, -2649.5068359, 7286.0483398, -7878.4140625, 4613.2226562
4: -2352.8940430, 1599.4140625, -7897.6425781, 5548.9702148, -7366.4130859, 9249.2265625

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.3357457, upper bound: 3613.2862198
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2950242, upper bound: 3613.2939840
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1985.4197998, 1726.6549072, -6041.5810547, 5507.9121094, -6958.6923828, 7662.6411133
1: -1599.6429443, 1695.9609375, -4860.4150391, 5441.4238281, -6587.9921875, 6456.2768555
2: -2379.5554199, 1834.1531982, -7384.6953125, 5846.4731445, -7690.5654297, 8970.0625000
3: -897.6294556, 2398.2255859, -2706.6523438, 7437.0869141, -8125.9843750, 4911.2773438
4: -2615.6225586, 1788.6892090, -8064.2343750, 5664.3725586, -7735.4750977, 9607.2294922

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4489783, upper bound: 3613.4898848
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668913, upper bound: 3613.4906595
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1981.3195801, 1721.9357910, -5963.8896484, 5439.5483398, -6889.4282227, 7578.9160156
1: -1596.3958740, 1691.2861328, -4798.9135742, 5374.4423828, -6520.6030273, 6388.7607422
2: -2374.8383789, 1829.1168213, -7296.6811523, 5775.2836914, -7618.2661133, 8872.5136719
3: -895.5196533, 2393.0617676, -2671.4301758, 7345.3837891, -8029.7636719, 4872.8496094
4: -2610.4223633, 1783.6278076, -7964.6191406, 5592.7431641, -7663.3637695, 9498.1572266

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4488997, upper bound: 3613.2930882
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668591, upper bound: 3613.2940211
time: 1.25 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.28 seconds
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3612.3401078, upper bound: 3613.4226259
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.2950962, upper bound: 3613.4525732
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 0, lower bound: -3612.3357457, upper bound: 3613.2862198
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.2950242, upper bound: 3613.2939840
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.4489783, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.4668913, upper bound: 3613.4906595
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.4488997, upper bound: 3613.2930882
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.28
Output dim: 0, lower bound: -3613.4668591, upper bound: 3613.2940211

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1712.5506592, 1487.8541260, -5963.7080078, 5445.6494141, -6625.9326172, 7349.2055664
1: -1380.6622314, 1458.6721191, -4797.6943359, 5379.7685547, -6309.7363281, 6161.0063477
2: -2060.8286133, 1577.6282959, -7292.6660156, 5779.2548828, -7308.3789062, 8626.4951172
3: -770.7150879, 2069.7250977, -2673.2473145, 7346.6152344, -7908.9340820, 4553.1845703
4: -2263.2678223, 1535.8945312, -7962.5034180, 5597.9438477, -7321.2734375, 9255.7998047

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.1970628, upper bound: 3612.5882741
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.1970628, upper bound: 3613.0996172
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1734.3096924, 1511.8684082, -5962.5375977, 5445.5825195, -6645.7939453, 7365.1201172
1: -1399.1408691, 1480.3322754, -4796.9946289, 5379.5639648, -6325.9584961, 6177.4946289
2: -2088.3193359, 1595.7926025, -7292.1279297, 5779.0068359, -7332.7841797, 8643.2841797
3: -779.2869873, 2098.8481445, -2672.6357422, 7346.3623047, -7916.0175781, 4578.9487305
4: -2292.2456055, 1554.7314453, -7961.6840820, 5597.7563477, -7348.1499023, 9271.7119141

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1451031, upper bound: 3612.5948873
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1451610, upper bound: 3613.1382963
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1913.9602051, 1664.4067383, -6026.0732422, 5495.2548828, -6875.2451172, 7587.4604492
1: -1542.3728027, 1635.8276367, -4847.8608398, 5429.0371094, -6518.8144531, 6384.4545898
2: -2296.8923340, 1768.6951904, -7366.1840820, 5832.9306641, -7595.2666016, 8887.1933594
3: -865.0184937, 2314.8002930, -2699.9755859, 7418.7783203, -8075.4804688, 4821.8149414
4: -2524.5927734, 1724.6793213, -8043.8823242, 5650.9868164, -7632.0751953, 9523.9658203

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991631
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1934.4174805, 1684.0871582, -6024.4023438, 5494.7910156, -6893.4018555, 7600.9067383
1: -1559.6101074, 1649.4707031, -4846.7490234, 5428.4975586, -6533.5815430, 6398.0429688
2: -2322.1723633, 1782.7749023, -7364.9809570, 5832.2871094, -7617.4042969, 8901.2812500
3: -872.7719727, 2339.7248535, -2699.1518555, 7417.8618164, -8081.2739258, 4843.9160156
4: -2551.3425293, 1736.4350586, -8042.3320312, 5650.4160156, -7656.4150391, 9537.0595703

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1909.9262695, 1660.0725098, -5948.2314453, 5426.5078125, -6805.7167969, 7503.6767578
1: -1539.2054443, 1631.2471924, -4786.2055664, 5361.5981445, -6451.1147461, 6316.8657227
2: -2292.2685547, 1763.7724609, -7278.0083008, 5761.2705078, -7522.6767578, 8789.5390625
3: -862.9491577, 2309.7221680, -2664.6088867, 7326.8105469, -7979.0136719, 4783.3388672
4: -2519.4956055, 1719.7012939, -7944.0952148, 5578.8886719, -7559.7001953, 9414.7480469

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930882
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1929.9622803, 1679.7984619, -5945.6923828, 5425.1459961, -6822.5737305, 7515.8559570
1: -1556.0819092, 1645.2132568, -4784.4130859, 5360.1518555, -6464.6318359, 6329.5322266
2: -2317.0366211, 1777.6666260, -7275.7856445, 5759.6757812, -7543.3593750, 8802.2470703
3: -870.4957275, 2334.2561035, -2663.3376465, 7324.8842773, -7983.5239258, 4804.6206055
4: -2545.6765137, 1731.2369385, -7941.4248047, 5577.4106445, -7582.5610352, 9426.3603516

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211
time: 0.76 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.38 seconds
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.38
Output dim: 0, lower bound: -3612.1970628, upper bound: 3612.5882741
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.38
Output dim: 0, lower bound: -3612.1970628, upper bound: 3613.0996172
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.1451031, upper bound: 3612.5948873
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.1451610, upper bound: 3613.1382963
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991631
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930882
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.38
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1913.9602051, 1664.4067383, -5968.2153320, 5450.4965820, -6820.8515625, 7524.7651367
1: -1542.3728027, 1635.8276367, -4801.0830078, 5385.6308594, -6466.9467773, 6333.8164062
2: -2296.8923340, 1768.6951904, -7300.1054688, 5785.8271484, -7538.2084961, 8813.6298828
3: -865.0184937, 2314.8002930, -2675.1586914, 7353.4331055, -8004.5708008, 4793.1616211
4: -2524.5927734, 1724.6793213, -7970.3173828, 5604.1020508, -7575.6997070, 9442.3847656

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991631
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991625
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1913.9602051, 1664.4067383, -5990.7578125, 5474.5341797, -6841.1640625, 7548.9873047
1: -1542.3728027, 1635.8276367, -4819.6035156, 5408.2246094, -6486.5756836, 6353.5424805
2: -2296.8923340, 1768.6951904, -7327.2709961, 5810.9760742, -7558.9853516, 8842.3222656
3: -865.0184937, 2314.8002930, -2686.0202637, 7382.4345703, -8033.4755859, 4802.7158203
4: -2524.5927734, 1724.6793213, -7999.3559570, 5628.3354492, -7596.6582031, 9473.8417969

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1934.4174805, 1684.0871582, -5745.3100586, 5259.0917969, -6630.1030273, 7306.6884766
1: -1559.6101074, 1649.4707031, -4622.6757812, 5195.6083984, -6275.6201172, 6161.6826172
2: -2322.1723633, 1782.7749023, -7042.6176758, 5582.2563477, -7337.3818359, 8555.1914062
3: -872.7719727, 2339.7248535, -2573.6015625, 7086.9218750, -7735.0620117, 4707.2534180
4: -2551.3425293, 1736.4350586, -7684.5551758, 5403.9902344, -7383.4223633, 9153.6123047

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1934.4174805, 1684.0871582, -5995.4038086, 5471.8613281, -6863.5053711, 7569.6220703
1: -1559.6101074, 1649.4707031, -4823.3286133, 5406.0258789, -6504.8349609, 6372.2514648
2: -2322.1723633, 1782.7749023, -7331.1665039, 5807.8369141, -7586.2954102, 8863.3417969
3: -872.7719727, 2339.7248535, -2686.5283203, 7384.3896484, -8043.2084961, 4829.9604492
4: -2551.3425293, 1736.4350586, -8004.8413086, 5626.2636719, -7625.2485352, 9495.1982422

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1909.9262695, 1660.0725098, -5889.7597656, 5381.2089844, -6750.3627930, 7440.1508789
1: -1539.2054443, 1631.2471924, -4738.9760742, 5317.6157227, -6398.2973633, 6265.5791016
2: -2292.2685547, 1763.7724609, -7211.4033203, 5713.5175781, -7464.5161133, 8715.0517578
3: -862.9491577, 2309.7221680, -2639.2905273, 7261.3652344, -7907.5756836, 4754.1015625
4: -2519.4956055, 1719.7012939, -7869.8764648, 5531.4531250, -7502.3720703, 9332.0712891

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1909.9262695, 1660.0725098, -5909.8779297, 5402.9282227, -6768.2519531, 7461.8725586
1: -1539.2054443, 1631.2471924, -4755.5927734, 5338.0581055, -6415.6918945, 6283.3334961
2: -2292.2685547, 1763.7724609, -7235.8125000, 5736.3188477, -7482.8310547, 8740.8339844
3: -862.9491577, 2309.7221680, -2649.2490234, 7287.4584961, -7933.4921875, 4762.3120117
4: -2519.4956055, 1719.7012939, -7895.8334961, 5553.1879883, -7520.7656250, 9360.2910156

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930114
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930882
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1929.9622803, 1679.7984619, -5715.1860352, 5233.7041016, -6605.1176758, 7271.9438477
1: -1556.0819092, 1645.2132568, -4599.9252930, 5171.1010742, -6251.6254883, 6133.9868164
2: -2317.0366211, 1777.6666260, -7010.1259766, 5555.9653320, -7311.6733398, 8515.2773438
3: -870.4957275, 2334.2561035, -2558.4243164, 7053.7944336, -7698.6889648, 4689.8242188
4: -2545.6765137, 1731.2369385, -7646.3701172, 5375.9829102, -7356.7587891, 9108.3691406

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1929.9622803, 1679.7984619, -5904.7426758, 5392.6445312, -6781.0102539, 7471.6918945
1: -1556.0819092, 1645.2132568, -4751.3168945, 5328.2910156, -6424.6152344, 6293.3271484
2: -2317.0366211, 1777.6666260, -7228.1127930, 5724.8862305, -7499.5756836, 8749.0000000
3: -870.4957275, 2334.2561035, -2645.2148438, 7277.8828125, -7930.7998047, 4784.2602539
4: -2545.6765137, 1731.2369385, -7888.6655273, 5543.1816406, -7539.1972656, 9367.6757812

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211
time: 0.91 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.95 seconds
IS_A1_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991631
IS_A1_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4474669, upper bound: 3611.8991625
IS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4474669, upper bound: 3613.4898848
IS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
IS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.2832242
IS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668645, upper bound: 3613.4906431
IS_A1_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
IS_A1_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4464879, upper bound: 3611.7100915
IS_A1_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930114
IS_A1_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4464879, upper bound: 3613.2930882
IS_A1_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
IS_A1_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2503309
IS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211
IS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.95
Output dim: 0, lower bound: -3613.4668200, upper bound: 3613.2940211

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1957.8157959, 1701.1000977, -5966.2104492, 5447.9570312, -6861.4472656, 7554.3735352
1: -1577.3613281, 1670.7596436, -4799.5024414, 5382.9755859, -6499.1601562, 6362.0195312
2: -2346.6181641, 1806.2084961, -7297.6806641, 5783.1591797, -7584.6591797, 8842.2099609
3: -884.1437378, 2366.0300293, -2674.0756836, 7350.7695312, -8018.4433594, 4842.2294922
4: -2580.5910645, 1761.1538086, -7967.6054688, 5601.5087891, -7628.1831055, 9469.0634766

Time for backsubstitution: 1.73 seconds
Binary search (step 3): status=Status.UNKNOWN, low=0.1250000, high=0.1875000, mid=0.1875000, abs_max=4653.345703125
rel_dist={0: [-3614.7606139725563, 3614.7606139725576]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.125
execution time: 1120.01 seconds
