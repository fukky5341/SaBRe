## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 3613.31311749156


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2502.3757324, 2150.9697266, -2502.3757324, 2150.9697266, -4653.3457031, 4653.3457031)
1: (-2015.7012939, 2107.5605469, -2015.7012939, 2107.5605469, -4123.2617188, 4123.2612305)
2: (-2978.1022949, 2285.2739258, -2978.1022949, 2285.2739258, -5263.3750000, 5263.3750000)
3: (-1126.3112793, 2983.9003906, -1126.3112793, 2983.9003906, -4110.2119141, 4110.2119141)
4: (-3276.4265137, 2226.9489746, -3276.4265137, 2226.9489746, -5503.3754883, 5503.3754883)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.81 + 2.07 = 3.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3614.7590211, upper bound: 3614.7590211

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3614.0950480, upper bound: 3613.7564454
time: 0.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7564440, upper bound: 3613.7564451
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.77 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3614.0950480, upper bound: 3613.7564454
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -3613.7564440, upper bound: 3613.7564451

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2272.8901367, 1965.4815674, -2502.3757324, 2150.9697266, -4423.8598633, 4467.8574219
1: -1831.3944092, 1927.9595947, -2015.7012939, 2107.5605469, -3938.9548340, 3943.6608887
2: -2712.4650879, 2088.7180176, -2978.1022949, 2285.2739258, -4997.7377930, 5066.8188477
3: -1026.7778320, 2724.7028809, -1126.3112793, 2983.9003906, -4010.6782227, 3851.0141602
4: -2982.6787109, 2035.6074219, -3276.4265137, 2226.9489746, -5209.6279297, 5312.0336914

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7200364, upper bound: 3613.7200364
time: 0.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.7200364, upper bound: 3613.7564451
time: 0.72 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4121.1752930, 3710.3796387, -2468.8952637, 2126.7873535, -6247.9628906, 6008.3168945
1: -3322.1665039, 3652.8771973, -1989.3304443, 2083.9890137, -5406.1552734, 5500.1254883
2: -5010.9453125, 3926.9860840, -2940.3427734, 2259.6884766, -7241.0971680, 6709.7031250
3: -1844.1132812, 5051.1713867, -1113.3750000, 2948.8332520, -4751.8159180, 6109.9306641
4: -5482.4799805, 3815.0322266, -3234.3330078, 2202.0334473, -7666.3291016, 6883.3364258

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907393, upper bound: 3613.0293854
time: 0.85 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4907418, upper bound: 3613.4907436
time: 1.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.28 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -3613.7200364, upper bound: 3613.7200364
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -3613.7200364, upper bound: 3613.7564451
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -3613.4907393, upper bound: 3613.0293854
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.28
Output dim: 0, lower bound: -3613.4907418, upper bound: 3613.4907436

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2272.8901367, 1965.4815674, -2272.8901367, 1965.4815674, -4238.3715820, 4238.3715820
1: -1831.3944092, 1927.9595947, -1831.3944092, 1927.9595947, -3759.3540039, 3759.3540039
2: -2712.4650879, 2088.7180176, -2712.4650879, 2088.7180176, -4801.1821289, 4801.1816406
3: -1026.7778320, 2724.7028809, -1026.7778320, 2724.7028809, -3751.4807129, 3751.4807129
4: -2982.6787109, 2035.6074219, -2982.6787109, 2035.6074219, -5018.2861328, 5018.2861328

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4263791, upper bound: 3612.9657041
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9656991, upper bound: 3612.9657041
time: 0.80 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2272.8901367, 1965.4815674, -4119.3598633, 3708.4355469, -5812.6948242, 6084.8408203
1: -1831.3944092, 1927.9595947, -3320.7141113, 3650.9270020, -5342.2055664, 5248.6738281
2: -2712.4650879, 2088.7180176, -5008.7421875, 3924.8901367, -6483.9067383, 7073.8037109
3: -1026.7778320, 2724.7028809, -1843.1733398, 5048.7666016, -6023.0395508, 4529.2675781
4: -2982.6787109, 2035.6074219, -5480.0571289, 3813.0034180, -6633.9165039, 7503.4116211

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4263791, upper bound: 3613.4907499
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9656991, upper bound: 3613.4907499
time: 0.77 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4119.8476562, 3708.9584961, -2222.4233398, 1926.0036621, -6045.8505859, 5762.9052734
1: -3321.1044922, 3651.4509277, -1791.0668945, 1889.2574463, -5210.3618164, 5302.5781250
2: -5009.3349609, 3925.4533691, -2654.4113770, 2046.9775391, -7032.2998047, 6426.9682617
3: -1843.4259033, 5049.4116211, -1005.4481201, 2669.2331543, -4474.1523438, 6002.6611328
4: -5480.7089844, 3813.5483398, -2918.5017090, 1994.1152344, -7462.2968750, 6570.7729492

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0293754, upper bound: 3613.0293754
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0293754, upper bound: 3613.0293849
time: 0.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4109.9501953, 3701.8200684, -4175.7622070, 3721.6879883, -7611.2470703, 7637.5053711
1: -3313.1657715, 3644.7248535, -3365.3410645, 3662.7861328, -6786.6469727, 6807.6713867
2: -4997.9687500, 3918.1044922, -5062.3271484, 3939.4084473, -8662.9638672, 8698.6621094
3: -1839.4263916, 5038.5551758, -1859.0759277, 5080.3408203, -6788.4311523, 6747.6855469
4: -5468.1376953, 3806.2521973, -5536.9765625, 3827.2668457, -9020.9951172, 9058.3867188

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0293849, upper bound: 3613.4907434
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0293849, upper bound: 3613.4907458
time: 1.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.43 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.4263791, upper bound: 3612.9657041
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.43
Output dim: 0, lower bound: -3612.9656991, upper bound: 3612.9657041
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.4263791, upper bound: 3613.4907499
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -3612.9656991, upper bound: 3613.4907499
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.0293754, upper bound: 3613.0293754
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.0293754, upper bound: 3613.0293849
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.0293849, upper bound: 3613.4907434
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 0, lower bound: -3613.0293849, upper bound: 3613.4907458

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2033.8878174, 1769.0169678, -2272.8901367, 1965.4815674, -3999.3693848, 4041.9062500
1: -1638.3045654, 1737.5650635, -1831.3944092, 1927.9595947, -3566.2641602, 3568.9594727
2: -2435.9470215, 1879.3935547, -2712.4650879, 2088.7180176, -4524.6650391, 4591.8583984
3: -920.3916626, 2455.5791016, -1026.7778320, 2724.7028809, -3645.0944824, 3482.3569336
4: -2677.5119629, 1832.2690430, -2982.6787109, 2035.6074219, -4713.1181641, 4814.9477539

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657041, upper bound: 3612.9657041
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9657041, upper bound: 3612.9657041
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2033.8878174, 1769.0169678, -4117.9086914, 3706.8845215, -5575.6459961, 5886.9257812
1: -1638.3045654, 1737.5650635, -3319.5551758, 3649.3706055, -5150.4296875, 5057.1196289
2: -2435.9470215, 1879.3935547, -5006.9843750, 3923.2189941, -6211.4829102, 6869.0815430
3: -920.3916626, 2455.5791016, -1842.4229736, 5046.8471680, -5917.5698242, 4262.8129883
4: -2677.5119629, 1832.2690430, -5478.1245117, 3811.3837891, -6332.8383789, 7304.7138672

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.0293938
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.4907459
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3987.9494629, 3574.2734375, -4109.9501953, 3701.8200684, -7440.1909180, 7448.8325195
1: -3211.7971191, 3524.2065430, -3313.1657715, 3644.7248535, -6646.9702148, 6634.4677734
2: -4848.2338867, 3788.9609375, -4997.9687500, 3918.1044922, -8470.0039062, 8495.9501953
3: -1780.8483887, 4870.8652344, -1839.4263916, 5038.5551758, -6661.7441406, 6569.6020508
4: -5298.1914062, 3677.4733887, -5468.1376953, 3806.2521973, -8803.5634766, 8855.1943359

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.0293938
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.4907437
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3904.8264160, 3541.1750488, -4175.7622070, 3721.6879883, -7394.2055664, 7458.3452148
1: -3146.5612793, 3492.1206055, -3365.3410645, 3662.7861328, -6610.6171875, 6638.0747070
2: -4764.1665039, 3751.3303223, -5062.3271484, 3939.4084473, -8411.3339844, 8511.1093750
3: -1749.9495850, 4810.4135742, -1859.0759277, 5080.3408203, -6689.0356445, 6507.8911133
4: -5208.8427734, 3642.6633301, -5536.9765625, 3827.2668457, -8741.6650391, 8875.1689453

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0293754, upper bound: 3612.9656950
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0293758, upper bound: 3613.0961465
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6221.5502930, 5678.5532227, -4175.7622070, 3721.6879883, -9577.6015625, 9228.8076172
1: -5007.5498047, 5610.1118164, -3365.3410645, 3662.7861328, -8360.1669922, 8444.2207031
2: -7607.0649414, 6029.1044922, -5062.3271484, 3939.4084473, -11031.6406250, 10407.7119141
3: -2787.6921387, 7665.8505859, -1859.0759277, 5080.3408203, -7577.4316406, 9204.2119141
4: -8304.2714844, 5840.1396484, -5536.9765625, 3827.2668457, -11609.9052734, 10690.5966797

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0293756, upper bound: 3612.9656991
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0293758, upper bound: 3613.4896157
time: 0.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.29 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9657041, upper bound: 3612.9657041
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9657041, upper bound: 3612.9657041
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.0293938
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.4907459
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.0293938
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -3612.9656950, upper bound: 3613.4907437
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.0293754, upper bound: 3612.9656950
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.0293758, upper bound: 3613.0961465
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.0293756, upper bound: 3612.9656991
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -3613.0293758, upper bound: 3613.4896157

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2033.8878174, 1769.0169678, -6050.0478516, 5518.5126953, -7023.1586914, 7717.2675781
1: -1638.3045654, 1737.5650635, -4868.0512695, 5452.2338867, -6642.5278320, 6509.7211914
2: -2435.9470215, 1879.3935547, -7397.1108398, 5859.2475586, -7766.1372070, 9031.6074219
3: -920.3916626, 2455.5791016, -2711.7541504, 7449.7900391, -8161.5053711, 4978.7387695
4: -2677.5119629, 1832.2690430, -8075.8657227, 5675.6782227, -7815.8369141, 9667.9228516

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2485752, upper bound: 3613.4905997
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4263741, upper bound: 3613.4907288
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3987.9494629, 3574.2734375, -6219.9755859, 5676.8320312, -9030.1044922, 9413.6601562
1: -3211.7971191, 3524.2065430, -5006.2758789, 5608.3950195, -8282.0820312, 8206.7744141
2: -4848.2338867, 3788.9609375, -7605.1562500, 6027.2573242, -10177.5390625, 10862.8183594
3: -1780.8483887, 4870.8652344, -2786.8632812, 7663.7309570, -9116.3017578, 7357.8901367
4: -5298.1914062, 3677.4733887, -8302.1718750, 5838.3637695, -10434.3447266, 11442.1083984

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9656079, upper bound: 3613.0112044
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9264001, upper bound: 3612.9791602
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6225.2827148, 5682.6333008, -6225.2827148, 5682.6333008, -11132.6572266, 11132.6582031
1: -5010.5698242, 5614.1787109, -5010.5698242, 5614.1787109, -9965.4804688, 9965.4804688
2: -7611.5888672, 6033.4814453, -7611.5888672, 6033.4814453, -12706.6269531, 12706.6269531
3: -2789.6572266, 7670.8759766, -2789.6572266, 7670.8759766, -9975.6748047, 9975.6748047
4: -8309.2480469, 5844.3500977, -8309.2480469, 5844.3500977, -13205.4160156, 13205.4160156

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4290690, upper bound: 3613.2932619
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2940005, upper bound: 3613.2930855
time: 0.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.39 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -3613.2485752, upper bound: 3613.4905997
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -3613.4263741, upper bound: 3613.4907288
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.39
Output dim: 0, lower bound: -3612.9656079, upper bound: 3613.0112044
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.39
Output dim: 0, lower bound: -3612.9264001, upper bound: 3612.9791602
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.39
Output dim: 0, lower bound: -3613.4290690, upper bound: 3613.2932619
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.39
Output dim: 0, lower bound: -3613.2940005, upper bound: 3613.2930855

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1789.6997070, 1554.3094482, -5939.8579102, 5428.8720703, -6685.7304688, 7390.6962891
1: -1442.5327148, 1525.0668945, -4779.5278320, 5363.8037109, -6355.0766602, 6206.5332031
2: -2150.3056641, 1650.0454102, -7268.1103516, 5763.3457031, -7380.9853516, 8669.3876953
3: -806.7396851, 2160.4394531, -2663.4941406, 7321.3022461, -7916.9995117, 4634.2812500
4: -2361.8547363, 1606.7463379, -7932.8291016, 5581.1235352, -7402.9658203, 9295.0009766

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2445504, upper bound: 3613.3748925
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2443563, upper bound: 3613.2939309
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1991.7487793, 1732.1446533, -6018.5751953, 5492.9931641, -6948.7109375, 7645.2011719
1: -1604.6556396, 1701.4029541, -4842.6162109, 5427.2324219, -6577.5795898, 6444.0249023
2: -2386.9431152, 1840.0159912, -7360.1411133, 5832.0346680, -7682.5917969, 8950.3671875
3: -900.5989380, 2405.9113770, -2697.9763184, 7412.9931641, -8102.1206055, 4911.4272461
4: -2623.7451172, 1794.3088379, -8035.0620117, 5648.8901367, -7727.6621094, 9583.3964844

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
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
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2445504, upper bound: 3613.4433261
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4261243, upper bound: 3613.2940026
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6237.6035156, 5690.1298828, -6206.2812500, 5665.5786133, -11122.9892578, 11116.6601562
1: -5019.8442383, 5621.1367188, -4995.5488281, 5597.3012695, -9953.1562500, 9953.3154297
2: -7624.3247070, 6040.0839844, -7589.5917969, 6015.2963867, -12695.4462891, 12685.8964844
3: -2793.9113770, 7683.4697266, -2780.7482910, 7648.1694336, -9953.7890625, 9977.5869141
4: -8325.0566406, 5851.8598633, -8285.0986328, 5826.8349609, -13197.3593750, 13182.6103516

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
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
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4296194, upper bound: 3611.7102511
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4322322, upper bound: 3613.2932540
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.43 seconds
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.2445504, upper bound: 3613.3748925
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.2443563, upper bound: 3613.2939309
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.2445504, upper bound: 3613.4433261
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.4261243, upper bound: 3613.2940026
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.4296194, upper bound: 3611.7102511
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.43
Output dim: 0, lower bound: -3613.4322322, upper bound: 3613.2932540

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1768.6116943, 1535.0305176, -5919.8940430, 5409.1225586, -6640.6142578, 7347.6352539
1: -1425.7852783, 1506.5795898, -4762.6455078, 5344.0400391, -6314.7578125, 6167.1562500
2: -2125.7839355, 1630.1907959, -7242.2314453, 5740.9257812, -7329.1743164, 8619.1767578
3: -797.1257324, 2135.0244141, -2653.5341797, 7295.2558594, -7880.0019531, 4595.8227539
4: -2334.9526367, 1587.6364746, -7906.5175781, 5560.5454102, -7350.2670898, 9243.5478516

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.7404769, upper bound: 3613.1188239
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6055279, upper bound: 3613.1188250
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1978.0886230, 1720.3070068, -6024.1987305, 5492.9257812, -6934.4077148, 7636.2504883
1: -1593.8343506, 1689.6645508, -4846.3759766, 5426.7211914, -6565.7343750, 6433.4697266
2: -2370.9899902, 1827.3648682, -7364.0839844, 5830.5966797, -7664.3325195, 8939.4238281
3: -894.1958618, 2389.3476562, -2698.9775391, 7416.2099609, -8099.4033203, 4894.1713867
4: -2606.2072754, 1782.1831055, -8041.6083984, 5648.8881836, -7708.5605469, 9574.5224609

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
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

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9911751, upper bound: 3613.4421356
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3441509, upper bound: 3613.4430272
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1960.4013672, 1701.3614502, -5939.9858398, 5419.3110352, -6845.7744141, 7534.9384766
1: -1579.8055420, 1670.9196777, -4779.6157227, 5354.4956055, -6481.7290039, 6349.0937500
2: -2350.5439453, 1807.1522217, -7268.5068359, 5753.6586914, -7569.8085938, 8821.9365234
3: -885.2645874, 2367.2934570, -2660.7985840, 7317.1000977, -7989.9218750, 4836.1596680
4: -2583.6481934, 1762.0908203, -7933.5825195, 5571.5605469, -7612.6557617, 9445.1718750

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
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
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9763156, upper bound: 3613.2923006
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3433947, upper bound: 3613.2939390
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6237.6035156, 5690.1298828, -6135.5400391, 5608.7929688, -11056.2919922, 11040.5615234
1: -5019.8442383, 5621.1367188, -4938.4672852, 5542.0693359, -9889.1152344, 9891.9628906
2: -7624.3247070, 6040.0839844, -7508.5771484, 5955.6547852, -12625.3808594, 12596.5751953
3: -2793.9113770, 7683.4697266, -2750.0100098, 7567.7348633, -9867.2265625, 9942.8320312
4: -8325.0566406, 5851.8598633, -8195.0068359, 5767.7641602, -13128.4345703, 13083.5761719

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8985877, upper bound: 3611.7095088
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8985876, upper bound: 3611.7102511
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6224.0009766, 5678.5751953, -6165.8349609, 5640.2246094, -11068.5996094, 11059.7177734
1: -5008.9531250, 5609.7636719, -4963.3056641, 5572.0395508, -9903.6650391, 9905.4863281
2: -7608.5654297, 6027.7626953, -7545.2167969, 5988.7241211, -12636.1630859, 12620.7597656
3: -2787.6606445, 7667.4243164, -2764.2185059, 7606.4638672, -9898.6035156, 9938.5175781
4: -8307.6777344, 5839.8510742, -8234.2939453, 5799.4824219, -13137.4443359, 13111.7685547

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2774791, upper bound: 3613.2931426
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.4322312, upper bound: 3613.2932111
time: 0.87 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.45 seconds
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3612.7404769, upper bound: 3613.1188239
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3612.6055279, upper bound: 3613.1188250
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 0, lower bound: -3612.9911751, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 0, lower bound: -3613.3441509, upper bound: 3613.4430272
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3612.9763156, upper bound: 3613.2923006
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 0, lower bound: -3613.3433947, upper bound: 3613.2939390
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3611.8985877, upper bound: 3611.7095088
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3611.8985876, upper bound: 3611.7102511
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 0, lower bound: -3613.2774791, upper bound: 3613.2931426
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 0, lower bound: -3613.4322312, upper bound: 3613.2932111

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1906.5887451, 1658.0129395, -6013.7993164, 5484.0839844, -6855.6870117, 7566.5532227
1: -1536.5211182, 1629.4772949, -4837.9506836, 5417.9882812, -6501.0156250, 6366.0620117
2: -2288.2800293, 1761.8527832, -7351.3964844, 5821.0444336, -7573.9853516, 8862.9960938
3: -861.5756836, 2305.8876953, -2694.4814453, 7403.6201172, -8055.1147461, 4807.2338867
4: -2515.1354980, 1718.1398926, -8027.7475586, 5639.4873047, -7610.0693359, 9498.4414062

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9797584, upper bound: 3611.8990932
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1927.2519531, 1677.4642334, -6011.9687500, 5483.9355469, -6873.7343750, 7579.7202148
1: -1553.9323730, 1642.8839111, -4836.6679688, 5417.8041992, -6515.7792969, 6379.3427734
2: -2313.8024902, 1776.0273438, -7350.1191406, 5820.7797852, -7596.1127930, 8876.7216797
3: -869.4301147, 2330.9794922, -2693.7277832, 7402.8237305, -8060.7954102, 4829.3471680
4: -2542.1501465, 1730.0141602, -8026.0302734, 5639.1772461, -7634.3188477, 9511.0312500

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392670, upper bound: 3611.8997270
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4430272
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1908.9503174, 1659.8024902, -5926.3828125, 5408.7832031, -6783.0917969, 7476.5805664
1: -1539.3900146, 1625.3630371, -4768.8032227, 5343.9389648, -6429.7514648, 6293.7036133
2: -2292.6147461, 1755.8453369, -7252.9438477, 5742.1274414, -7499.2841797, 8757.2607422
3: -860.1731567, 2308.4780273, -2654.8325195, 7302.0317383, -7949.2416992, 4770.2065430
4: -2518.7443848, 1709.7966309, -7916.2519531, 5560.1918945, -7536.0698242, 9379.4658203

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
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
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.9573009, upper bound: 3612.7879236
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1893186, upper bound: 3613.0448736
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6194.5292969, 5654.6108398, -6142.9277344, 5621.5571289, -11011.1455078, 11001.6953125
1: -4985.2236328, 5586.3261719, -4944.8652344, 5553.7827148, -9852.8388672, 9853.3964844
2: -7574.4887695, 6002.5395508, -7518.7475586, 5968.9780273, -12570.5976562, 12556.6064453
3: -2774.6374512, 7633.3090820, -2754.0969238, 7580.2089844, -9853.5693359, 9886.8593750
4: -8269.8330078, 5814.9877930, -8204.9492188, 5780.0385742, -13067.8906250, 13044.2705078

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.0915166, upper bound: 3612.7867989
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3294335, upper bound: 3613.0442059
time: 0.81 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.31 seconds
NS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.31
Output dim: 0, lower bound: -3612.9797584, upper bound: 3611.8990932
NS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 0, lower bound: -3613.3392670, upper bound: 3611.8997270
NS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4430272
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.31
Output dim: 0, lower bound: -3612.9573009, upper bound: 3612.7879236
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.31
Output dim: 0, lower bound: -3613.1893186, upper bound: 3613.0448736
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.31
Output dim: 0, lower bound: -3613.0915166, upper bound: 3612.7867989
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 0, lower bound: -3613.3294335, upper bound: 3613.0442059

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1906.5887451, 1658.0129395, -5972.6826172, 5458.7021484, -6816.0371094, 7521.7705078
1: -1536.5211182, 1629.4772949, -4805.0092773, 5392.6352539, -6463.4379883, 6330.0585938
2: -2288.2800293, 1761.8527832, -7305.8720703, 5794.1547852, -7531.8271484, 8810.7333984
3: -861.5756836, 2305.8876953, -2677.9885254, 7360.7451172, -8005.9750977, 4785.2207031
4: -2515.1354980, 1718.1398926, -7975.8505859, 5611.9282227, -7568.8808594, 9440.1103516

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
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

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1927.2519531, 1677.4642334, -5962.2563477, 5446.1845703, -6826.8876953, 7525.3388672
1: -1553.9323730, 1642.8839111, -4796.3994141, 5381.3041992, -6471.3183594, 6335.3706055
2: -2313.8024902, 1776.0273438, -7293.5063477, 5781.2255859, -7547.1860352, 8812.9082031
3: -869.4301147, 2330.9794922, -2672.8046875, 7347.2553711, -7999.9506836, 4804.8232422
4: -2542.1501465, 1730.0141602, -7962.8725586, 5599.6152344, -7585.7583008, 9440.1455078

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
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
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392668, upper bound: 3611.8997270
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392668, upper bound: 3611.8997276
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1927.2519531, 1677.4642334, -5984.2934570, 5469.8530273, -6846.4091797, 7548.8720703
1: -1553.9323730, 1642.8839111, -4814.4882812, 5403.5644531, -6490.2485352, 6354.5307617
2: -2313.8024902, 1776.0273438, -7320.1645508, 5805.9580078, -7567.1269531, 8840.8212891
3: -869.4301147, 2330.9794922, -2683.3947754, 7375.7807617, -8028.1337891, 4813.9477539
4: -2542.1501465, 1730.0141602, -7991.3173828, 5623.3818359, -7605.8686523, 9470.7236328

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
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
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4337388
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4337388
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6176.4604492, 5639.0312500, -6149.6850586, 5628.9096680, -10997.4501953, 10991.5029297
1: -4970.7480469, 5571.0219727, -4951.0434570, 5561.4536133, -9843.7753906, 9842.1689453
2: -7553.6025391, 5985.9218750, -7527.7724609, 5976.0888672, -12554.2382812, 12546.6171875
3: -2766.2797852, 7611.9672852, -2757.1350098, 7591.1567383, -9852.6757812, 9867.3593750
4: -8246.9306641, 5798.9204102, -8215.1289062, 5788.8032227, -13049.3779297, 13035.9404297

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
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

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8915350, upper bound: 3612.7814389
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3611.8915350, upper bound: 3613.0441824
time: 0.87 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.66 seconds
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3612.9797584, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3613.3392668, upper bound: 3611.8997270
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3613.3392668, upper bound: 3611.8997276
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4337388
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.66
Output dim: 0, lower bound: -3613.3392670, upper bound: 3613.4337388
NS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.66
Output dim: 0, lower bound: -3611.8915350, upper bound: 3612.7814389
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.66
Output dim: 0, lower bound: -3611.8915350, upper bound: 3613.0441824

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1957.8157959, 1701.1000977, -5972.7197266, 5458.0380859, -6865.7744141, 7561.7973633
1: -1577.3613281, 1670.7596436, -4805.0869141, 5391.8134766, -6503.1669922, 6367.9614258
2: -2346.6181641, 1806.2084961, -7305.9282227, 5793.4829102, -7588.7270508, 8850.5869141
3: -884.1437378, 2366.0300293, -2677.8093262, 7360.6054688, -8026.5522461, 4844.4067383
4: -2580.5910645, 1761.1538086, -7975.8066406, 5611.2451172, -7632.5971680, 9478.1826172

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.2768195
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.4421356
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1856.4094238, 1608.3928223, -5962.0034180, 5449.4023438, -6757.9550781, 7465.7573242
1: -1496.7210693, 1578.8179932, -4796.4736328, 5383.1943359, -6415.2856445, 6275.3071289
2: -2230.0136719, 1707.1827393, -7293.1879883, 5783.9296875, -7464.5400391, 8748.7636719
3: -836.8752441, 2243.5935059, -2673.3107910, 7348.0410156, -7970.6513672, 4719.2299805
4: -2450.8525391, 1663.9034424, -7961.8325195, 5601.8320312, -7495.5092773, 9377.7558594

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.2768195
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.4421356
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1976.4196777, 1719.2852783, -5962.6796875, 5445.8823242, -6875.0195312, 7564.3105469
1: -1593.0754395, 1682.9068604, -4796.7939453, 5380.8310547, -6509.6450195, 6371.8315430
2: -2369.6191406, 1820.1040039, -7294.0507812, 5780.9140625, -7601.8535156, 8851.7382812
3: -891.3933716, 2388.5690918, -2672.7851562, 7347.6376953, -8020.4331055, 4861.6420898
4: -2604.9167480, 1773.1571045, -7963.3593750, 5599.2749023, -7647.1674805, 9477.5439453

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3381040, upper bound: 3611.8599500
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3392486, upper bound: 3611.8996457
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1878.8485107, 1630.7659912, -5952.0195312, 5437.3164062, -6770.9624023, 7471.4506836
1: -1515.4637451, 1596.6520996, -4788.2011719, 5372.2680664, -6424.8310547, 6282.9091797
2: -2257.5788574, 1724.0441895, -7281.3413086, 5771.4213867, -7482.2636719, 8753.4658203
3: -845.8417969, 2271.2236328, -2668.3024902, 7335.0883789, -7965.8959961, 4741.5532227
4: -2480.0874023, 1678.2492676, -7949.4248047, 5589.9218750, -7514.9057617, 9380.2109375

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3241118, upper bound: 3611.8699596
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3247447, upper bound: 3611.8970446
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1976.4196777, 1719.2852783, -5984.6806641, 5469.5502930, -6894.5336914, 7587.8105469
1: -1593.0754395, 1682.9068604, -4814.8579102, 5403.0849609, -6528.5585938, 6390.9687500
2: -2369.6191406, 1820.1040039, -7320.6679688, 5805.6479492, -7621.7846680, 8879.6142578
3: -891.3933716, 2388.5690918, -2683.3593750, 7376.1250000, -8048.5810547, 4870.7397461
4: -2604.9167480, 1773.1571045, -7991.7563477, 5623.0473633, -7667.2744141, 9508.0781250

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.2775144
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.4331294
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1878.8485107, 1630.7659912, -5973.9335938, 5460.8593750, -6790.3916016, 7494.8701172
1: -1515.4637451, 1596.6520996, -4806.1992188, 5394.4013672, -6443.6552734, 6301.9868164
2: -2257.5788574, 1724.0441895, -7307.8554688, 5796.0258789, -7502.1044922, 8781.2470703
3: -845.8417969, 2271.2236328, -2678.8403320, 7363.4663086, -7993.9448242, 4750.6328125
4: -2480.0874023, 1678.2492676, -7977.7036133, 5613.5761719, -7534.9345703, 9410.6376953

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.2775144
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.4331293
time: 0.86 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.69 seconds
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 3.69
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.2768195
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.69
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.2768195
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3612.8126902, upper bound: 3613.4421356
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3381040, upper bound: 3611.8599500
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3392486, upper bound: 3611.8996457
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3241118, upper bound: 3611.8699596
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3247447, upper bound: 3611.8970446
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.2775144
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.4331294
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.2775144
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.69
Output dim: 0, lower bound: -3613.3424209, upper bound: 3613.4331293

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1957.8157959, 1701.1000977, -5953.2485352, 5442.5351562, -6845.8989258, 7540.7739258
1: -1577.3613281, 1670.7596436, -4789.3535156, 5376.6708984, -6484.0756836, 6350.6865234
2: -2346.6181641, 1806.2084961, -7283.2934570, 5777.0112305, -7568.0029297, 8825.2246094
3: -884.1437378, 2366.0300293, -2669.3625488, 7338.2714844, -8001.3232422, 4834.9765625
4: -2580.5910645, 1761.1538086, -7950.7041016, 5594.9833984, -7611.9267578, 9450.1728516

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8652763, upper bound: 3613.4886042
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0028747, upper bound: 3613.4899085
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1856.4094238, 1608.3928223, -5942.4541016, 5433.8730469, -6738.0512695, 7444.6542969
1: -1496.7210693, 1578.8179932, -4780.6728516, 5368.0307617, -6396.1689453, 6257.9624023
2: -2230.0136719, 1707.1827393, -7270.4487305, 5767.4194336, -7443.7739258, 8723.2988281
3: -836.8752441, 2243.5935059, -2664.8317871, 7325.6083984, -7945.3208008, 4709.7612305
4: -2450.8525391, 1663.9034424, -7936.6279297, 5585.5312500, -7474.7934570, 9349.6474609

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8126106, upper bound: 3613.4409999
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.8126106, upper bound: 3613.4421046
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1961.3797607, 1705.5402832, -5892.1308594, 5384.8012695, -6792.5288086, 7475.4155273
1: -1581.0527344, 1669.5144043, -4740.7509766, 5320.6679688, -6431.9179688, 6298.6103516
2: -2351.8120117, 1805.6251221, -7212.1049805, 5714.8110352, -7511.2119141, 8748.3720703
3: -884.3420410, 2370.2119141, -2638.9731445, 7264.5966797, -7925.7944336, 4806.5234375
4: -2585.4350586, 1759.0778809, -7873.1728516, 5535.5869141, -7557.6269531, 9365.6982422

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3220295, upper bound: 3611.8599769
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3220295, upper bound: 3611.8600427
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1976.4196777, 1719.2852783, -5931.5019531, 5419.9721680, -6846.3198242, 7531.6352539
1: -1593.0754395, 1682.9068604, -4771.9150391, 5355.3476562, -6481.6494141, 6345.6679688
2: -2369.6191406, 1820.1040039, -7257.8916016, 5753.3730469, -7571.3715820, 8813.2255859
3: -891.3933716, 2388.5690918, -2658.5563965, 7310.9038086, -7981.9306641, 4846.6020508
4: -2604.9167480, 1773.1571045, -7923.5004883, 5572.4252930, -7617.6367188, 9434.9931641

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3234210, upper bound: 3611.8996755
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3234210, upper bound: 3611.8997456
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1872.9963379, 1625.8043213, -5971.2470703, 5460.7587891, -6778.2856445, 7483.7128906
1: -1510.7347412, 1591.8133545, -4803.9531250, 5394.9868164, -6434.1577148, 6291.6274414
2: -2250.6926270, 1718.8409424, -7306.3486328, 5796.5532227, -7489.5449219, 8768.8632812
3: -843.2492676, 2264.2958984, -2677.9829102, 7360.6215820, -7985.4477539, 4741.1992188
4: -2472.5434570, 1673.1947021, -7976.0307617, 5613.9536133, -7520.5742188, 9397.7880859

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
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
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3226210, upper bound: 3611.8699038
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3240990, upper bound: 3611.8698932
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1878.8485107, 1630.7659912, -5946.5322266, 5432.7866211, -6764.9248047, 7465.3476562
1: -1515.4637451, 1596.6520996, -4783.7827148, 5367.8359375, -6419.1093750, 6277.9946289
2: -2257.5788574, 1724.0441895, -7275.0097656, 5766.6616211, -7475.8969727, 8746.1552734
3: -845.8417969, 2271.2236328, -2665.9304199, 7328.6914062, -7958.8208008, 4738.5107422
4: -2480.0874023, 1678.2492676, -7942.4106445, 5585.2407227, -7508.6416016, 9372.1826172

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3227861, upper bound: 3611.8969732
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3247233, upper bound: 3611.8969635
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1976.4196777, 1719.2852783, -5717.1591797, 5246.0126953, -6643.7797852, 7305.6416016
1: -1593.0754395, 1682.9068604, -4601.1728516, 5182.2436523, -6282.8496094, 6165.2563477
2: -2369.6191406, 1820.1040039, -7012.0126953, 5567.7451172, -7354.7060547, 8548.2812500
3: -891.3933716, 2388.5690918, -2563.1135254, 7062.0043945, -7718.7275391, 4739.6821289
4: -2604.9167480, 1773.1571045, -7648.9506836, 5390.3374023, -7407.5410156, 9140.6718750

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2348943, upper bound: 3613.1084311
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.2830502
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1976.4196777, 1719.2852783, -5965.1889648, 5454.0034180, -6874.6210938, 7566.7666016
1: -1593.0754395, 1682.9068604, -4799.1083984, 5387.8994141, -6509.4301758, 6373.6762695
2: -2369.6191406, 1820.1040039, -7298.0102539, 5789.1445312, -7601.0415039, 8854.2265625
3: -891.3933716, 2388.5690918, -2674.9267578, 7353.7553711, -8023.3198242, 4861.3339844
4: -2604.9167480, 1773.1571045, -7966.6274414, 5606.7480469, -7646.5878906, 9480.0429688

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.2348943, upper bound: 3613.4306332
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.4905125
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1878.8485107, 1630.7659912, -5706.6538086, 5237.4824219, -6539.7485352, 7212.9365234
1: -1515.4637451, 1596.6520996, -4592.6572266, 5173.7441406, -6198.1093750, 6076.4204102
2: -2257.5788574, 1724.0441895, -6999.4560547, 5558.3076172, -7235.1889648, 8450.1699219
3: -845.8417969, 2271.2236328, -2558.6945801, 7049.5375977, -7664.4189453, 4619.6655273
4: -2480.0874023, 1678.2492676, -7635.2114258, 5381.0244141, -7275.3491211, 9043.5439453

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3390713, upper bound: 3613.2703137
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3423772, upper bound: 3613.2773673
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1878.8485107, 1630.7659912, -5954.3881836, 5445.2900391, -6770.4575195, 7473.7724609
1: -1515.4637451, 1596.6520996, -4790.4023438, 5379.1953125, -6424.5083008, 6284.6484375
2: -2257.5788574, 1724.0441895, -7285.1235352, 5779.4921875, -7481.3315430, 8755.7919922
3: -845.8417969, 2271.2236328, -2670.3874512, 7341.0288086, -7968.6152344, 4741.2050781
4: -2480.0874023, 1678.2492676, -7952.5058594, 5597.2460938, -7514.2138672, 9382.5351562

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3390713, upper bound: 3613.4287043
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.4330401
time: 0.89 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 3.74 seconds
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3612.8652763, upper bound: 3613.4886042
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.0028747, upper bound: 3613.4899085
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3612.8126106, upper bound: 3613.4409999
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3612.8126106, upper bound: 3613.4421046
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3220295, upper bound: 3611.8599769
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3220295, upper bound: 3611.8600427
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3234210, upper bound: 3611.8996755
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3234210, upper bound: 3611.8997456
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3226210, upper bound: 3611.8699038
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3240990, upper bound: 3611.8698932
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3227861, upper bound: 3611.8969732
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3247233, upper bound: 3611.8969635
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.2348943, upper bound: 3613.1084311
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.2830502
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.2348943, upper bound: 3613.4306332
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.4905125
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3390713, upper bound: 3613.2703137
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3423772, upper bound: 3613.2773673
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3390713, upper bound: 3613.4287043
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.74
Output dim: 0, lower bound: -3613.3423773, upper bound: 3613.4330401

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1896.3950195, 1645.5610352, -5930.2495117, 5422.8369141, -6764.0292969, 7462.2675781
1: -1528.4937744, 1616.5988770, -4770.9042969, 5357.2446289, -6415.0180664, 6278.2270508
2: -2274.6560059, 1747.3880615, -7256.1035156, 5755.7045898, -7473.9091797, 8738.6054688
3: -854.7360840, 2291.9133301, -2658.6887207, 7310.9223633, -7943.9047852, 4749.9091797
4: -2501.5344238, 1703.7141113, -7920.8994141, 5574.2939453, -7511.5942383, 9362.0498047

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6036710, upper bound: 3612.9718273
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0163912, upper bound: 3613.3900995
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1929.3913574, 1676.6303711, -5949.1665039, 5439.2021484, -6814.7099609, 7512.7636719
1: -1554.7067871, 1646.7808838, -4786.0620117, 5373.3486328, -6458.5776367, 6323.8164062
2: -2313.6779785, 1779.9383545, -7278.3457031, 5773.3691406, -7532.0566406, 8794.5126953
3: -870.8229370, 2332.1132812, -2667.6210938, 7333.4375000, -7983.4985352, 4799.6455078
4: -2544.4128418, 1736.0589600, -7945.2875977, 5591.3862305, -7572.8520508, 9420.2539062

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
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
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.6036710, upper bound: 3613.0384054
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0163912, upper bound: 3613.3938050
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1821.0299072, 1573.3370361, -5920.4028320, 5414.7744141, -6682.6455078, 7386.9604492
1: -1468.6983643, 1545.0521240, -4762.9902344, 5349.2187500, -6348.5263672, 6205.9370117
2: -2188.3640137, 1670.3804932, -7244.4418945, 5746.8154297, -7380.5258789, 8659.3808594
3: -819.7427979, 2199.3959961, -2654.5183105, 7299.2836914, -7900.0698242, 4654.8911133
4: -2404.9475098, 1627.8707275, -7908.1284180, 5565.5693359, -7408.1147461, 9283.5322266

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.5031629, upper bound: 3612.8438198
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9723709, upper bound: 3613.3369331
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1831.4421387, 1587.4954834, -5937.5986328, 5429.8081055, -6709.5537109, 7418.8627930
1: -1476.8382568, 1557.8118896, -4776.7963867, 5363.9658203, -6372.6958008, 6233.5273438
2: -2201.0939941, 1684.3861084, -7264.6372070, 5762.9838867, -7411.0717773, 8695.2636719
3: -824.7568970, 2213.7043457, -2662.7343750, 7319.8227539, -7928.1997070, 4678.0903320
4: -2419.1140137, 1642.0966797, -7930.2587891, 5581.2216797, -7439.4399414, 9322.0185547

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
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
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.4551416, upper bound: 3612.8454913
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3612.9364483, upper bound: 3613.3380144
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1913.1997070, 1662.6721191, -5887.0849609, 5380.8896484, -6741.2470703, 7429.1147461
1: -1542.7233887, 1627.6203613, -4736.6630859, 5316.7827148, -6390.3471680, 6254.4501953
2: -2295.4938965, 1759.7990723, -7205.9707031, 5710.5405273, -7451.7490234, 8698.5634766
3: -861.1240234, 2312.4680176, -2636.8488770, 7258.6699219, -7897.2197266, 4747.6113281
4: -2523.5541992, 1714.2946777, -7866.4653320, 5531.3417969, -7492.6079102, 9316.0195312

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.6437495
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8599769
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5890.2016602, 5383.2812500, -6778.7172852, 7463.1650391
1: -1571.0087891, 1659.4431152, -4739.1962891, 5319.1538086, -6420.5913086, 6287.1723633
2: -2337.5932617, 1794.0677490, -7209.7861328, 5713.1420898, -7495.6318359, 8735.0888672
3: -878.3445435, 2355.6574707, -2638.1594238, 7262.3583984, -7917.5395508, 4791.2778320
4: -2569.6921387, 1748.0450439, -7870.6406250, 5533.9326172, -7540.6157227, 9352.8085938

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.6438153
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8600231
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1913.1997070, 1662.6721191, -5924.2661133, 5414.3823242, -6776.7470703, 7468.4409180
1: -1542.7233887, 1627.6203613, -4766.0615234, 5349.7724609, -6424.9985352, 6285.5000000
2: -2295.4938965, 1759.7990723, -7249.0903320, 5747.2626953, -7490.2304688, 8744.9785156
3: -861.1240234, 2312.4680176, -2655.5305176, 7302.3994141, -7943.1933594, 4767.1674805
4: -2523.5541992, 1714.2946777, -7913.8515625, 5566.3447266, -7529.5356445, 9367.0410156

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.8699231
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8969931
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5928.6816406, 5417.8046875, -6816.9052734, 7505.3706055
1: -1571.0087891, 1659.4431152, -4769.6445312, 5353.1850586, -6457.7387695, 6320.5058594
2: -2337.5932617, 1794.0677490, -7254.4858398, 5750.9741211, -7537.4023438, 8784.7958984
3: -878.3445435, 2355.6574707, -2657.3918457, 7307.6318359, -7965.8647461, 4812.7910156
4: -2569.6921387, 1748.0450439, -7919.7563477, 5570.0375977, -7580.5136719, 9407.2158203

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.8699926
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8970628
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1830.9565430, 1585.6020508, -5947.0468750, 5439.4716797, -6714.4150391, 7418.7060547
1: -1477.4514160, 1552.6820068, -4784.6035156, 5374.0024414, -6379.4243164, 6232.8032227
2: -2201.3454590, 1676.2984619, -7277.7827148, 5773.6210938, -7416.7524414, 8696.5156250
3: -823.0755005, 2212.5537109, -2666.7568359, 7331.5927734, -7934.6220703, 4678.0083008
4: -2418.3100586, 1631.6048584, -7944.7314453, 5591.7426758, -7443.7568359, 9323.2685547

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.6437336
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.8698932
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1848.2148438, 1604.5887451, -5965.7949219, 5456.1118164, -6749.4150391, 7457.5991211
1: -1491.0156250, 1571.0233154, -4799.6142578, 5390.3564453, -6410.3164062, 6267.0771484
2: -2222.0197754, 1695.5529785, -7299.8559570, 5791.5219727, -7456.5087891, 8740.2138672
3: -831.0613403, 2234.7067871, -2675.6413574, 7354.0439453, -7967.5859375, 4709.5859375
4: -2441.0344238, 1650.9539795, -7968.9106445, 5609.0703125, -7484.8564453, 9369.5732422

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.6437336
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.8698932
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5922.8681641, 5412.1459961, -6702.1152344, 7401.3613281
1: -1482.5800781, 1557.9315186, -4764.8437500, 5347.4912109, -6365.3417969, 6220.0659180
2: -2208.8120117, 1681.9722900, -7247.0825195, 5744.3745117, -7404.2695312, 8674.9716797
3: -825.8901978, 2220.0764160, -2654.9204102, 7300.3891602, -7908.9345703, 4676.1425781
4: -2426.4777832, 1637.0947266, -7911.8076172, 5563.6962891, -7433.0385742, 9298.8085938

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3216877, upper bound: 3611.8599500
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3216878, upper bound: 3611.8969640
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5941.2939453, 5428.3750000, -6737.1235352, 7440.1777344
1: -1496.4664307, 1576.5871582, -4779.6166992, 5363.4384766, -6396.1704102, 6254.3125000
2: -2229.9545898, 1701.5405273, -7268.7661133, 5761.8637695, -7444.0932617, 8718.5078125
3: -834.0612793, 2242.6994629, -2663.6699219, 7322.4184570, -7941.6357422, 4708.0283203
4: -2449.7573242, 1656.7730713, -7935.5722656, 5580.6113281, -7474.2910156, 9344.9824219

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
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
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3236144, upper bound: 3611.8599500
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3236144, upper bound: 3611.8969640
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5713.2500000, 5242.8085938, -6613.4941406, 7278.3413086
1: -1571.0087891, 1659.4431152, -4598.0102539, 5179.0527344, -6258.0610352, 6139.2495117
2: -2337.5932617, 1794.0677490, -7007.2666016, 5564.2500000, -7319.8188477, 8518.6035156
3: -878.3445435, 2355.6574707, -2561.4477539, 7057.3510742, -7701.3959961, 4705.4379883
4: -2569.6921387, 1748.0450439, -7643.7587891, 5386.8740234, -7369.5356445, 9111.5419922

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
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

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2008038, upper bound: 3612.8457402
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3253953, upper bound: 3613.1734300
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1913.1997070, 1662.6721191, -5941.6967773, 5433.8505859, -6790.6059570, 7486.6831055
1: -1542.7233887, 1627.6203613, -4780.2578125, 5368.0048828, -6438.5405273, 6300.0053711
2: -2295.4938965, 1759.7990723, -7270.2016602, 5767.3496094, -7504.4799805, 8766.0058594
3: -861.1240234, 2312.4680176, -2664.0410156, 7325.7890625, -7964.5791016, 4774.2841797
4: -2523.5541992, 1714.2946777, -7936.1381836, 5585.5834961, -7543.6083984, 9390.1054688

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1810003, upper bound: 3612.8130812
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3117860, upper bound: 3613.3320611
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5961.0966797, 5450.6728516, -6844.2324219, 7539.2836914
1: -1571.0087891, 1659.4431152, -4795.8105469, 5384.5732422, -6484.5185547, 6347.5361328
2: -2337.5932617, 1794.0677490, -7293.0478516, 5785.5058594, -7566.0239258, 8824.3378906
3: -878.3445435, 2355.6574707, -2673.1875000, 7348.9184570, -8005.7680664, 4827.0234375
4: -2569.6921387, 1748.0450439, -7961.1943359, 5603.1542969, -7608.4609375, 9450.6728516

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2023555, upper bound: 3613.0284362
time: 0.94 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3264759, upper bound: 3613.3944043
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5684.9082031, 5218.5947266, -6478.4169922, 7150.8100586
1: -1482.5800781, 1557.9315186, -4575.2958984, 5155.1601562, -6145.8686523, 6020.0151367
2: -2208.8120117, 1681.9722900, -6973.7338867, 5537.9370117, -7165.1865234, 8381.1250000
3: -825.8901978, 2220.0764160, -2548.4875488, 7023.5229492, -7616.7402344, 4558.0732422
4: -2426.4777832, 1637.0947266, -7607.0327148, 5361.2753906, -7201.2880859, 8972.5273438

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.1985057, upper bound: 3612.7232779
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3213624, upper bound: 3613.1677401
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5702.3540039, 5233.9199219, -6512.7919922, 7188.7138672
1: -1496.4664307, 1576.5871582, -4589.2045898, 5170.1967773, -6176.0170898, 6053.4687500
2: -2229.9545898, 1701.5405273, -6994.2915039, 5554.4252930, -7204.3017578, 8423.6220703
3: -834.0612793, 2242.6994629, -2556.8532715, 7044.4096680, -7648.4140625, 4589.6064453
4: -2449.7573242, 1656.7730713, -7629.5595703, 5377.2314453, -7241.8696289, 9017.5498047

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
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
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2008041, upper bound: 3612.7641859
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3253953, upper bound: 3613.1715965
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5931.5717773, 5425.5996094, -6708.4501953, 7410.6455078
1: -1482.5800781, 1557.9315186, -4772.1049805, 5359.7900391, -6371.5512695, 6227.3764648
2: -2208.8120117, 1681.9722900, -7258.1674805, 5758.2456055, -7410.5751953, 8685.5927734
3: -825.8901978, 2220.0764160, -2659.7680664, 7313.8281250, -7919.8105469, 4679.2348633
4: -2426.4777832, 1637.0947266, -7922.9667969, 5576.6367188, -7439.4155273, 9310.2392578

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2019503, upper bound: 3612.8104746
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3244303, upper bound: 3613.3254119
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5949.9218750, 5441.6118164, -6743.4072266, 7449.3852539
1: -1496.4664307, 1576.5871582, -4786.8291016, 5375.5263672, -6402.3037109, 6261.5771484
2: -2229.9545898, 1701.5405273, -7279.7607422, 5775.4804688, -7450.3261719, 8729.0507812
3: -834.0612793, 2242.6994629, -2668.4809570, 7335.7280273, -7952.4067383, 4711.0844727
4: -2449.7573242, 1656.7730713, -7946.6313477, 5593.3315430, -7480.6245117, 9356.3222656

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.2023555, upper bound: 3612.8334776
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3264759, upper bound: 3613.3291709
time: 1.01 seconds

## Summary of splitting at layer (split count: 11)
- Time for NS candidates: 6.36 seconds
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.6036710, upper bound: 3612.9718273
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.0163912, upper bound: 3613.3900995
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.6036710, upper bound: 3613.0384054
NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.0163912, upper bound: 3613.3938050
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.5031629, upper bound: 3612.8438198
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.9723709, upper bound: 3613.3369331
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.4551416, upper bound: 3612.8454913
NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3612.9364483, upper bound: 3613.3380144
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.6437495
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8599769
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.6438153
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8600231
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.8699231
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8969931
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2499831, upper bound: 3611.8699926
NS_A1_B2_A1_B2_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3119032, upper bound: 3611.8970628
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.6437336
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.8698932
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.6437336
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2453302, upper bound: 3611.8698932
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3216877, upper bound: 3611.8599500
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3216878, upper bound: 3611.8969640
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3236144, upper bound: 3611.8599500
NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3236144, upper bound: 3611.8969640
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2008038, upper bound: 3612.8457402
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3253953, upper bound: 3613.1734300
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.1810003, upper bound: 3612.8130812
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3117860, upper bound: 3613.3320611
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2023555, upper bound: 3613.0284362
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3264759, upper bound: 3613.3944043
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.1985057, upper bound: 3612.7232779
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3213624, upper bound: 3613.1677401
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2008041, upper bound: 3612.7641859
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3253953, upper bound: 3613.1715965
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2019503, upper bound: 3612.8104746
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3244303, upper bound: 3613.3254119
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.2023555, upper bound: 3612.8334776
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.36
Output dim: 0, lower bound: -3613.3264759, upper bound: 3613.3291709

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1896.3950195, 1645.5610352, -5882.3144531, 5377.1093750, -6724.3022461, 7416.1376953
1: -1528.4937744, 1616.5988770, -4732.2802734, 5312.2836914, -6375.0883789, 6241.1059570
2: -2274.6560059, 1747.3880615, -7197.5317383, 5707.6176758, -7431.6474609, 8683.0888672
3: -854.7360840, 2291.9133301, -2637.1345215, 7250.7143555, -7886.4091797, 4730.1806641
4: -2501.5344238, 1703.7141113, -7857.3662109, 5527.8681641, -7471.0307617, 9301.5761719

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
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
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1929.3913574, 1676.6303711, -5900.3613281, 5392.6538086, -6774.1411133, 7465.7402344
1: -1554.7067871, 1646.7808838, -4746.7661133, 5327.5732422, -6417.8217773, 6285.9912109
2: -2313.6779785, 1779.9383545, -7218.7910156, 5724.4135742, -7488.9111328, 8737.9619141
3: -870.8229370, 2332.1132812, -2645.6772461, 7272.1450195, -7924.9023438, 4779.5175781
4: -2544.4128418, 1736.0589600, -7880.6723633, 5544.1557617, -7531.4384766, 9358.6503906

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3611.9723819, upper bound: 3613.3737642
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.0079973, upper bound: 3613.3737641
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1821.0299072, 1573.3370361, -5871.1049805, 5367.6875000, -6641.6557617, 7339.4863281
1: -1468.6983643, 1545.0521240, -4723.3061523, 5302.9111328, -6307.3432617, 6167.7568359
2: -2188.3640137, 1670.3804932, -7184.2783203, 5697.2954102, -7336.9326172, 8602.2910156
3: -819.7427979, 2199.3959961, -2632.3569336, 7237.3291016, -7840.8798828, 4634.5815430
4: -2404.9475098, 1627.8707275, -7842.8647461, 5517.8422852, -7366.2993164, 9221.3525391

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1831.4421387, 1587.4954834, -5888.0385742, 5382.4785156, -6668.2895508, 7371.0996094
1: -1476.8382568, 1557.8118896, -4736.9072266, 5317.4179688, -6331.2441406, 6195.1171875
2: -2201.0939941, 1684.3861084, -7204.1928711, 5713.2138672, -7367.1977539, 8637.8505859
3: -824.7568970, 2213.7043457, -2640.4560547, 7257.5556641, -7868.6694336, 4657.6508789
4: -2419.1140137, 1642.0966797, -7864.6782227, 5533.2392578, -7397.3364258, 9259.4726562

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5869.4741211, 5366.0004883, -6650.9799805, 7344.9453125
1: -1482.5800781, 1557.9315186, -4722.5532227, 5302.0478516, -6315.5996094, 6175.3183594
2: -2208.8120117, 1681.9722900, -7185.3364258, 5694.5092773, -7349.3452148, 8608.4980469
3: -825.8901978, 2220.0764160, -2629.3134766, 7237.6201172, -7842.7089844, 4648.6206055
4: -2426.4777832, 1637.0947266, -7843.7773438, 5515.6455078, -7380.0595703, 9225.5478516

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5907.0810547, 5400.0292969, -6687.0034180, 7384.7255859
1: -1482.5800781, 1557.9315186, -4752.2900391, 5335.5678711, -6350.7602539, 6206.7294922
2: -2208.8120117, 1681.9722900, -7228.9370117, 5731.7465820, -7388.3344727, 8655.4414062
3: -825.8901978, 2220.0764160, -2648.1701660, 7282.0122070, -7889.3442383, 4668.3769531
4: -2426.4777832, 1637.0947266, -7891.7094727, 5551.1943359, -7417.5151367, 9277.1621094

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 39

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5872.5048828, 5368.5581055, -6670.7021484, 7366.0034180
1: -1496.4664307, 1576.5871582, -4725.0004883, 5304.5527344, -6331.5332031, 6195.3759766
2: -2229.9545898, 1701.5405273, -7189.0068359, 5697.2128906, -7372.3779297, 8631.1484375
3: -834.0612793, 2242.6994629, -2630.6220703, 7241.3305664, -7855.6088867, 4671.7631836
4: -2449.7573242, 1656.7730713, -7847.7763672, 5518.3139648, -7405.3320312, 9248.9785156

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.8600054, upper bound: 3611.8588550
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.8600054, upper bound: 3611.8599500
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5911.5351562, 5403.7729492, -6709.5410156, 7408.8720703
1: -1496.4664307, 1576.5871582, -4755.8833008, 5339.2675781, -6369.3212891, 6229.2509766
2: -2229.9545898, 1701.5405273, -7234.3305664, 5735.7177734, -7414.8110352, 8681.6318359
3: -834.0612793, 2242.6994629, -2650.0883789, 7287.4345703, -7904.8100586, 4693.5820312
4: -2449.7573242, 1656.7730713, -7897.5908203, 5555.1191406, -7445.9145508, 9304.2177734

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.8600054, upper bound: 3611.8958713
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3610.8600054, upper bound: 3611.8968363
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5667.0209961, 5198.6635742, -6575.2084961, 7234.0346680
1: -1571.0087891, 1659.4431152, -4560.6538086, 5135.6645508, -6219.6562500, 6103.5200195
2: -2337.5932617, 1794.0677490, -6950.5913086, 5517.8608398, -7279.2050781, 8465.1669922
3: -878.3445435, 2355.6574707, -2540.7402344, 6999.0830078, -7645.9663086, 4686.5678711
4: -2569.6921387, 1748.0450439, -7582.3090820, 5342.1196289, -7330.5463867, 9053.3564453

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8681786, upper bound: 3613.1008742
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3613.3113564, upper bound: 3613.1533948
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1913.1997070, 1662.6721191, -5896.0371094, 5390.3339844, -6752.9023438, 7442.8369141
1: -1542.7233887, 1627.6203613, -4743.4541016, 5325.2011719, -6400.6137695, 6264.7153320
2: -2295.4938965, 1759.7990723, -7214.3129883, 5721.5737305, -7464.3530273, 8713.1835938
3: -861.1240234, 2312.4680176, -2643.5715332, 7268.4545898, -7909.9047852, 4755.5917969
4: -2523.5541992, 1714.2946777, -7875.5307617, 5541.4135742, -7505.1010742, 9332.5781250

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1948.8031006, 1695.5235596, -5915.2104492, 5406.9853516, -6806.2973633, 7495.1928711
1: -1571.0087891, 1659.4431152, -4758.8427734, 5341.6245117, -6446.3916016, 6312.0590820
2: -2337.5932617, 1794.0677490, -7236.9262695, 5739.5639648, -7525.6630859, 8771.2451172
3: -878.3445435, 2355.6574707, -2652.6228027, 7291.3081055, -7950.8071289, 4808.2119141
4: -2569.6921387, 1748.0450439, -7900.3276367, 5558.8525391, -7569.7187500, 9392.8496094

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
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

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3612.8684860, upper bound: 3613.2805701
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3613.3122505, upper bound: 3613.3743653
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5637.6552734, 5173.4028320, -6439.2529297, 7105.5146484
1: -1482.5800781, 1557.9315186, -4537.1323242, 5110.7045898, -6106.5283203, 5983.5014648
2: -2208.8120117, 1681.9722900, -6915.8447266, 5490.4150391, -7123.5898438, 8326.5292969
3: -825.8901978, 2220.0764160, -2527.3225098, 6963.9643555, -7560.0942383, 4538.7773438
4: -2426.4777832, 1637.0947266, -7544.2661133, 5315.4912109, -7161.4238281, 8913.0810547

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5655.0708008, 5188.7363281, -6473.5629883, 7143.3642578
1: -1496.4664307, 1576.5871582, -4551.0185547, 5125.7797852, -6136.6503906, 6016.9106445
2: -2229.9545898, 1701.5405273, -6936.3823242, 5506.9350586, -7162.6611328, 8368.9628906
3: -834.0612793, 2242.6994629, -2535.6591797, 6984.8115234, -7591.7026367, 4570.2641602
4: -2449.7573242, 1656.7730713, -7566.7670898, 5331.4633789, -7201.9399414, 8958.0361328

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1837.3029785, 1590.9776611, -5884.5180664, 5380.7197266, -6669.4711914, 7365.4199219
1: -1482.5800781, 1557.9315186, -4734.2158203, 5315.6357422, -6332.3579102, 6190.9990234
2: -2208.8120117, 1681.9722900, -7200.6518555, 5711.0253906, -7369.0942383, 8631.1601562
3: -825.8901978, 2220.0764160, -2638.6669922, 7254.7192383, -7863.4101562, 4659.9301758
4: -2426.4777832, 1637.0947266, -7860.5927734, 5531.1557617, -7399.6479492, 9250.9628906

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1854.9573975, 1610.3144531, -5902.8925781, 5396.8095703, -6704.4272461, 7404.1586914
1: -1496.4664307, 1576.5871582, -4748.9643555, 5331.4731445, -6363.1381836, 6225.2016602
2: -2229.9545898, 1701.5405273, -7222.3012695, 5728.3569336, -7408.8525391, 8674.6308594
3: -834.0612793, 2242.6994629, -2647.3891602, 7276.6679688, -7896.0273438, 4691.7592773
4: -2449.7573242, 1656.7730713, -7884.3105469, 5547.9375000, -7440.8505859, 9297.0527344

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
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

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.88 + 356.33 = 360.21 seconds
