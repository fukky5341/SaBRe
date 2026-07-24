## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 743.673742927666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447)
1: (-226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553)
2: (-160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006)
3: (-390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531)
4: (-263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049)

## BASE Result
execution time: IAR + LP analysis = 1.80 + 1.91 = 3.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -743.6896151, upper bound: 743.6896151


# Binary Search by BASE starts (time budget: 1196.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=860.0533447265625
rel_dist={0: [-743.6895975863799, 743.6895975863799]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=860.0533447265625
rel_dist={0: [-743.6893068054288, 743.6893068054287]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=860.0533447265625
rel_dist={0: [-743.6887393664056, 743.6887393664056]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=860.0533447265625
rel_dist={0: [-743.6883248579472, 743.6883248579472]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=860.0533447265625
rel_dist={0: [-743.6880449320028, 743.6880449320029]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=860.0533447265625
rel_dist={0: [-743.6877737489342, 743.6877737489342]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=860.0533447265625
rel_dist={0: [-743.6876008715927, 743.6876008715928]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=860.0533447265625
rel_dist={0: [-743.6875013877294, 743.6875013877293]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=860.0533447265625
rel_dist={0: [-743.6874472022885, 743.6874472022885]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=860.0533447265625
rel_dist={0: [-743.6874172086395, 743.6874172086395]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=860.0533447265625
rel_dist={0: [-743.6874005417767, 743.6874005417767]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=860.0533447265625
rel_dist={0: [-743.6873920794325, 743.6873920794324]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=860.0533447265625
rel_dist={0: [-743.6873878482721, 743.687387848272]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=860.0533447265625
rel_dist={0: [-743.6873857327159, 743.6873857327159]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=860.0533447265625
rel_dist={0: [-743.6873846764196, 743.6873846749841]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=860.0533447265625
rel_dist={0: [-743.6873841475402, 743.6873841492904]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=860.0533447265625
rel_dist={0: [-743.6873838833546, 743.6873838817382]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=860.0533447265625
rel_dist={0: [-743.6873837518972, 743.6873837516682]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=860.0533447265625
rel_dist={0: [-743.6873836840066, 743.6873836867503]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=860.0533447265625
rel_dist={0: [-743.6873836604632, 743.6873836596296]}

## Binary Search Result
Binary search time: 75.70 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1120.60 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6832858, upper bound: 743.6871528
time: 0.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 0, lower bound: -743.6832858, upper bound: 743.6871528
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -138.5269012, 721.5264282, -850.1054688, 808.8294678
1: -210.4309387, 796.4016113, -226.4326935, 857.1390381, -1067.5698242, 1022.8342896
2: -148.6245575, 824.4069214, -160.1910706, 887.5496826, -1036.1741943, 984.5979614
3: -362.4870300, 699.0101929, -390.1859741, 752.6910400, -1115.1779785, 1089.1961670
4: -244.7495728, 707.2666626, -263.8327942, 761.5472412, -1006.2968140, 971.0994263

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655
time: 0.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655
time: 0.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -250.1162415, 1337.2148438, -136.9550476, 714.2769775, -964.3931885, 1474.1699219
1: -412.2479858, 1588.0107422, -223.8997040, 848.4136353, -1260.6616211, 1811.9104004
2: -290.3746338, 1641.2213135, -158.4233246, 878.5940552, -1168.9687500, 1799.6445312
3: -711.9147949, 1393.7313232, -385.8987732, 744.7566528, -1456.6713867, 1779.6301270
4: -478.8499146, 1407.6657715, -260.9156494, 753.7153931, -1232.5653076, 1668.5814209

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805909, upper bound: 743.6790662
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801571, upper bound: 743.6801571
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -743.6815655, upper bound: 743.6815655
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -743.6805909, upper bound: 743.6790662
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 0, lower bound: -743.6801571, upper bound: 743.6801571

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -128.5790863, 670.3025513, -798.8816528, 798.8816528
1: -210.4309387, 796.4016113, -210.4309387, 796.4016113, -1006.8324585, 1006.8324585
2: -148.6245575, 824.4069214, -148.6245575, 824.4069214, -973.0314331, 973.0314331
3: -362.4870300, 699.0101929, -362.4870300, 699.0101929, -1061.4970703, 1061.4970703
4: -244.7495728, 707.2666626, -244.7495728, 707.2666626, -952.0162354, 952.0162354

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813688, upper bound: 743.6866848
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813432, upper bound: 743.6830471
time: 0.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -250.1162415, 1337.2148438, -1465.7939453, 920.4188232
1: -210.4309387, 796.4016113, -412.2479858, 1588.0107422, -1798.4415283, 1208.6496582
2: -148.6245575, 824.4069214, -290.3746338, 1641.2213135, -1789.8458252, 1114.7813721
3: -362.4870300, 699.0101929, -711.9147949, 1393.7313232, -1756.2182617, 1410.9250488
4: -244.7495728, 707.2666626, -478.8499146, 1407.6657715, -1652.4152832, 1186.1165771

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813688, upper bound: 743.6866848
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813432, upper bound: 743.6830471
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -250.1162415, 1337.2148438, -120.0922165, 627.7528687, -877.8690796, 1457.3070068
1: -412.2479858, 1588.0107422, -196.3129883, 745.1196289, -1157.3676758, 1784.3237305
2: -290.3746338, 1641.2213135, -138.9442291, 772.6103516, -1062.9848633, 1780.1655273
3: -711.9147949, 1393.7313232, -338.6782532, 653.0143433, -1364.9291992, 1732.4093018
4: -478.8499146, 1407.6657715, -228.7958984, 661.7055664, -1140.5554199, 1636.4616699

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6666895, upper bound: 743.6633821
time: 0.67 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -247.4304504, 1323.2124023, -142.7722015, 747.6100464, -995.0405273, 1465.9846191
1: -407.7603149, 1571.3062744, -232.2279968, 887.3795166, -1295.1397705, 1803.5343018
2: -287.2580872, 1624.0274658, -165.1784821, 919.9380493, -1207.1961670, 1789.2058105
3: -704.2532959, 1378.9536133, -401.7619019, 777.4907837, -1481.7441406, 1780.7155762
4: -473.7201538, 1392.8149414, -272.7248840, 787.1499023, -1260.8698730, 1665.5397949

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6813688, upper bound: 743.6866848
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6813432, upper bound: 743.6830471
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6813688, upper bound: 743.6866848
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6813432, upper bound: 743.6830471
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6666895, upper bound: 743.6633821
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -128.5790863, 670.3025513, -783.2761841, 719.3760986
1: -184.9509888, 701.4193115, -210.4309387, 796.4016113, -981.3526001, 911.8501587
2: -130.6758118, 727.0247803, -148.6245575, 824.4069214, -955.0827026, 875.6492920
3: -318.8685303, 614.4975586, -362.4870300, 699.0101929, -1017.8787231, 976.9845581
4: -215.1691284, 622.4899902, -244.7495728, 707.2666626, -922.4357910, 867.2395020

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842199, upper bound: 743.6842199
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842199, upper bound: 743.6842332
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -125.8274994, 656.2141113, -790.6890259, 830.4873657
1: -218.6857605, 836.5494995, -205.8549194, 779.5963135, -998.2821045, 1042.4044189
2: -155.4785309, 866.9288940, -145.4328918, 807.0996704, -962.5781250, 1012.3618164
3: -378.4638367, 732.4861450, -354.6449890, 684.0853271, -1062.5489502, 1087.1311035
4: -256.7464294, 741.4154663, -239.4897461, 692.2363281, -948.9826660, 980.9050903

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842199
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842332
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -250.1162415, 1337.2148438, -1450.1884766, 840.9132080
1: -184.9509888, 701.4193115, -412.2479858, 1588.0107422, -1772.9616699, 1113.6672363
2: -130.6758118, 727.0247803, -290.3746338, 1641.2213135, -1771.8970947, 1017.3994141
3: -318.8685303, 614.4975586, -711.9147949, 1393.7313232, -1712.5998535, 1326.4123535
4: -215.1691284, 622.4899902, -478.8499146, 1407.6657715, -1622.8348389, 1101.3397217

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -247.4304504, 1323.2124023, -1457.6872559, 952.0904541
1: -218.6857605, 836.5494995, -407.7603149, 1571.3062744, -1789.9920654, 1244.3096924
2: -155.4785309, 866.9288940, -287.2580872, 1624.0274658, -1779.5059814, 1154.1870117
3: -378.4638367, 732.4861450, -704.2532959, 1378.9536133, -1757.4174805, 1436.7395020
4: -256.7464294, 741.4154663, -473.7201538, 1392.8149414, -1649.4254150, 1215.1356201

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -247.2919464, 1322.5637207, -120.0922165, 627.7528687, -875.0447388, 1442.6557617
1: -407.6062622, 1570.5925293, -196.3129883, 745.1196289, -1152.7258301, 1766.9055176
2: -287.0827942, 1623.1872559, -138.9442291, 772.6103516, -1059.6931152, 1762.1314697
3: -703.9343872, 1378.3023682, -338.6782532, 653.0143433, -1356.9487305, 1716.9803467
4: -473.4362183, 1392.1020508, -228.7958984, 661.7055664, -1135.1417236, 1620.8979492

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -239.8428802, 1282.4177246, -142.7722015, 747.6100464, -987.4528198, 1425.1899414
1: -395.2958069, 1522.6656494, -232.2279968, 887.3795166, -1282.6752930, 1754.8936768
2: -278.3708496, 1574.2863770, -165.1784821, 919.9380493, -1198.3088379, 1739.4647217
3: -682.6391602, 1335.6937256, -401.7619019, 777.4907837, -1460.1297607, 1737.4555664
4: -458.9256897, 1349.7456055, -272.7248840, 787.1499023, -1246.0754395, 1622.4704590

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -243.7597351, 1305.2003174, -142.7578430, 747.5330811, -991.2928467, 1447.9580078
1: -401.8029785, 1549.6038818, -232.2046509, 887.2879639, -1289.0908203, 1781.8084717
2: -283.0040894, 1601.8430176, -165.1617279, 919.8441772, -1202.8482666, 1767.0047607
3: -693.8829956, 1359.1678467, -401.7213135, 777.4093628, -1471.2921143, 1760.8891602
4: -466.5953064, 1373.1367188, -272.6969910, 787.0690308, -1253.6640625, 1645.8336182

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.29 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6842199, upper bound: 743.6842199
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6842199, upper bound: 743.6842332
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842199
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842332
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6798833, upper bound: 743.6737212
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.29
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -112.9736404, 590.7969971, -703.7706299, 703.7706299
1: -184.9509888, 701.4193115, -184.9509888, 701.4193115, -886.3702393, 886.3703003
2: -130.6758118, 727.0247803, -130.6758118, 727.0247803, -857.7005615, 857.7005615
3: -318.8685303, 614.4975586, -318.8685303, 614.4975586, -933.3660278, 933.3660278
4: -215.1691284, 622.4899902, -215.1691284, 622.4899902, -837.6589966, 837.6589966

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6839743, upper bound: 743.6877314
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842589, upper bound: 743.6878936
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -134.4748993, 704.6600342, -817.6336670, 725.2719116
1: -184.9509888, 701.4193115, -218.6857605, 836.5494995, -1021.5004272, 920.1050415
2: -130.6758118, 727.0247803, -155.4785309, 866.9288940, -997.6045532, 882.5032349
3: -318.8685303, 614.4975586, -378.4638367, 732.4861450, -1051.3546143, 992.9613037
4: -215.1691284, 622.4899902, -256.7464294, 741.4154663, -956.5845947, 879.2362671

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6839743, upper bound: 743.6877314
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842589, upper bound: 743.6878936
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -112.9736404, 590.7969971, -725.2719116, 817.6336670
1: -218.6857605, 836.5494995, -184.9509888, 701.4193115, -920.1051025, 1021.5004883
2: -155.4785309, 866.9288940, -130.6758118, 727.0247803, -882.5032349, 997.6045532
3: -378.4638367, 732.4861450, -318.8685303, 614.4975586, -992.9613037, 1051.3546143
4: -256.7464294, 741.4154663, -215.1691284, 622.4899902, -879.2362671, 956.5845337

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6820706
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842199
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -134.4748993, 704.6600342, -839.1348877, 839.1348877
1: -218.6857605, 836.5494995, -218.6857605, 836.5494995, -1055.2352295, 1055.2352295
2: -155.4785309, 866.9288940, -155.4785309, 866.9288940, -1022.4072266, 1022.4073486
3: -378.4638367, 732.4861450, -378.4638367, 732.4861450, -1110.9495850, 1110.9497070
4: -256.7464294, 741.4154663, -256.7464294, 741.4154663, -998.1618042, 998.1618042

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6820706
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6842199
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -242.5670013, 1296.6496582, -1409.6232910, 833.3640137
1: -184.9509888, 701.4193115, -399.8506775, 1539.6394043, -1724.5903320, 1101.2698975
2: -130.6758118, 727.0247803, -281.5342102, 1591.7542725, -1722.4300537, 1008.5589600
3: -318.8685303, 614.4975586, -690.4160156, 1350.7003174, -1669.5688477, 1304.9134521
4: -215.1691284, 622.4899902, -464.1332703, 1364.8315430, -1580.0007324, 1086.6230469

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6781712, upper bound: 743.6769091
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -112.9629974, 590.7405396, -246.7570343, 1320.7597656, -1433.7227783, 837.4974976
1: -184.9334412, 701.3522949, -406.8105774, 1568.1750488, -1753.1085205, 1108.1628418
2: -130.6633606, 726.9550171, -286.4840698, 1620.9423828, -1751.6055908, 1013.4390869
3: -318.8380127, 614.4383545, -702.4439087, 1375.6123047, -1694.4503174, 1316.8823242
4: -215.1484680, 622.4302979, -472.3248596, 1389.6550293, -1604.8034668, 1094.7551270

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6778725, upper bound: 743.6768206
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -239.8428802, 1282.4177246, -1416.8925781, 944.5027466
1: -218.6857605, 836.5494995, -395.2958069, 1522.6656494, -1741.3514404, 1231.8453369
2: -155.4785309, 866.9288940, -278.3708496, 1574.2863770, -1729.7648926, 1145.2996826
3: -378.4638367, 732.4861450, -682.6391602, 1335.6937256, -1714.1574707, 1415.1250000
4: -256.7464294, 741.4154663, -458.9256897, 1349.7456055, -1606.3453369, 1200.3411865

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787728, upper bound: 743.6778605
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -134.4611969, 704.5866699, -243.7597351, 1305.2003174, -1439.6614990, 948.3464355
1: -218.6634216, 836.4621582, -401.8029785, 1549.6038818, -1768.2673340, 1238.2650146
2: -155.4625854, 866.8391724, -283.0040894, 1601.8430176, -1757.3056641, 1149.8432617
3: -378.4249573, 732.4086914, -693.8829956, 1359.1678467, -1737.5927734, 1426.2915039
4: -256.7197571, 741.3384399, -466.5953064, 1373.1367188, -1629.8494873, 1207.9334717

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784839, upper bound: 743.6778326
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -247.2919464, 1322.5637207, -112.9736404, 590.7969971, -838.0889282, 1435.5371094
1: -407.6062622, 1570.5925293, -184.9509888, 701.4193115, -1109.0255127, 1755.5434570
2: -287.0827942, 1623.1872559, -130.6758118, 727.0247803, -1014.1075439, 1753.8630371
3: -703.9343872, 1378.3023682, -318.8685303, 614.4975586, -1318.4318848, 1697.1708984
4: -473.4362183, 1392.1020508, -215.1691284, 622.4899902, -1095.9261475, 1607.2709961

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763800, upper bound: 743.6698956
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720721, upper bound: 743.6693660
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6735899, upper bound: 743.6699034
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -247.2919464, 1322.5637207, -233.4036713, 1251.2222900, -1498.5142822, 1555.9671631
1: -407.6062622, 1570.5925293, -384.9095154, 1485.3535156, -1892.9595947, 1955.5017090
2: -287.0827942, 1623.1872559, -271.0440063, 1535.6855469, -1822.7683105, 1894.2312012
3: -703.9343872, 1378.3023682, -665.0811768, 1302.4888916, -2006.4233398, 2043.3834229
4: -473.4362183, 1392.1020508, -447.0042725, 1316.2604980, -1789.6967773, 1839.1059570

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763800, upper bound: 743.6698956
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720721, upper bound: 743.6693660
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6735899, upper bound: 743.6699034
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -239.8428802, 1282.4177246, -136.1856384, 713.1320801, -952.9748535, 1418.6033936
1: -395.2958069, 1522.6656494, -221.5577393, 846.6779785, -1241.9737549, 1744.2233887
2: -278.3708496, 1574.2863770, -157.4887085, 877.3947754, -1155.7656250, 1731.7751465
3: -682.6391602, 1335.6937256, -383.4264526, 741.7033691, -1424.3424072, 1719.1201172
4: -458.9256897, 1349.7456055, -260.0836182, 750.6716309, -1209.5972900, 1609.6905518

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -239.8428802, 1282.4177246, -248.7903900, 1326.3448486, -1566.1877441, 1531.2081299
1: -395.2958069, 1522.6656494, -408.3983154, 1574.8801270, -1970.1759033, 1931.0638428
2: -278.3708496, 1574.2863770, -288.6805115, 1628.2045898, -1906.5754395, 1862.9667969
3: -682.6391602, 1335.6937256, -706.7491455, 1382.5224609, -2065.1616211, 2042.4428711
4: -458.9256897, 1349.7456055, -476.3330994, 1396.8769531, -1855.8023682, 1826.0787354

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -243.7597351, 1305.2003174, -136.1719208, 713.0587158, -956.8184814, 1441.3719482
1: -401.8029785, 1549.6038818, -221.5353394, 846.5904541, -1248.3931885, 1771.1391602
2: -283.0040894, 1601.8430176, -157.4727478, 877.3049316, -1160.3089600, 1759.3157959
3: -693.8829956, 1359.1678467, -383.3876038, 741.6259766, -1435.5087891, 1742.5552979
4: -466.5953064, 1373.1367188, -260.0570374, 750.5943604, -1217.1894531, 1633.1937256

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -243.7597351, 1305.2003174, -248.7789764, 1326.2866211, -1570.0462646, 1553.9791260
1: -401.8029785, 1549.6038818, -408.3800049, 1574.8106689, -1976.6136475, 1957.9835205
2: -283.0040894, 1601.8430176, -288.6672668, 1628.1330566, -1911.1372070, 1890.5101318
3: -693.8829956, 1359.1678467, -706.7171021, 1382.4606934, -2076.3437500, 2065.8850098
4: -466.5953064, 1373.1367188, -476.3110962, 1396.8149414, -1863.4100342, 1849.4477539

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.69 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6839743, upper bound: 743.6877314
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6842589, upper bound: 743.6878936
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6839743, upper bound: 743.6877314
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6842589, upper bound: 743.6878936
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6820706
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6842332, upper bound: 743.6842199
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6820706
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6829823, upper bound: 743.6842199
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6781712, upper bound: 743.6769091
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6778725, upper bound: 743.6768206
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6787728, upper bound: 743.6778605
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6784839, upper bound: 743.6778326
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6720721, upper bound: 743.6693660
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6735899, upper bound: 743.6699034
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6720721, upper bound: 743.6693660
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6735899, upper bound: 743.6699034
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -112.3619843, 587.6158447, -689.4331055, 644.1060791
1: -166.8022614, 631.1797485, -183.9725494, 697.6421509, -864.4443970, 815.1522827
2: -117.8099060, 654.6407471, -129.9730530, 723.1051636, -840.9150391, 784.6137695
3: -287.4974365, 552.7945557, -317.1716309, 611.1724243, -898.6698608, 869.9661255
4: -193.9462280, 560.3420410, -214.0091095, 619.1318359, -813.0780640, 774.3511353

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6864317, upper bound: 743.6828601
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826666, upper bound: 743.6824669
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -112.9736404, 590.7969971, -700.7548828, 687.8096313
1: -180.0325623, 682.4268188, -184.9509888, 701.4193115, -881.4517822, 867.3778076
2: -127.1877441, 707.4198608, -130.6758118, 727.0247803, -854.2125244, 838.0957031
3: -310.3284607, 597.7552490, -318.8685303, 614.4975586, -924.8258667, 916.6237793
4: -209.4055481, 605.6980591, -215.1691284, 622.4899902, -831.8955078, 820.8670654

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6881740, upper bound: 743.6879819
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6881740, upper bound: 743.6881741
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -133.7364807, 700.7603149, -802.5775757, 665.4805908
1: -166.8022614, 631.1797485, -217.4989166, 831.9152222, -998.7174683, 848.6786499
2: -117.8099060, 654.6407471, -154.6292267, 862.1469727, -979.9568481, 809.2698975
3: -287.4974365, 552.7945557, -376.4168396, 728.4112549, -1015.9086914, 929.2113037
4: -193.9462280, 560.3420410, -255.3415527, 737.3234253, -931.2696533, 815.6835938

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818251, upper bound: 743.6864300
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818251, upper bound: 743.6877314
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -134.4748993, 704.6600342, -814.6177979, 709.3108521
1: -180.0325623, 682.4268188, -218.6857605, 836.5494995, -1016.5820312, 901.1125488
2: -127.1877441, 707.4198608, -155.4785309, 866.9288940, -994.1166382, 862.8983154
3: -310.3284607, 597.7552490, -378.4638367, 732.4861450, -1042.8145752, 976.2189941
4: -209.4055481, 605.6980591, -256.7464294, 741.4154663, -950.8210449, 862.4443359

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6866200
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6878936
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -112.3619843, 587.6158447, -714.2901611, 775.9389648
1: -205.9380188, 787.8771362, -183.9725494, 697.6421509, -903.5801392, 971.8496094
2: -146.4624939, 816.3840942, -129.9730530, 723.1051636, -869.5676270, 946.3571777
3: -356.4977722, 689.8046875, -317.1716309, 611.1724243, -967.6701660, 1006.9762573
4: -241.9350891, 698.1940918, -214.0091095, 619.1318359, -861.0668945, 912.2031860

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6857037, upper bound: 743.6803648
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820073, upper bound: 743.6799799
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -112.9736404, 590.7969971, -722.8632202, 804.6921387
1: -214.7357178, 821.2271729, -184.9509888, 701.4193115, -916.1550293, 1006.1781616
2: -152.6842041, 851.0744629, -130.6758118, 727.0247803, -879.7089844, 981.7502441
3: -371.6125488, 719.0783691, -318.8685303, 614.4975586, -986.1101074, 1037.9468994
4: -252.1509399, 727.8773804, -215.1691284, 622.4899902, -874.6407471, 943.0465088

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6877314, upper bound: 743.6839743
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6877314, upper bound: 743.6842589
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -133.7364807, 700.7603149, -827.4346313, 797.3134155
1: -205.9380188, 787.8771362, -217.4989166, 831.9152222, -1037.8532715, 1005.3760376
2: -146.4624939, 816.3840942, -154.6292267, 862.1469727, -1008.6094971, 971.0132446
3: -356.4977722, 689.8046875, -376.4168396, 728.4112549, -1084.9089355, 1066.2215576
4: -241.9350891, 698.1940918, -255.3415527, 737.3234253, -979.2585449, 953.5356445

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808331, upper bound: 743.6808331
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808331, upper bound: 743.6820706
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -134.4748993, 704.6600342, -836.7262573, 826.1933594
1: -214.7357178, 821.2271729, -218.6857605, 836.5494995, -1051.2850342, 1039.9128418
2: -152.6842041, 851.0744629, -155.4785309, 866.9288940, -1019.6129761, 1006.5529175
3: -371.6125488, 719.0783691, -378.4638367, 732.4861450, -1104.0983887, 1097.5421143
4: -252.1509399, 727.8773804, -256.7464294, 741.4154663, -993.5663452, 984.6237183

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820840, upper bound: 743.6829823
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820840, upper bound: 743.6842199
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -241.6424255, 1291.9398193, -1393.7569580, 773.3864746
1: -166.8022614, 631.1797485, -398.3587646, 1534.0263672, -1700.8286133, 1029.5384521
2: -117.8099060, 654.6407471, -280.4677124, 1585.9602051, -1703.7700195, 935.1084595
3: -287.4974365, 552.7945557, -687.8258057, 1345.7094727, -1633.2066650, 1240.6203613
4: -193.9462280, 560.3420410, -462.3746033, 1359.8139648, -1553.7600098, 1022.7166748

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6727926, upper bound: 743.6715464
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6769919, upper bound: 743.6765238
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6360544, upper bound: 743.6410043
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -242.5670013, 1296.6496582, -1406.6072998, 817.4028931
1: -180.0325623, 682.4268188, -399.8506775, 1539.6394043, -1719.6719971, 1082.2773438
2: -127.1877441, 707.4198608, -281.5342102, 1591.7542725, -1718.9420166, 988.9540405
3: -310.3284607, 597.7552490, -690.4160156, 1350.7003174, -1661.0288086, 1288.1712646
4: -209.4055481, 605.6980591, -464.1332703, 1364.8315430, -1574.2370605, 1069.8312988

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6769975, upper bound: 743.6821124
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798096, upper bound: 743.6833353
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -101.8052139, 531.6807251, -245.8241119, 1316.0106201, -1417.8157959, 777.5048218
1: -166.7823639, 631.1045532, -405.3073120, 1562.5107422, -1729.2930908, 1036.4117432
2: -117.7958450, 654.5631714, -285.4082642, 1615.1057129, -1732.9011230, 939.9713745
3: -287.4628296, 552.7276611, -699.8385620, 1370.5721436, -1658.0349121, 1252.5660400
4: -193.9229126, 560.2749634, -470.5489197, 1384.5959473, -1578.5187988, 1030.8237305

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6726950, upper bound: 743.6714051
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6765903, upper bound: 743.6763484
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6355355, upper bound: 743.6409918
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -109.9467926, 574.7775269, -246.7570343, 1320.7597656, -1430.7062988, 821.5345459
1: -180.0142975, 682.3574829, -406.8105774, 1568.1750488, -1748.1893311, 1089.1680908
2: -127.1748505, 707.3479614, -286.4840698, 1620.9423828, -1748.1171875, 993.8320312
3: -310.2967224, 597.6939697, -702.4439087, 1375.6123047, -1685.9090576, 1300.1379395
4: -209.3841095, 605.6361694, -472.3248596, 1389.6550293, -1599.0388184, 1077.9610596

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768838, upper bound: 743.6818411
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788908, upper bound: 743.6830849
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -238.9404907, 1277.8133545, -1404.4876709, 902.5173340
1: -205.9380188, 787.8771362, -393.8414307, 1517.1793213, -1723.1173096, 1181.7185059
2: -146.4624939, 816.3840942, -277.3300476, 1568.6225586, -1715.0850830, 1093.7138672
3: -356.4977722, 689.8046875, -680.1136475, 1330.8182373, -1687.3160400, 1369.9183350
4: -241.9350891, 698.1940918, -457.2096558, 1344.8417969, -1586.5640869, 1155.4033203

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785069, upper bound: 743.6775260
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784121, upper bound: 743.6776251
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -239.8428802, 1282.4177246, -1414.4840088, 931.5612793
1: -214.7357178, 821.2271729, -395.2958069, 1522.6656494, -1737.4011230, 1216.5229492
2: -152.6842041, 851.0744629, -278.3708496, 1574.2863770, -1726.9703369, 1129.4453125
3: -371.6125488, 719.0783691, -682.6391602, 1335.6937256, -1707.3061523, 1401.7175293
4: -252.1509399, 727.8773804, -458.9256897, 1349.7456055, -1601.7054443, 1186.8031006

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797513, upper bound: 743.6805455
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -126.6598892, 663.4994507, -242.8523254, 1300.5788574, -1427.2387695, 906.3518066
1: -205.9143219, 787.7847900, -400.3416748, 1544.0921631, -1750.0064697, 1188.1263428
2: -146.4456940, 816.2894287, -281.9579773, 1596.1600342, -1742.6057129, 1098.2473145
3: -356.4568481, 689.7225952, -691.3504028, 1354.2634277, -1710.7202148, 1381.0729980
4: -241.9069214, 698.1126099, -464.8681641, 1368.2100830, -1610.0455322, 1162.9805908

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6778032, upper bound: 743.6773254
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780556, upper bound: 743.6775607
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -132.0523224, 691.6439209, -243.7597351, 1305.2003174, -1437.2525635, 935.4035645
1: -214.7131042, 821.1384277, -401.8029785, 1549.6038818, -1764.3167725, 1222.9414062
2: -152.6679993, 850.9833984, -283.0040894, 1601.8430176, -1754.5107422, 1133.9875488
3: -371.5732117, 718.9995728, -693.8829956, 1359.1678467, -1730.7410889, 1412.8824463
4: -252.1238403, 727.7990112, -466.5953064, 1373.1367188, -1625.2093506, 1194.3939209

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788325, upper bound: 743.6802932
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -234.0531921, 1251.5848389, -136.1856384, 713.1320801, -947.1851196, 1387.7705078
1: -385.9094543, 1485.8415527, -221.5577393, 846.6779785, -1232.5872803, 1707.3992920
2: -271.6626587, 1536.5400391, -157.4887085, 877.3947754, -1149.0573730, 1694.0288086
3: -666.4055786, 1303.0963135, -383.4264526, 741.7033691, -1408.1088867, 1686.5227051
4: -447.7946777, 1317.2755127, -260.0836182, 750.6716309, -1198.4663086, 1577.2559814

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775260, upper bound: 743.6785069
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805455, upper bound: 743.6797513
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -237.7599335, 1271.9384766, -135.9785767, 712.0637207, -949.8236694, 1407.9168701
1: -391.7726440, 1510.1634521, -221.2243958, 845.4016724, -1237.1741943, 1731.3876953
2: -275.9483032, 1561.4024658, -157.2509155, 876.0835571, -1152.0318604, 1718.6533203
3: -676.6433105, 1324.4810791, -382.8513489, 740.5696411, -1417.2127686, 1707.3322754
4: -454.8610840, 1338.3919678, -259.6904602, 749.5358887, -1204.3967285, 1598.0821533

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776251, upper bound: 743.6784121
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818684, upper bound: 743.6802873
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -234.0531921, 1251.5848389, -248.7903900, 1326.3448486, -1560.3979492, 1500.3752441
1: -385.9094543, 1485.8415527, -408.3983154, 1574.8801270, -1960.7894287, 1894.2398682
2: -271.6626587, 1536.5400391, -288.6805115, 1628.2045898, -1899.8671875, 1825.2203369
3: -666.4055786, 1303.0963135, -706.7491455, 1382.5224609, -2048.9279785, 2009.8454590
4: -447.7946777, 1317.2755127, -476.3330994, 1396.8769531, -1844.6716309, 1793.6086426

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -237.7599335, 1271.9384766, -248.6099701, 1325.4173584, -1563.1772461, 1520.5484619
1: -391.7726440, 1510.1634521, -408.1109009, 1573.7746582, -1965.5473633, 1918.2742920
2: -275.9483032, 1561.4024658, -288.4735413, 1627.0687256, -1903.0169678, 1849.8759766
3: -676.6433105, 1324.4810791, -706.2512817, 1381.5433350, -2058.1865234, 2030.7321777
4: -454.8610840, 1338.3919678, -475.9911499, 1395.8955078, -1850.7563477, 1814.3830566

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -238.2329407, 1275.9495850, -136.1719208, 713.0587158, -951.2916260, 1412.1213379
1: -392.8231812, 1514.6723633, -221.5353394, 846.5904541, -1239.4129639, 1736.2077637
2: -276.6023865, 1566.0113525, -157.4727478, 877.3049316, -1153.9073486, 1723.4841309
3: -678.3593750, 1328.2314453, -383.3876038, 741.6259766, -1419.9853516, 1711.6188965
4: -455.9827576, 1342.2722168, -260.0570374, 750.5943604, -1206.5769043, 1602.3292236

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773254, upper bound: 743.6778032
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802932, upper bound: 743.6788325
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -241.1236572, 1291.6192627, -135.9648743, 711.9904175, -953.1140747, 1427.5841064
1: -397.3216553, 1533.5045166, -221.2020264, 845.3142090, -1242.6358643, 1754.7062988
2: -279.9212036, 1585.0710449, -157.2349854, 875.9938354, -1155.9150391, 1742.3060303
3: -686.0750732, 1344.8690186, -382.8125610, 740.4920654, -1426.5671387, 1727.6812744
4: -461.4537659, 1358.4593506, -259.6637878, 749.4586792, -1210.9124756, 1618.1229248

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775607, upper bound: 743.6780556
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818597, upper bound: 743.6799724
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -238.2329407, 1275.9495850, -248.7789764, 1326.2866211, -1564.5194092, 1524.7285156
1: -392.8231812, 1514.6723633, -408.3800049, 1574.8106689, -1967.6334229, 1923.0521240
2: -276.6023865, 1566.0113525, -288.6672668, 1628.1330566, -1904.7354736, 1854.6784668
3: -678.3593750, 1328.2314453, -706.7171021, 1382.4606934, -2060.8200684, 2034.9484863
4: -455.9827576, 1342.2722168, -476.3110962, 1396.8149414, -1852.7974854, 1818.5832520

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -241.1236572, 1291.6192627, -248.5985870, 1325.3591309, -1566.4826660, 1540.2177734
1: -397.3216553, 1533.5045166, -408.0926208, 1573.7056885, -1971.0273438, 1941.5970459
2: -279.9212036, 1585.0710449, -288.4602966, 1626.9970703, -1906.9182129, 1873.5313721
3: -686.0750732, 1344.8690186, -706.2192383, 1381.4818115, -2067.5568848, 2051.0878906
4: -461.4537659, 1358.4593506, -475.9691467, 1395.8334961, -1857.2871094, 1834.4283447

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.45 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6864317, upper bound: 743.6828601
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6826666, upper bound: 743.6824669
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6881740, upper bound: 743.6879819
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6881740, upper bound: 743.6881741
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6818251, upper bound: 743.6864300
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6818251, upper bound: 743.6877314
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6866200
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6878936
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6857037, upper bound: 743.6803648
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6820073, upper bound: 743.6799799
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6877314, upper bound: 743.6839743
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6877314, upper bound: 743.6842589
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6808331, upper bound: 743.6808331
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6808331, upper bound: 743.6820706
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6820840, upper bound: 743.6829823
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6820840, upper bound: 743.6842199
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6769919, upper bound: 743.6765238
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6360544, upper bound: 743.6410043
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6798096, upper bound: 743.6833353
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6803456, upper bound: 743.6846403
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6765903, upper bound: 743.6763484
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6355355, upper bound: 743.6409918
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6788908, upper bound: 743.6830849
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6800307, upper bound: 743.6846308
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6785069, upper bound: 743.6775260
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6784121, upper bound: 743.6776251
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6797513, upper bound: 743.6805455
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6802873, upper bound: 743.6818684
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6778032, upper bound: 743.6773254
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6780556, upper bound: 743.6775607
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6788325, upper bound: 743.6802932
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6799724, upper bound: 743.6818597
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6775260, upper bound: 743.6785069
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6805455, upper bound: 743.6797513
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6776251, upper bound: 743.6784121
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6818684, upper bound: 743.6802873
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6774072, upper bound: 743.6785003
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6787301, upper bound: 743.6790363
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6773254, upper bound: 743.6778032
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6802932, upper bound: 743.6788325
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6775607, upper bound: 743.6780556
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6818597, upper bound: 743.6799724
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6771549, upper bound: 743.6775815
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 0, lower bound: -743.6787214, upper bound: 743.6787214

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -107.1509323, 559.7775879, -661.5948486, 638.8950195
1: -166.8022614, 631.1797485, -175.4253082, 664.3189697, -831.1212158, 806.6050415
2: -117.8099060, 654.6407471, -123.9257889, 689.1486206, -806.9584961, 778.5665283
3: -287.4974365, 552.7945557, -302.4336243, 581.6336670, -869.1311035, 855.2281494
4: -193.9462280, 560.3420410, -203.9763184, 589.5694580, -783.5156860, 764.3183594

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824569
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824569
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -101.5842285, 530.4865723, -112.6894531, 589.4049072, -690.9891357, 643.1759644
1: -166.4159393, 629.6966553, -184.3572845, 699.5338745, -865.9497070, 814.0538940
2: -117.5372925, 653.0960693, -130.3259277, 725.4063721, -842.9436035, 783.4219360
3: -286.8287659, 551.4840088, -318.0965271, 612.4905396, -899.3190918, 869.5805664
4: -193.4944153, 559.0191040, -214.5905151, 620.7008057, -814.1951904, 773.6096191

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824669
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824669
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -101.8172531, 531.7440796, -641.7019653, 676.6532593
1: -180.0325623, 682.4268188, -166.8022614, 631.1797485, -811.2122803, 849.2290649
2: -127.1877441, 707.4198608, -117.8099060, 654.6407471, -781.8284912, 825.2297363
3: -310.3284607, 597.7552490, -287.4974365, 552.7945557, -863.1228638, 885.2526855
4: -209.4055481, 605.6980591, -193.9462280, 560.3420410, -769.7475586, 799.6442871

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6878274, upper bound: 743.6879819
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6881568, upper bound: 743.6878969
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -109.9578705, 574.8359985, -684.7937622, 684.7938232
1: -180.0325623, 682.4268188, -180.0325623, 682.4268188, -862.4593506, 862.4593506
2: -127.1877441, 707.4198608, -127.1877441, 707.4198608, -834.6076050, 834.6076050
3: -310.3284607, 597.7552490, -310.3284607, 597.7552490, -908.0836182, 908.0836182
4: -209.4055481, 605.6980591, -209.4055481, 605.6980591, -815.1035767, 815.1035767

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6878274, upper bound: 743.6879833
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6881568, upper bound: 743.6879224
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -126.6743240, 663.5769653, -765.3942261, 658.4183960
1: -166.8022614, 631.1797485, -205.9380188, 787.8771362, -954.6793823, 837.1177368
2: -117.8099060, 654.6407471, -146.4624939, 816.3840942, -934.1939697, 801.1032715
3: -287.4974365, 552.7945557, -356.4977722, 689.8046875, -977.3021240, 909.2922363
4: -193.9462280, 560.3420410, -241.9350891, 698.1940918, -892.1403198, 802.2770996

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802518, upper bound: 743.6855479
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798607, upper bound: 743.6818058
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -132.0662384, 691.7185059, -793.5357666, 663.8103027
1: -166.8022614, 631.1797485, -214.7357178, 821.2271729, -988.0294189, 845.9154663
2: -117.8099060, 654.6407471, -152.6842041, 851.0744629, -968.8842773, 807.3249512
3: -287.4974365, 552.7945557, -371.6125488, 719.0783691, -1006.5758057, 924.4071045
4: -193.9462280, 560.3420410, -252.1509399, 727.8773804, -921.8236084, 812.4929199

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802518, upper bound: 743.6861357
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799781, upper bound: 743.6823936
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -126.6743240, 663.5769653, -773.5347900, 701.5103149
1: -180.0325623, 682.4268188, -205.9380188, 787.8771362, -967.9096680, 888.3648682
2: -127.1877441, 707.4198608, -146.4624939, 816.3840942, -943.5718384, 853.8823242
3: -310.3284607, 597.7552490, -356.4977722, 689.8046875, -1000.1330566, 954.2529907
4: -209.4055481, 605.6980591, -241.9350891, 698.1940918, -907.5996094, 847.6331787

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817556, upper bound: 743.6864796
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6866200
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -132.0662384, 691.7185059, -801.6763306, 706.9021606
1: -180.0325623, 682.4268188, -214.7357178, 821.2271729, -1001.2597046, 897.1625366
2: -127.1877441, 707.4198608, -152.6842041, 851.0744629, -978.2622070, 860.1040649
3: -310.3284607, 597.7552490, -371.6125488, 719.0783691, -1029.4067383, 969.3677979
4: -209.4055481, 605.6980591, -252.1509399, 727.8773804, -937.2829590, 857.8488159

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817556, upper bound: 743.6872602
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6872725
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -107.1509323, 559.7775879, -686.4519043, 770.7279053
1: -205.9380188, 787.8771362, -175.4253082, 664.3189697, -870.2569580, 963.3024292
2: -146.4624939, 816.3840942, -123.9257889, 689.1486206, -835.6110840, 940.3098755
3: -356.4977722, 689.8046875, -302.4336243, 581.6336670, -938.1313477, 992.2382812
4: -241.9350891, 698.1940918, -203.9763184, 589.5694580, -831.5045166, 902.1703491

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6270127, upper bound: 743.6295727
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6849190, upper bound: 743.6800165
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -126.4196625, 662.2023315, -112.6894531, 589.4049072, -715.8245239, 774.8917847
1: -205.5193329, 786.2418823, -184.3572845, 699.5338745, -905.0532227, 970.5990601
2: -146.1662140, 814.6990356, -130.3259277, 725.4063721, -871.5725098, 945.0249634
3: -355.7765503, 688.3573608, -318.0965271, 612.4905396, -968.2669678, 1006.4538574
4: -241.4436340, 696.7399292, -214.5905151, 620.7008057, -862.1444092, 911.3304443

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6236556, upper bound: 743.6292227
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815619, upper bound: 743.6796665
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -101.8172531, 531.7440796, -663.8103027, 793.5357666
1: -214.7357178, 821.2271729, -166.8022614, 631.1797485, -845.9154663, 988.0294189
2: -152.6842041, 851.0744629, -117.8099060, 654.6407471, -807.3249512, 968.8843384
3: -371.6125488, 719.0783691, -287.4974365, 552.7945557, -924.4071045, 1006.5758057
4: -252.1509399, 727.8773804, -193.9462280, 560.3420410, -812.4929199, 921.8236084

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6827554, upper bound: 743.6824066
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6823936, upper bound: 743.6823641
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -109.9578705, 574.8359985, -706.9022217, 801.6763306
1: -214.7357178, 821.2271729, -180.0325623, 682.4268188, -897.1625366, 1001.2597046
2: -152.6842041, 851.0744629, -127.1877441, 707.4198608, -860.1040649, 978.2622070
3: -371.6125488, 719.0783691, -310.3284607, 597.7552490, -969.3677979, 1029.4067383
4: -252.1509399, 727.8773804, -209.4055481, 605.6980591, -857.8488159, 937.2829590

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6827554, upper bound: 743.6824085
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6823936, upper bound: 743.6823659
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -126.6743240, 663.5769653, -790.2512817, 790.2512817
1: -205.9380188, 787.8771362, -205.9380188, 787.8771362, -993.8150635, 993.8150635
2: -146.4624939, 816.3840942, -146.4624939, 816.3840942, -962.8465576, 962.8465576
3: -356.4977722, 689.8046875, -356.4977722, 689.8046875, -1046.3022461, 1046.3023682
4: -241.9350891, 698.1940918, -241.9350891, 698.1940918, -940.1291504, 940.1291504

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6226942, upper bound: 743.6299399
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802906, upper bound: 743.6802906
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -132.0662384, 691.7185059, -818.3928223, 795.6431885
1: -205.9380188, 787.8771362, -214.7357178, 821.2271729, -1027.1651611, 1002.6128540
2: -146.4624939, 816.3840942, -152.6842041, 851.0744629, -997.5369263, 969.0682983
3: -356.4977722, 689.8046875, -371.6125488, 719.0783691, -1075.5759277, 1061.4171143
4: -241.9350891, 698.1940918, -252.1509399, 727.8773804, -969.8125000, 950.3448486

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6226942, upper bound: 743.6312672
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802906, upper bound: 743.6816178
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -126.6743240, 663.5769653, -795.6431885, 818.3928223
1: -214.7357178, 821.2271729, -205.9380188, 787.8771362, -1002.6128540, 1027.1651611
2: -152.6842041, 851.0744629, -146.4624939, 816.3840942, -969.0682983, 997.5369873
3: -371.6125488, 719.0783691, -356.4977722, 689.8046875, -1061.4171143, 1075.5760498
4: -252.1509399, 727.8773804, -241.9350891, 698.1940918, -950.3448486, 969.8125000

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802683, upper bound: 743.6817473
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799066, upper bound: 743.6817047
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -132.0662384, 691.7185059, -823.7847290, 823.7847290
1: -214.7357178, 821.2271729, -214.7357178, 821.2271729, -1035.9626465, 1035.9626465
2: -152.6842041, 851.0744629, -152.6842041, 851.0744629, -1003.7586060, 1003.7586670
3: -371.6125488, 719.0783691, -371.6125488, 719.0783691, -1090.6907959, 1090.6907959
4: -252.1509399, 727.8773804, -252.1509399, 727.8773804, -980.0282593, 980.0282593

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802683, upper bound: 743.6822461
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799066, upper bound: 743.6822223
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -235.9107513, 1261.3925781, -1363.2098389, 767.6548462
1: -166.8022614, 631.1797485, -389.0668335, 1497.5471191, -1664.3493652, 1020.2465210
2: -117.8099060, 654.6407471, -273.8268127, 1548.5727539, -1666.3825684, 928.4675293
3: -287.4974365, 552.7945557, -671.7572632, 1313.4250488, -1600.9222412, 1224.5517578
4: -193.9462280, 560.3420410, -451.3553162, 1327.6536865, -1521.5998535, 1011.6973267

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6308395, upper bound: 743.6227417
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687211, upper bound: 743.6687409
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -236.8246307, 1266.0694580, -1376.0272217, 811.6605225
1: -180.0325623, 682.4268188, -390.5416260, 1503.1191406, -1683.1517334, 1072.9682617
2: -127.1877441, 707.4198608, -274.8814697, 1554.3244629, -1681.5122070, 982.3013306
3: -310.3284607, 597.7552490, -674.3184814, 1318.3746338, -1628.7031250, 1272.0737305
4: -209.4055481, 605.6980591, -453.0951233, 1332.6312256, -1542.0367432, 1058.7932129

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6693150, upper bound: 743.6749183
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6694625, upper bound: 743.6715042
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -109.7855301, 573.9273682, -240.4738922, 1286.2316895, -1396.0170898, 814.4011841
1: -179.7562714, 681.3433838, -396.3256836, 1527.1926270, -1706.9488525, 1077.6690674
2: -126.9896317, 706.3111572, -279.1056824, 1578.9581299, -1705.9477539, 985.4168701
3: -309.8507996, 596.7990723, -684.4314575, 1339.5058594, -1649.3566895, 1281.2304688
4: -209.0772400, 604.7420654, -460.0658569, 1353.5141602, -1562.5914307, 1064.8077393

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671573, upper bound: 743.6726637
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6673025, upper bound: 743.6692544
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -101.8052139, 531.6807251, -240.4978027, 1287.8784180, -1389.6835938, 772.1785278
1: -166.7823639, 631.1045532, -396.6472473, 1528.9200439, -1695.7022705, 1027.7515869
2: -117.7958450, 654.5631714, -279.2397461, 1580.6298828, -1698.4254150, 933.8029175
3: -287.4628296, 552.7276611, -684.8715820, 1340.8245850, -1628.2873535, 1237.5992432
4: -193.9229126, 560.2749634, -460.3276978, 1354.8927002, -1548.8156738, 1020.6024170

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6576330, upper bound: 743.6559065
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761151, upper bound: 743.6761022
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.9467926, 574.7775269, -241.4227905, 1292.5960693, -1402.5426025, 816.2003174
1: -180.0142975, 682.3574829, -398.1369019, 1534.5454102, -1714.5595703, 1080.4943848
2: -127.1748505, 707.3479614, -280.3066406, 1586.4302979, -1713.6051025, 987.6546021
3: -310.2967224, 597.6939697, -687.4537964, 1345.8271484, -1656.1239014, 1285.1477051
4: -209.3841095, 605.6361694, -462.0883789, 1359.9160156, -1569.2998047, 1067.7246094

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6722829, upper bound: 743.6779561
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6724355, upper bound: 743.6745316
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.7744827, 573.8688965, -244.2327423, 1307.8359375, -1417.6103516, 818.1015625
1: -179.7380219, 681.2741699, -402.5149841, 1552.8361816, -1732.5740967, 1083.7891846
2: -126.9767380, 706.2391357, -283.5379944, 1604.9887695, -1731.9653320, 989.7770386
3: -309.8190613, 596.7377930, -694.9628296, 1361.9521484, -1671.7712402, 1291.7005615
4: -209.0558472, 604.6801147, -467.4091187, 1375.6409912, -1584.6967773, 1072.0892334

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794935, upper bound: 743.6846308
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794935, upper bound: 743.6846308
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -233.1640320, 1247.0372314, -1373.7115479, 896.7409668
1: -205.9380188, 787.8771362, -384.4758301, 1480.4250488, -1686.3630371, 1172.3529053
2: -146.4624939, 816.3840942, -270.6368103, 1530.9520264, -1677.4145508, 1087.0207520
3: -356.4977722, 689.8046875, -663.9157715, 1298.2878418, -1654.7855225, 1353.7204590
4: -241.9350891, 698.1940918, -446.1032104, 1312.4387207, -1554.1961670, 1144.2969971

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6145465, upper bound: 743.6222926
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780208, upper bound: 743.6771759
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -126.4276962, 662.2866211, -236.9096832, 1267.5852051, -1394.0128174, 899.1962891
1: -205.5394745, 786.3381348, -390.3914185, 1504.9875488, -1710.5267334, 1176.7293701
2: -146.1788940, 814.8035278, -274.9619141, 1556.0416260, -1702.2204590, 1089.7653809
3: -355.8104553, 688.4416504, -674.2388916, 1319.8978271, -1675.7081299, 1362.6804199
4: -241.4658966, 696.8294067, -453.2406921, 1333.7613525, -1575.2271729, 1150.0700684

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6057726, upper bound: 743.6144950
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779261, upper bound: 743.6773207
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -234.0531921, 1251.5848389, -1383.6511230, 925.7716064
1: -214.7357178, 821.2271729, -385.9094543, 1485.8415527, -1700.5771484, 1207.1365967
2: -152.6842041, 851.0744629, -271.6626587, 1536.5400391, -1689.2238770, 1122.7370605
3: -371.6125488, 719.0783691, -666.4055786, 1303.0963135, -1674.7088623, 1385.4838867
4: -252.1509399, 727.8773804, -447.7946777, 1317.2755127, -1569.2708740, 1175.6721191

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6693572, upper bound: 743.6715137
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6692206, upper bound: 743.6712352
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -131.8593750, 690.6527100, -237.7599335, 1271.9384766, -1403.7978516, 928.4125977
1: -214.4025421, 819.9540405, -391.7726440, 1510.1634521, -1724.5660400, 1211.7266846
2: -152.4466095, 849.7660522, -275.9483032, 1561.4024658, -1713.8487549, 1125.7143555
3: -371.0381775, 717.9470825, -676.6433105, 1324.4810791, -1695.5190430, 1394.5898438
4: -251.7581177, 726.7439575, -454.8610840, 1338.3919678, -1590.1500244, 1181.6049805

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6671996, upper bound: 743.6692639
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6670630, upper bound: 743.6689854
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -126.6598892, 663.4994507, -237.3340454, 1271.3621826, -1398.0220947, 900.8334961
1: -205.9143219, 787.7847900, -391.3764954, 1509.2041016, -1715.1184082, 1179.1611328
2: -146.4456940, 816.2894287, -275.5660095, 1560.3720703, -1706.8176270, 1091.8554688
3: -356.4568481, 689.7225952, -675.8513184, 1323.3682861, -1679.8251953, 1365.5739746
4: -241.9069214, 698.1126099, -454.2720337, 1337.3880615, -1579.2572021, 1152.3845215

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6141143, upper bound: 743.6222215
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773135, upper bound: 743.6770560
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -126.4132614, 662.2090454, -240.2502747, 1287.1490479, -1413.5622559, 902.4592896
1: -205.5158539, 786.2457275, -395.9108276, 1528.1837158, -1733.6994629, 1182.1564941
2: -146.1620483, 814.7088623, -278.9119873, 1579.5756836, -1725.7377930, 1093.6208496
3: -355.7695923, 688.3594971, -683.6232910, 1340.1411133, -1695.9106445, 1371.9827881
4: -241.4377747, 696.7479248, -459.7886353, 1353.7015381, -1595.1392822, 1156.5366211

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6081458, upper bound: 743.6170209
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775695, upper bound: 743.6772935
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -132.0523224, 691.6439209, -238.2329407, 1275.9495850, -1408.0019531, 929.8768311
1: -214.7131042, 821.1384277, -392.8231812, 1514.6723633, -1729.3853760, 1213.9615479
2: -152.6679993, 850.9833984, -276.6023865, 1566.0113525, -1718.6790771, 1127.5858154
3: -371.5732117, 718.9995728, -678.3593750, 1328.2314453, -1699.8046875, 1397.3588867
4: -252.1238403, 727.7990112, -455.9827576, 1342.2722168, -1594.3789062, 1183.7814941

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6723251, upper bound: 743.6745411
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6721885, upper bound: 743.6742626
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -131.8454132, 690.5778198, -241.1236572, 1291.6192627, -1423.4647217, 931.7014771
1: -214.3798828, 819.8651123, -397.3216553, 1533.5045166, -1747.8842773, 1217.1866455
2: -152.4303436, 849.6747437, -279.9212036, 1585.0710449, -1737.5013428, 1129.5959473
3: -370.9987488, 717.8682861, -686.0750732, 1344.8690186, -1715.8676758, 1403.9431152
4: -251.7309570, 726.6652832, -461.4537659, 1358.4593506, -1610.1901855, 1188.1190186

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791806, upper bound: 743.6803797
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791806, upper bound: 743.6818597
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -233.1640320, 1247.0372314, -128.1188507, 670.6936646, -903.8576050, 1375.1561279
1: -384.4758301, 1480.4250488, -208.3668671, 796.3804932, -1180.8562012, 1688.7918701
2: -270.6368103, 1530.9520264, -148.1596985, 825.1849976, -1095.8217773, 1679.1114502
3: -663.9157715, 1298.2878418, -360.6906738, 697.5521851, -1361.4680176, 1658.9785156
4: -446.1032104, 1312.4387207, -244.7476044, 705.9866333, -1152.0897217, 1557.0157471

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6766522, upper bound: 743.6780035
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771759, upper bound: 743.6780208
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -234.0531921, 1251.5848389, -133.4891968, 698.7611694, -932.8143311, 1385.0739746
1: -385.9094543, 1485.8415527, -217.1278076, 829.6489868, -1215.5584717, 1702.9692383
2: -271.6626587, 1536.5400391, -154.3556061, 859.7757568, -1131.4384766, 1690.8956299
3: -666.4055786, 1303.0963135, -375.7439880, 726.7465820, -1393.1519775, 1678.8403320
4: -447.7946777, 1317.2755127, -254.9263458, 735.5737915, -1183.3684082, 1572.0529785

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798421, upper bound: 743.6795350
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800652, upper bound: 743.6793765
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -236.9096832, 1267.5852051, -127.8715134, 669.4003296, -906.3099976, 1395.4566650
1: -390.3914185, 1504.9875488, -207.9667969, 794.8379517, -1185.2293701, 1712.9543457
2: -274.9619141, 1556.0416260, -147.8750763, 823.6004639, -1098.5621338, 1703.9166260
3: -674.2388916, 1319.8978271, -360.0009155, 696.1857910, -1370.4246826, 1679.8986816
4: -453.2406921, 1333.7613525, -244.2768555, 704.6179810, -1157.8586426, 1578.0380859

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767077, upper bound: 743.6777003
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773207, upper bound: 743.6779261
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -237.7599335, 1271.9384766, -133.2825623, 697.6956787, -935.4556274, 1405.2209473
1: -391.7726440, 1510.1634521, -216.7950287, 828.3763428, -1220.1488037, 1726.9583740
2: -275.9483032, 1561.4024658, -154.1183014, 858.4676514, -1134.4160156, 1715.5207520
3: -676.6433105, 1324.4810791, -375.1701355, 725.6158447, -1402.2589111, 1699.6510010
4: -454.8610840, 1338.3919678, -254.5339508, 734.4407959, -1189.3017578, 1592.9259033

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785019, upper bound: 743.6782874
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782405, upper bound: 743.6783345
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -234.0531921, 1251.5848389, -241.1274872, 1284.9580078, -1519.0111084, 1492.7121582
1: -385.9094543, 1485.8415527, -395.7741699, 1525.5197754, -1911.4291992, 1881.6156006
2: -271.6626587, 1536.5400391, -279.7143250, 1577.7105713, -1849.3732910, 1816.2542725
3: -666.4055786, 1303.0963135, -684.9904785, 1338.7708740, -2005.1765137, 1988.0867920
4: -447.7946777, 1317.2755127, -461.4421082, 1353.3442383, -1801.1389160, 1778.7176514

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771822
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770237
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -234.0531921, 1251.5848389, -245.6122437, 1310.8828125, -1544.9359131, 1497.1970215
1: -385.9094543, 1485.8415527, -403.3295288, 1556.3529053, -1942.2623291, 1889.1711426
2: -271.6626587, 1536.5400391, -284.9976196, 1609.1540527, -1880.8166504, 1821.5375977
3: -666.4055786, 1303.0963135, -697.8605347, 1365.7233887, -2032.1289062, 2000.9567871
4: -447.7946777, 1317.2755127, -470.1723633, 1380.0172119, -1827.8118896, 1787.4478760

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771954
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770370
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -237.7599335, 1271.9384766, -240.9488220, 1284.0413818, -1521.8012695, 1512.8872070
1: -391.7726440, 1510.1634521, -395.4900208, 1524.4262695, -1916.1988525, 1905.6534424
2: -275.9483032, 1561.4024658, -279.5095215, 1576.5880127, -1852.5362549, 1840.9118652
3: -676.6433105, 1324.4810791, -684.4973755, 1337.8015137, -2014.4445801, 2008.9781494
4: -454.8610840, 1338.3919678, -461.1033630, 1352.3737793, -1807.2346191, 1799.4952393

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782940, upper bound: 743.6782697
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780326, upper bound: 743.6783168
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -237.7599335, 1271.9384766, -245.4285889, 1309.9379883, -1547.6978760, 1517.3670654
1: -391.7726440, 1510.1634521, -403.0364075, 1555.2268066, -1946.9993896, 1913.1998291
2: -275.9483032, 1561.4024658, -284.7868347, 1607.9970703, -1883.9453125, 1846.1893311
3: -676.6433105, 1324.4810791, -697.3526611, 1364.7255859, -2041.3686523, 2021.8337402
4: -454.8610840, 1338.3919678, -469.8246155, 1379.0159912, -1833.8768311, 1808.2165527

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782940, upper bound: 743.6782697
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780326, upper bound: 743.6783168
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -237.3340454, 1271.3621826, -128.1044159, 670.6162109, -907.9502563, 1399.4665527
1: -391.3764954, 1509.2041016, -208.3432159, 796.2881470, -1187.6646729, 1717.5473633
2: -275.5660095, 1560.3720703, -148.1428528, 825.0903320, -1100.6563721, 1708.5148926
3: -675.8513184, 1323.3682861, -360.6497803, 697.4700317, -1373.3212891, 1684.0180664
4: -454.2720337, 1337.3880615, -244.7194672, 705.9050903, -1160.1771240, 1582.0769043

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750083, upper bound: 743.6764146
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6722005, upper bound: 743.6720897
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -238.2329407, 1275.9495850, -133.4752808, 698.6866455, -936.9195557, 1409.4248047
1: -392.8231812, 1514.6723633, -217.1052246, 829.5604248, -1222.3829346, 1731.7775879
2: -276.6023865, 1566.0113525, -154.3394012, 859.6849365, -1136.2873535, 1720.3507080
3: -678.3593750, 1328.2314453, -375.7047424, 726.6677856, -1405.0270996, 1703.9361572
4: -455.9827576, 1342.2722168, -254.8992767, 735.4954224, -1191.4780273, 1597.1607666

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793793, upper bound: 743.6785477
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774838, upper bound: 743.6766610
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742626, upper bound: 743.6721885
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -240.2502747, 1287.1490479, -127.8571014, 669.3228149, -909.5730591, 1415.0061035
1: -395.9108276, 1528.1837158, -207.9431763, 794.7456055, -1190.6563721, 1736.1269531
2: -278.9119873, 1579.5756836, -147.8582306, 823.5057983, -1102.4177246, 1727.4338379
3: -683.6232910, 1340.1411133, -359.9600220, 696.1035156, -1379.7268066, 1700.1010742
4: -459.7886353, 1353.7015381, -244.2486877, 704.5364990, -1164.3250732, 1597.9501953

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758410, upper bound: 743.6763303
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6772935, upper bound: 743.6775695
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -241.1236572, 1291.6192627, -133.2686310, 697.6209717, -938.7446289, 1424.8879395
1: -397.3216553, 1533.5045166, -216.7724152, 828.2872925, -1225.6088867, 1750.2769775
2: -279.9212036, 1585.0710449, -154.1020966, 858.3765869, -1138.2977295, 1739.1730957
3: -686.0750732, 1344.8690186, -375.1307373, 725.5369873, -1411.6120605, 1719.9997559
4: -461.4537659, 1358.4593506, -254.5068054, 734.3624268, -1195.8161621, 1612.9659424

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784932, upper bound: 743.6779933
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6778183, upper bound: 743.6776281
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -238.2329407, 1275.9495850, -241.1274872, 1284.9580078, -1523.1909180, 1517.0769043
1: -392.8231812, 1514.6723633, -395.7741699, 1525.5197754, -1918.3425293, 1910.4464111
2: -276.6023865, 1566.0113525, -279.7143250, 1577.7105713, -1854.3129883, 1845.7255859
3: -678.3593750, 1328.2314453, -684.9904785, 1338.7708740, -2017.1302490, 2013.2219238
4: -455.9827576, 1342.2722168, -461.4421082, 1353.3442383, -1809.3267822, 1803.7143555

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733878, upper bound: 743.6736185
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762779, upper bound: 743.6770523
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -238.2329407, 1275.9495850, -245.6122437, 1310.8828125, -1549.1157227, 1521.5617676
1: -392.8231812, 1514.6723633, -403.3295288, 1556.3529053, -1949.1756592, 1918.0019531
2: -276.6023865, 1566.0113525, -284.9976196, 1609.1540527, -1885.7564697, 1851.0090332
3: -678.3593750, 1328.2314453, -697.8605347, 1365.7233887, -2044.0827637, 2026.0920410
4: -455.9827576, 1342.2722168, -470.1723633, 1380.0172119, -1835.9998779, 1812.4445801

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733878, upper bound: 743.6736185
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762779, upper bound: 743.6770523
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -241.1236572, 1291.6192627, -240.9488220, 1284.0413818, -1525.1649170, 1532.5678711
1: -397.3216553, 1533.5045166, -395.4900208, 1524.4262695, -1921.7479248, 1928.9945068
2: -279.9212036, 1585.0710449, -279.5095215, 1576.5880127, -1856.5092773, 1864.5805664
3: -686.0750732, 1344.8690186, -684.4973755, 1337.8015137, -2023.8765869, 2029.3660889
4: -461.4537659, 1358.4593506, -461.1033630, 1352.3737793, -1813.8275146, 1819.5626221

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782853, upper bound: 743.6779756
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776104, upper bound: 743.6776104
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -241.1236572, 1291.6192627, -245.4285889, 1309.9379883, -1551.0615234, 1537.0478516
1: -397.3216553, 1533.5045166, -403.0364075, 1555.2268066, -1952.5484619, 1936.5407715
2: -279.9212036, 1585.0710449, -284.7868347, 1607.9970703, -1887.9182129, 1869.8579102
3: -686.0750732, 1344.8690186, -697.3526611, 1364.7255859, -2050.8002930, 2042.2216797
4: -461.4537659, 1358.4593506, -469.8246155, 1379.0159912, -1840.4696045, 1828.2839355

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782853, upper bound: 743.6779756
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776104, upper bound: 743.6776104
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.03 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824569
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824569
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824669
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6825492, upper bound: 743.6824669
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6878274, upper bound: 743.6879819
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6881568, upper bound: 743.6878969
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6878274, upper bound: 743.6879833
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6881568, upper bound: 743.6879224
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802518, upper bound: 743.6855479
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6798607, upper bound: 743.6818058
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802518, upper bound: 743.6861357
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6799781, upper bound: 743.6823936
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6817556, upper bound: 743.6864796
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6866200
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6817556, upper bound: 743.6872602
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6821096, upper bound: 743.6872725
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6270127, upper bound: 743.6295727
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6849190, upper bound: 743.6800165
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6236556, upper bound: 743.6292227
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6815619, upper bound: 743.6796665
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6827554, upper bound: 743.6824066
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6823936, upper bound: 743.6823641
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6827554, upper bound: 743.6824085
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6823936, upper bound: 743.6823659
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6226942, upper bound: 743.6299399
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802906, upper bound: 743.6802906
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6226942, upper bound: 743.6312672
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802906, upper bound: 743.6816178
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802683, upper bound: 743.6817473
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6799066, upper bound: 743.6817047
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6802683, upper bound: 743.6822461
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6799066, upper bound: 743.6822223
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6308395, upper bound: 743.6227417
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6687211, upper bound: 743.6687409
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6693150, upper bound: 743.6749183
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6694625, upper bound: 743.6715042
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6671573, upper bound: 743.6726637
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6673025, upper bound: 743.6692544
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6576330, upper bound: 743.6559065
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6761151, upper bound: 743.6761022
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6722829, upper bound: 743.6779561
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6724355, upper bound: 743.6745316
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6794935, upper bound: 743.6846308
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6794935, upper bound: 743.6846308
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6145465, upper bound: 743.6222926
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6780208, upper bound: 743.6771759
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6057726, upper bound: 743.6144950
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6779261, upper bound: 743.6773207
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6693572, upper bound: 743.6715137
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6692206, upper bound: 743.6712352
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6671996, upper bound: 743.6692639
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6670630, upper bound: 743.6689854
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6141143, upper bound: 743.6222215
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6773135, upper bound: 743.6770560
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6081458, upper bound: 743.6170209
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6775695, upper bound: 743.6772935
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6723251, upper bound: 743.6745411
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6721885, upper bound: 743.6742626
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6791806, upper bound: 743.6803797
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6791806, upper bound: 743.6818597
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6766522, upper bound: 743.6780035
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6771759, upper bound: 743.6780208
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6798421, upper bound: 743.6795350
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6800652, upper bound: 743.6793765
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6767077, upper bound: 743.6777003
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6773207, upper bound: 743.6779261
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6785019, upper bound: 743.6782874
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6782405, upper bound: 743.6783345
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771822
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770237
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771954
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770370
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6782940, upper bound: 743.6782697
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6780326, upper bound: 743.6783168
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6782940, upper bound: 743.6782697
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6780326, upper bound: 743.6783168
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6750083, upper bound: 743.6764146
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6722005, upper bound: 743.6720897
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6774838, upper bound: 743.6766610
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6742626, upper bound: 743.6721885
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6758410, upper bound: 743.6763303
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6772935, upper bound: 743.6775695
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6784932, upper bound: 743.6779933
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6778183, upper bound: 743.6776281
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6733878, upper bound: 743.6736185
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6762779, upper bound: 743.6770523
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6733878, upper bound: 743.6736185
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6762779, upper bound: 743.6770523
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6782853, upper bound: 743.6779756
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6776104, upper bound: 743.6776104
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6782853, upper bound: 743.6779756
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.03
Output dim: 0, lower bound: -743.6776104, upper bound: 743.6776104

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -96.9885025, 506.0981140, -107.1509323, 559.7775879, -656.7660522, 613.2490234
1: -158.8952179, 600.4038696, -175.4253082, 664.3189697, -823.2141724, 775.8291626
2: -112.2119446, 623.2306519, -123.9257889, 689.1486206, -801.3605347, 747.1564331
3: -273.8602295, 525.4290771, -302.4336243, 581.6336670, -855.4938965, 827.8626709
4: -184.6551361, 532.9910278, -203.9763184, 589.5694580, -774.2244873, 736.9672852

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6862966, upper bound: 743.6827388
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6862966, upper bound: 743.6828601
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -100.3710098, 523.8244629, -107.1509323, 559.7775879, -660.1485596, 630.9754028
1: -164.2590027, 621.7972412, -175.4253082, 664.3189697, -828.5780029, 797.2225342
2: -116.0657654, 644.9105225, -123.9257889, 689.1486206, -805.2143555, 768.8363037
3: -283.3581238, 544.3154907, -302.4336243, 581.6336670, -864.9918213, 846.7491455
4: -191.0703430, 551.7825928, -203.9763184, 589.5694580, -780.6397095, 755.7588501

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6862966, upper bound: 743.6827388
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6862966, upper bound: 743.6828601
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -96.9885025, 506.0981140, -112.6894531, 589.4049072, -686.3933716, 618.7875977
1: -158.8952179, 600.4038696, -184.3572845, 699.5338745, -858.4290161, 784.7609863
2: -112.2119446, 623.2306519, -130.3259277, 725.4063721, -837.6182861, 753.5565796
3: -273.8602295, 525.4290771, -318.0965271, 612.4905396, -886.3507690, 843.5256348
4: -184.6551361, 532.9910278, -214.5905151, 620.7008057, -805.3558350, 747.5815430

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687253, upper bound: 743.6692722
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807005, upper bound: 743.6808797
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -100.3710098, 523.8244629, -112.6894531, 589.4049072, -689.7758789, 636.5139160
1: -164.2590027, 621.7972412, -184.3572845, 699.5338745, -863.7928467, 806.1544189
2: -116.0657654, 644.9105225, -130.3259277, 725.4063721, -841.4721680, 775.2364502
3: -283.3581238, 544.3154907, -318.0965271, 612.4905396, -895.8485107, 862.4119873
4: -191.0703430, 551.7825928, -214.5905151, 620.7008057, -811.7711182, 766.3731079

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687253, upper bound: 743.6692722
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807005, upper bound: 743.6806894
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -100.3385391, 522.5124512, -101.8172531, 531.7440796, -632.0826416, 624.3297119
1: -164.1583099, 620.3449097, -166.8022614, 631.1797485, -795.3380737, 787.1471558
2: -115.9284592, 643.2648926, -117.8099060, 654.6407471, -770.5692139, 761.0747681
3: -282.9005737, 543.8035278, -287.4974365, 552.7945557, -835.6950684, 831.3009644
4: -190.9041138, 550.8670654, -193.9462280, 560.3420410, -751.2460938, 744.8132935

Time for backsubstitution: 1.99 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=860.0533447265625
rel_dist={0: [-743.6895975863799, 743.6895975863799]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831560, upper bound: 743.6848170
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -743.6831560, upper bound: 743.6848170
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.79
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -138.5269012, 721.5264282, -850.1054688, 808.8294678
1: -210.4309387, 796.4016113, -226.4326935, 857.1390381, -1067.5698242, 1022.8342896
2: -148.6245575, 824.4069214, -160.1910706, 887.5496826, -1036.1741943, 984.5979614
3: -362.4870300, 699.0101929, -390.1859741, 752.6910400, -1115.1779785, 1089.1961670
4: -244.7495728, 707.2666626, -263.8327942, 761.5472412, -1006.2968140, 971.0994263

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100
time: 0.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -250.1162415, 1337.2148438, -134.5297546, 702.8942261, -953.0104370, 1471.7445068
1: -412.2479858, 1588.0107422, -219.9175720, 834.7116089, -1246.9595947, 1807.9282227
2: -290.3746338, 1641.2213135, -155.6542053, 864.5371704, -1154.9118652, 1796.8753662
3: -711.9147949, 1393.7313232, -379.1771545, 732.3038330, -1444.2186279, 1772.9083252
4: -478.8499146, 1407.6657715, -256.3384705, 741.4115601, -1220.2612305, 1664.0042725

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804542, upper bound: 743.6790119
time: 0.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801126, upper bound: 743.6801126
time: 0.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 0, lower bound: -743.6815100, upper bound: 743.6815100
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 0, lower bound: -743.6804542, upper bound: 743.6790119
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.26
Output dim: 0, lower bound: -743.6801126, upper bound: 743.6801126

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -128.5790863, 670.3025513, -798.8816528, 798.8816528
1: -210.4309387, 796.4016113, -210.4309387, 796.4016113, -1006.8324585, 1006.8324585
2: -148.6245575, 824.4069214, -148.6245575, 824.4069214, -973.0314331, 973.0314331
3: -362.4870300, 699.0101929, -362.4870300, 699.0101929, -1061.4970703, 1061.4970703
4: -244.7495728, 707.2666626, -244.7495728, 707.2666626, -952.0162354, 952.0162354

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813499, upper bound: 743.6839560
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813373, upper bound: 743.6824221
time: 0.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -250.0711670, 1336.9914551, -1465.5705566, 920.3736572
1: -210.4309387, 796.4016113, -412.1743164, 1587.7460938, -1798.1770020, 1208.5759277
2: -148.6245575, 824.4069214, -290.3225403, 1640.9449463, -1789.5694580, 1114.7293701
3: -362.4870300, 699.0101929, -711.7879028, 1393.4953613, -1755.9822998, 1410.7980957
4: -244.7495728, 707.2666626, -478.7636414, 1407.4260254, -1652.1755371, 1186.0302734

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813499, upper bound: 743.6839560
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813373, upper bound: 743.6824221
time: 0.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -250.1162415, 1337.2148438, -117.6025543, 615.9862061, -866.1023560, 1454.8172607
1: -412.2479858, 1588.0107422, -192.1948242, 730.9327393, -1143.1806641, 1780.2055664
2: -290.3746338, 1641.2213135, -136.0982056, 758.0935059, -1048.4681396, 1777.3194580
3: -711.9147949, 1393.7313232, -331.7175598, 640.1293945, -1352.0441895, 1725.4488525
4: -478.8499146, 1407.6657715, -224.0883484, 649.0126953, -1127.8625488, 1631.7540283

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6624852, upper bound: 743.6620560
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -241.8035278, 1293.8023682, -139.2280731, 730.2432861, -972.0468140, 1433.0303955
1: -398.3430481, 1536.2239990, -226.5614319, 866.5744629, -1264.9174805, 1762.7852783
2: -280.7285767, 1587.9199219, -161.1650085, 898.6210327, -1179.3491211, 1749.0848389
3: -688.1906738, 1347.9317627, -392.0348816, 758.8276367, -1447.0183105, 1739.9664307
4: -462.9659729, 1361.6401367, -266.0524597, 768.7173462, -1231.6833496, 1627.6926270

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6813499, upper bound: 743.6839560
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6813373, upper bound: 743.6824221
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6813499, upper bound: 743.6839560
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6813373, upper bound: 743.6824221
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6624852, upper bound: 743.6620560
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.27
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -128.5790863, 670.3025513, -783.2761841, 719.3760986
1: -184.9509888, 701.4193115, -210.4309387, 796.4016113, -981.3526001, 911.8501587
2: -130.6758118, 727.0247803, -148.6245575, 824.4069214, -955.0827026, 875.6492920
3: -318.8685303, 614.4975586, -362.4870300, 699.0101929, -1017.8787231, 976.9845581
4: -215.1691284, 622.4899902, -244.7495728, 707.2666626, -922.4357910, 867.2395020

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -119.8804779, 625.7113647, -760.1862793, 824.5404053
1: -218.6857605, 836.5494995, -195.9631042, 743.2177124, -961.9034424, 1032.5124512
2: -155.4785309, 866.9288940, -138.5350189, 769.6444702, -925.1229248, 1005.4638062
3: -378.4638367, 732.4861450, -337.7071228, 651.7893066, -1030.2530518, 1070.1932373
4: -256.7464294, 741.4154663, -228.1236267, 659.7103882, -916.4567261, 969.5390625

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -249.9945068, 1336.6119385, -1449.5854492, 840.7914429
1: -184.9509888, 701.4193115, -412.0491943, 1587.2961426, -1772.2470703, 1113.4685059
2: -130.6758118, 727.0247803, -290.2339172, 1640.4764404, -1771.1522217, 1017.2586670
3: -318.8685303, 614.4975586, -711.5721436, 1393.0952148, -1711.9637451, 1326.0697021
4: -215.1691284, 622.4899902, -478.6170044, 1407.0190430, -1622.1879883, 1101.1068115

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802447, upper bound: 743.6825643
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799194, upper bound: 743.6825606
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -241.6986847, 1293.2805176, -1427.7553711, 946.3587036
1: -218.6857605, 836.5494995, -398.1716919, 1535.6054688, -1754.2912598, 1234.7210693
2: -155.4785309, 866.9288940, -280.6075134, 1587.2760010, -1742.7545166, 1147.5363770
3: -378.4638367, 732.4861450, -687.8954468, 1347.3818359, -1725.8455811, 1420.3813477
4: -256.7464294, 741.4154663, -462.7652588, 1361.0806885, -1617.7687988, 1204.1806641

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802756, upper bound: 743.6812932
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798928, upper bound: 743.6812912
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -247.2919464, 1322.5637207, -117.6025543, 615.9862061, -863.2780151, 1440.1660156
1: -407.6062622, 1570.5925293, -192.1948242, 730.9327393, -1138.5390625, 1762.7872314
2: -287.0827942, 1623.1872559, -136.0982056, 758.0935059, -1045.1762695, 1759.2852783
3: -703.9343872, 1378.3023682, -331.7175598, 640.1293945, -1344.0637207, 1710.0198975
4: -473.4362183, 1392.1020508, -224.0883484, 649.0126953, -1122.4488525, 1616.1900635

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -234.1127167, 1252.4116211, -138.9972992, 729.0382690, -963.1508789, 1391.4088135
1: -385.7048645, 1486.8763428, -226.1775818, 865.1339111, -1250.8387451, 1713.0537109
2: -271.7210693, 1537.4530029, -160.8949127, 897.1444092, -1168.8651123, 1698.3479004
3: -666.2803955, 1304.0643311, -391.3748779, 757.5421143, -1423.8225098, 1695.4392090
4: -447.9721680, 1317.9611816, -265.6047974, 767.4345093, -1215.4064941, 1583.5659180

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -238.0945129, 1275.5880127, -137.7148743, 722.2373657, -960.3318481, 1413.3026123
1: -392.3188171, 1514.2913818, -224.1061554, 857.0432739, -1249.3620605, 1738.3973389
2: -276.4233093, 1565.4890137, -159.4017639, 888.8259277, -1165.2491455, 1724.8907471
3: -677.7077026, 1327.9642334, -387.7702942, 750.3582764, -1428.0659180, 1715.7343750
4: -455.7666016, 1341.7694092, -263.1090698, 760.2684326, -1216.0350342, 1604.8784180

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6802447, upper bound: 743.6825643
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6799194, upper bound: 743.6825606
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6802756, upper bound: 743.6812932
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6798928, upper bound: 743.6812912
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6749224, upper bound: 743.6715404
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -112.9736404, 590.7969971, -703.7706299, 703.7706299
1: -184.9509888, 701.4193115, -184.9509888, 701.4193115, -886.3702393, 886.3703003
2: -130.6758118, 727.0247803, -130.6758118, 727.0247803, -857.7005615, 857.7005615
3: -318.8685303, 614.4975586, -318.8685303, 614.4975586, -933.3660278, 933.3660278
4: -215.1691284, 622.4899902, -215.1691284, 622.4899902, -837.6589966, 837.6589966

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6837792, upper bound: 743.6853119
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842412, upper bound: 743.6863911
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -134.4748993, 704.6600342, -817.6336670, 725.2719116
1: -184.9509888, 701.4193115, -218.6857605, 836.5494995, -1021.5004272, 920.1050415
2: -130.6758118, 727.0247803, -155.4785309, 866.9288940, -997.6045532, 882.5032349
3: -318.8685303, 614.4975586, -378.4638367, 732.4861450, -1051.3546143, 992.9613037
4: -215.1691284, 622.4899902, -256.7464294, 741.4154663, -956.5845947, 879.2362671

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6837792, upper bound: 743.6853119
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6842412, upper bound: 743.6863911
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -112.9736404, 590.7969971, -725.2719116, 817.6336670
1: -218.6857605, 836.5494995, -184.9509888, 701.4193115, -920.1051025, 1021.5004883
2: -155.4785309, 866.9288940, -130.6758118, 727.0247803, -882.5032349, 997.6045532
3: -378.4638367, 732.4861450, -318.8685303, 614.4975586, -992.9613037, 1051.3546143
4: -256.7464294, 741.4154663, -215.1691284, 622.4899902, -879.2362671, 956.5845337

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6824945, upper bound: 743.6815688
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -134.4748993, 704.6600342, -134.4748993, 704.6600342, -839.1348877, 839.1348877
1: -218.6857605, 836.5494995, -218.6857605, 836.5494995, -1055.2352295, 1055.2352295
2: -155.4785309, 866.9288940, -155.4785309, 866.9288940, -1022.4072266, 1022.4073486
3: -378.4638367, 732.4861450, -378.4638367, 732.4861450, -1110.9495850, 1110.9497070
4: -256.7464294, 741.4154663, -256.7464294, 741.4154663, -998.1618042, 998.1618042

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6824945, upper bound: 743.6815688
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -112.7308502, 589.5107422, -242.4403687, 1296.0229492, -1408.7537842, 831.9511108
1: -184.5492554, 699.8809814, -399.6437378, 1538.8968506, -1723.4460449, 1099.5244141
2: -130.3913879, 725.4604492, -281.3880005, 1590.9802246, -1721.3715820, 1006.8484497
3: -318.1757202, 613.1235962, -690.0591431, 1350.0394287, -1668.2148438, 1303.1827393
4: -214.6965790, 621.1246338, -463.8907166, 1364.1589355, -1578.8553467, 1085.0152588

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6691878, upper bound: 743.6726786
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6644689, upper bound: 743.6655167
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -111.8562088, 584.8630371, -246.6399078, 1320.1796875, -1432.0357666, 831.5029297
1: -183.1069946, 694.3749390, -406.6190796, 1567.4874268, -1750.5943604, 1100.9938965
2: -129.3710327, 719.6994629, -286.3486938, 1620.2257080, -1749.5966797, 1006.0481567
3: -315.6600342, 608.2840576, -702.1134033, 1375.0004883, -1690.6605225, 1310.3973389
4: -213.0032654, 616.2142334, -472.1006470, 1389.0323486, -1602.0352783, 1088.3149414

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6719979, upper bound: 743.6752804
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6678681, upper bound: 743.6690289
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -134.2406464, 703.4353027, -234.0027924, 1251.8658447, -1386.1063232, 937.4379272
1: -218.2982178, 835.0853271, -385.5253296, 1486.2292480, -1704.5273438, 1220.6104736
2: -155.2046356, 865.4285889, -271.5943298, 1536.7788086, -1691.9832764, 1137.0229492
3: -377.7966003, 731.1794434, -665.9708252, 1303.4887695, -1681.2854004, 1397.1500244
4: -256.2921753, 740.1130981, -447.7616882, 1317.3756104, -1573.5969238, 1187.8747559

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6692468, upper bound: 743.6709954
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6690921, upper bound: 743.6707585
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -133.0907135, 697.2184448, -237.9948883, 1275.0931396, -1408.1838379, 935.2132568
1: -216.4374847, 827.6925659, -392.1560364, 1513.7052002, -1730.1427002, 1219.8485107
2: -153.8671112, 857.8410034, -276.3081970, 1564.8771973, -1718.7442627, 1134.1490479
3: -374.5537415, 724.6454468, -677.4267578, 1327.4427490, -1701.9964600, 1402.0722656
4: -254.0527649, 733.6042480, -455.5760498, 1341.2382812, -1595.2910156, 1189.1802979

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6721670, upper bound: 743.6739475
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6721034, upper bound: 743.6737945
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -247.1703949, 1321.9611816, -112.9736404, 590.7969971, -837.9674072, 1434.9348145
1: -407.4075012, 1569.8782959, -184.9509888, 701.4193115, -1108.8267822, 1754.8293457
2: -286.9421997, 1622.4426270, -130.6758118, 727.0247803, -1013.9669800, 1753.1184082
3: -703.5916748, 1377.6669922, -318.8685303, 614.4975586, -1318.0891113, 1696.5355225
4: -473.2034912, 1391.4558105, -215.1691284, 622.4899902, -1095.6933594, 1606.6248779

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720339, upper bound: 743.6693217
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733336, upper bound: 743.6698634
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -247.2919464, 1322.5637207, -233.4036713, 1251.2222900, -1498.5142822, 1555.9671631
1: -407.6062622, 1570.5925293, -384.9095154, 1485.3535156, -1892.9595947, 1955.5017090
2: -287.0827942, 1623.1872559, -271.0440063, 1535.6855469, -1822.7683105, 1894.2312012
3: -703.9343872, 1378.3023682, -665.0811768, 1302.4888916, -2006.4233398, 2043.3834229
4: -473.4362183, 1392.1020508, -447.0042725, 1316.2604980, -1789.6967773, 1839.1059570

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720339, upper bound: 743.6693217
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733336, upper bound: 743.6698634
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -234.0027924, 1251.8658447, -135.9503326, 711.9039917, -945.9066772, 1387.8160400
1: -385.5253296, 1486.2292480, -221.1683655, 845.2096558, -1230.7347412, 1707.3975830
2: -271.5943298, 1536.7788086, -157.2134399, 875.8896484, -1147.4840088, 1693.9920654
3: -665.9708252, 1303.4887695, -382.7561340, 740.3925781, -1406.3632812, 1686.2448730
4: -447.7616882, 1317.3756104, -259.6274414, 749.3641968, -1197.1257324, 1576.9403076

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -234.1127167, 1252.4116211, -248.5480194, 1325.0283203, -1559.1409912, 1500.9595947
1: -385.7048645, 1486.8763428, -407.9991150, 1573.3110352, -1959.0158691, 1894.8753662
2: -271.7210693, 1537.4530029, -288.3969727, 1626.5987549, -1898.3194580, 1825.8499756
3: -666.2803955, 1304.0643311, -706.0611572, 1381.1342773, -2047.4144287, 2010.1254883
4: -447.9721680, 1317.9611816, -475.8623047, 1395.4949951, -1843.4669189, 1793.8232422

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -237.9948883, 1275.0931396, -134.8095093, 705.7186890, -943.7135010, 1409.9025879
1: -392.1560364, 1513.7052002, -219.3261414, 837.8544922, -1230.0104980, 1733.0313721
2: -276.3081970, 1564.8771973, -155.8865051, 868.3437500, -1144.6519775, 1720.7636719
3: -677.4267578, 1327.4427490, -379.5442505, 733.8942261, -1411.3210449, 1706.9870605
4: -455.5760498, 1341.2382812, -257.4041443, 742.8944092, -1198.4704590, 1598.6422119

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -238.0945129, 1275.5880127, -247.6035004, 1320.2728271, -1558.3673096, 1523.1912842
1: -392.3188171, 1514.2913818, -406.4921570, 1567.6557617, -1959.9746094, 1920.7835693
2: -276.4233093, 1565.4890137, -287.3008118, 1620.7482910, -1897.1711426, 1852.7897949
3: -677.7077026, 1327.9642334, -703.4101562, 1376.1145020, -2053.8217773, 2031.3743896
4: -455.7666016, 1341.7694092, -474.0431519, 1390.4128418, -1846.1794434, 1815.8125000

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.48 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6837792, upper bound: 743.6853119
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6842412, upper bound: 743.6863911
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6837792, upper bound: 743.6853119
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6842412, upper bound: 743.6863911
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6824945, upper bound: 743.6815688
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6824945, upper bound: 743.6815688
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6841634, upper bound: 743.6841634
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6691878, upper bound: 743.6726786
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6644689, upper bound: 743.6655167
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6719979, upper bound: 743.6752804
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6678681, upper bound: 743.6690289
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6692468, upper bound: 743.6709954
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6690921, upper bound: 743.6707585
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6721670, upper bound: 743.6739475
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6721034, upper bound: 743.6737945
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6720339, upper bound: 743.6693217
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6733336, upper bound: 743.6698634
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6720339, upper bound: 743.6693217
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6733336, upper bound: 743.6698634
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.48
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -110.8724747, 579.8894043, -681.7066650, 642.6165771
1: -166.8022614, 631.1797485, -181.5839996, 688.4622192, -855.2644653, 812.7637329
2: -117.8099060, 654.6407471, -128.2591858, 713.5875244, -831.3973999, 782.8999023
3: -287.4974365, 552.7945557, -313.0301514, 603.0814819, -890.5789185, 865.8247070
4: -193.9462280, 560.3420410, -211.1773682, 610.9684448, -804.9146729, 771.5193481

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6826008
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826130, upper bound: 743.6824652
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -112.4618683, 588.1077881, -698.0656738, 687.2977295
1: -180.0325623, 682.4268188, -184.1177216, 698.2144165, -878.2469482, 866.5445557
2: -127.1877441, 707.4198608, -130.0846252, 723.7250366, -850.9127808, 837.5045166
3: -310.3284607, 597.7552490, -317.4218750, 611.6633911, -921.9916992, 915.1771240
4: -209.4055481, 605.6980591, -214.1909485, 619.6528931, -829.0584717, 819.8889160

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6828174
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826289, upper bound: 743.6826289
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -101.8172531, 531.7440796, -132.3188629, 693.3843384, -795.2015991, 664.0628662
1: -166.8022614, 631.1797485, -215.2234497, 823.1541138, -989.9563599, 846.4031982
2: -117.8099060, 654.6407471, -153.0005798, 853.0865479, -970.8964233, 807.6413574
3: -287.4974365, 552.7945557, -372.5046692, 720.6976929, -1008.1951294, 925.2991333
4: -193.9462280, 560.3420410, -252.6539307, 729.5548706, -923.5010986, 812.9959106

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6823433, upper bound: 743.6823197
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6823426, upper bound: 743.6823029
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -134.0515442, 702.4326782, -812.3905029, 708.8873901
1: -180.0325623, 682.4268188, -217.9949341, 833.9020996, -1013.9346313, 900.4217529
2: -127.1877441, 707.4198608, -154.9886780, 864.2003174, -991.3880615, 862.4085693
3: -310.3284607, 597.7552490, -377.2652588, 730.1502075, -1040.4786377, 975.0204468
4: -209.4055481, 605.6980591, -255.9379883, 739.0676270, -948.4731445, 861.6360474

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816112, upper bound: 743.6842317
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816112, upper bound: 743.6863911
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -110.8724747, 579.8894043, -706.5637207, 774.4494629
1: -205.9380188, 787.8771362, -181.5839996, 688.4622192, -894.4002686, 969.4611206
2: -146.4624939, 816.3840942, -128.2591858, 713.5875244, -860.0500488, 944.6433105
3: -356.4977722, 689.8046875, -313.0301514, 603.0814819, -959.5792236, 1002.8348389
4: -241.9350891, 698.1940918, -211.1773682, 610.9684448, -852.9035645, 909.3713379

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6829841, upper bound: 743.6796203
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6818401, upper bound: 743.6798457
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -112.4618683, 588.1077881, -720.1740112, 804.1802979
1: -214.7357178, 821.2271729, -184.1177216, 698.2144165, -912.9501343, 1005.3449097
2: -152.6842041, 851.0744629, -130.0846252, 723.7250366, -876.4092407, 981.1590576
3: -371.6125488, 719.0783691, -317.4218750, 611.6633911, -983.2759399, 1036.5002441
4: -252.1509399, 727.8773804, -214.1909485, 619.6528931, -871.8037720, 942.0682983

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6847063, upper bound: 743.6825479
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825137, upper bound: 743.6823579
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -126.6743240, 663.5769653, -132.3188629, 693.3843384, -820.0586548, 795.8956909
1: -205.9380188, 787.8771362, -215.2234497, 823.1541138, -1029.0921631, 1003.1005859
2: -146.4624939, 816.3840942, -153.0005798, 853.0865479, -999.5490723, 969.3846436
3: -356.4977722, 689.8046875, -372.5046692, 720.6976929, -1077.1953125, 1062.3093262
4: -241.9350891, 698.1940918, -252.6539307, 729.5548706, -971.4899902, 950.8479614

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808580, upper bound: 743.6792723
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6811141, upper bound: 743.6796523
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -134.0515442, 702.4326782, -834.4989014, 825.7699585
1: -214.7357178, 821.2271729, -217.9949341, 833.9020996, -1048.6375732, 1039.2221680
2: -152.6842041, 851.0744629, -154.9886780, 864.2003174, -1016.8844604, 1006.0630493
3: -371.6125488, 719.0783691, -377.2652588, 730.1502075, -1101.7625732, 1096.3436279
4: -252.1509399, 727.8773804, -255.9379883, 739.0676270, -991.2185059, 983.8153687

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816271, upper bound: 743.6824945
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816271, upper bound: 743.6841634
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -106.6491776, 557.0265503, -245.9369354, 1316.4085693, -1423.0577393, 802.9635010
1: -174.5665131, 661.0840454, -405.4624939, 1563.0124512, -1737.5789795, 1066.5465088
2: -123.3279953, 685.7707520, -285.5304871, 1615.6025391, -1738.9305420, 971.3012695
3: -300.9313660, 578.7736206, -700.1190186, 1371.0561523, -1671.9874268, 1278.8925781
4: -202.9776459, 586.6771240, -470.7452087, 1385.0557861, -1588.0334473, 1057.4222412

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6560710, upper bound: 743.6580244
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6719979, upper bound: 743.6752804
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6479223, upper bound: 743.6467335
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6712914, upper bound: 743.6736922
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -128.4824677, 673.2764893, -237.2514801, 1271.1210938, -1399.6035156, 910.5278931
1: -208.8581543, 799.1115112, -390.9345703, 1508.9925537, -1717.8507080, 1190.0458984
2: -148.5339050, 828.4558105, -275.4432373, 1560.0041504, -1708.5380859, 1103.8990479
3: -361.5465698, 699.2052612, -675.3197632, 1323.2871094, -1684.8337402, 1374.5249023
4: -245.2442322, 708.1198730, -454.1433716, 1337.0460205, -1582.2900391, 1162.2631836

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6721670, upper bound: 743.6739475
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6615683, upper bound: 743.6631783
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6714489, upper bound: 743.6734282
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -134.1256256, 701.2588501, -236.7171173, 1268.0472412, -1402.1727295, 937.9759521
1: -217.7872162, 832.5407104, -390.0391235, 1505.3530273, -1723.1402588, 1222.5797119
2: -155.0041962, 862.9439087, -274.8130798, 1556.2519531, -1711.2561035, 1137.7569580
3: -377.1127625, 728.9760742, -673.7623901, 1320.1090088, -1697.2216797, 1402.7384033
4: -256.0598450, 737.6608887, -453.1065369, 1333.8542480, -1589.9138184, 1190.7673340

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6721034, upper bound: 743.6737945
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6633260, upper bound: 743.6644261
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6710899, upper bound: 743.6731890
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -228.1680603, 1220.8045654, -135.1497345, 707.7058105, -935.8739014, 1355.9543457
1: -376.0632629, 1449.1357422, -219.8629150, 840.1950684, -1216.2583008, 1668.9986572
2: -264.8314514, 1498.7453613, -156.2884979, 870.7460327, -1135.5773926, 1655.0336914
3: -649.6043701, 1270.6430664, -380.5076599, 735.9333496, -1385.5377197, 1651.1507568
4: -436.5395813, 1284.6467285, -258.0939331, 744.9163818, -1181.4558105, 1542.7133789

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709954, upper bound: 743.6692468
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6707585, upper bound: 743.6690921
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -231.7823181, 1240.4442139, -134.9353943, 706.6663208, -938.4486084, 1375.3795166
1: -381.7567139, 1472.6444092, -219.5348358, 838.9530640, -1220.7097168, 1692.1791992
2: -269.0010071, 1522.7327881, -156.0481110, 869.4616089, -1138.4626465, 1678.7808838
3: -659.5261841, 1291.3950195, -379.9378967, 734.8337402, -1394.3598633, 1671.3328857
4: -443.4151611, 1305.1009521, -257.6999512, 743.7962646, -1187.2114258, 1562.8007812

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687429, upper bound: 743.6670832
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6685164, upper bound: 743.6669214
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -228.2903748, 1221.4112549, -247.7070007, 1320.5257568, -1548.8160400, 1469.1181641
1: -376.2631531, 1449.8553467, -406.6309814, 1567.9364014, -1944.1995850, 1856.4863281
2: -264.9724731, 1499.4958496, -287.4219666, 1621.0898438, -1886.0621338, 1786.9177246
3: -649.9490967, 1271.2828369, -703.7034912, 1376.3798828, -2026.3287354, 1974.9863281
4: -436.7735901, 1285.2982178, -474.2442627, 1390.7633057, -1827.5368652, 1759.5424805

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -231.8300323, 1240.6802979, -247.6737366, 1320.5225830, -1552.3524170, 1488.3540039
1: -381.8343506, 1472.9246826, -406.6047668, 1567.9399414, -1949.7742920, 1879.5294189
2: -269.0559692, 1523.0240479, -287.3927612, 1621.0802002, -1890.1362305, 1810.4167480
3: -659.6599121, 1291.6446533, -703.6442261, 1376.3773193, -2036.0372314, 1995.2885742
4: -443.5063782, 1305.3544922, -474.2022095, 1390.7286377, -1834.2348633, 1779.5565186

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -232.2552948, 1244.6214600, -134.0093079, 701.5220947, -933.7772827, 1378.6307373
1: -382.8396912, 1477.3162842, -218.0206146, 832.8422241, -1215.6818848, 1695.3367920
2: -269.6584473, 1527.5560303, -154.9617004, 863.2020874, -1132.8604736, 1682.5177002
3: -661.3165283, 1295.2194824, -377.2954407, 729.4368896, -1390.7532959, 1672.5147705
4: -444.5473633, 1309.1094971, -255.8708649, 738.4479370, -1182.9952393, 1564.9801025

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739475, upper bound: 743.6721670
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737945, upper bound: 743.6721034
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -235.2493896, 1260.7008057, -133.7964630, 700.4891357, -935.7384033, 1394.4971924
1: -387.4753418, 1496.6846924, -217.6956787, 831.6082764, -1219.0836182, 1714.3803711
2: -273.0856323, 1547.0997314, -154.7227020, 861.9245605, -1135.0102539, 1701.8223877
3: -669.2602539, 1312.4061279, -376.7311707, 728.3449707, -1397.6052246, 1689.1370850
4: -450.2021790, 1325.7854004, -255.4794006, 737.3348389, -1187.5369873, 1581.2647705

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799576, upper bound: 743.6792152
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798261, upper bound: 743.6790458
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -232.3674927, 1245.1796875, -246.7635803, 1315.7788086, -1548.1459961, 1491.9432373
1: -383.0233459, 1477.9774170, -405.1253967, 1562.2897949, -1945.3131104, 1883.1027832
2: -269.7881775, 1528.2460938, -286.3269043, 1615.2491455, -1885.0373535, 1814.5727539
3: -661.6334839, 1295.8068848, -701.0546265, 1371.3670654, -2033.0004883, 1996.8615723
4: -444.7619629, 1309.7084961, -472.4271240, 1385.6883545, -1830.4501953, 1782.1356201

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -235.2903442, 1260.9036865, -246.7353973, 1315.7934570, -1551.0836182, 1507.6390381
1: -387.5422058, 1496.9259033, -405.1068726, 1562.3157959, -1949.8580322, 1902.0327148
2: -273.1328430, 1547.3504639, -286.3029480, 1615.2628174, -1888.3956299, 1833.6531982
3: -669.3754883, 1312.6206055, -701.0077515, 1371.3862305, -2040.7614746, 2013.6284180
4: -450.2804260, 1326.0034180, -472.3934631, 1385.6756592, -1835.9558105, 1798.3968506

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6826008
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6826130, upper bound: 743.6824652
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6828174
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6826289, upper bound: 743.6826289
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6823433, upper bound: 743.6823197
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6823426, upper bound: 743.6823029
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6816112, upper bound: 743.6842317
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6816112, upper bound: 743.6863911
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6829841, upper bound: 743.6796203
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6818401, upper bound: 743.6798457
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6847063, upper bound: 743.6825479
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6825137, upper bound: 743.6823579
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6808580, upper bound: 743.6792723
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6811141, upper bound: 743.6796523
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6816271, upper bound: 743.6824945
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6816271, upper bound: 743.6841634
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6479223, upper bound: 743.6467335
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6712914, upper bound: 743.6736922
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6615683, upper bound: 743.6631783
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6714489, upper bound: 743.6734282
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6633260, upper bound: 743.6644261
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6710899, upper bound: 743.6731890
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6709954, upper bound: 743.6692468
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6707585, upper bound: 743.6690921
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6687429, upper bound: 743.6670832
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6685164, upper bound: 743.6669214
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6773839, upper bound: 743.6784837
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6787265, upper bound: 743.6790187
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6739475, upper bound: 743.6721670
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6737945, upper bound: 743.6721034
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6799576, upper bound: 743.6792152
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6798261, upper bound: 743.6790458
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6771325, upper bound: 743.6775730
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.51
Output dim: 0, lower bound: -743.6787181, upper bound: 743.6787181

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -101.4000015, 529.5228882, -105.6504974, 551.9562378, -653.3561401, 635.1734009
1: -166.1188049, 628.5053711, -173.0208282, 655.0476685, -821.1664429, 801.5261841
2: -117.3259125, 651.9213867, -122.2005386, 679.5362549, -796.8621826, 774.1219482
3: -286.3185120, 550.4246216, -298.2593384, 573.4670410, -859.7854614, 848.6839600
4: -193.1430359, 557.9732056, -201.1262665, 581.3231812, -774.4661865, 759.0994263

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6824808, upper bound: 743.6824440
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6824808, upper bound: 743.6824440
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -100.3938141, 524.0641479, -111.1111984, 581.2710571, -681.6648560, 635.1752930
1: -164.4435730, 622.1174927, -181.8360748, 689.8695068, -854.3131104, 803.9535522
2: -116.1450043, 645.2062378, -128.5153961, 715.3720703, -831.5169678, 773.7216187
3: -283.4144287, 544.7871704, -313.7316895, 603.9739380, -887.3883667, 858.5188599
4: -191.1868896, 552.2608643, -211.5987244, 612.1093140, -803.2961426, 763.8596191

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6825021, upper bound: 743.6817793
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821649, upper bound: 743.6821307
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -109.5107651, 572.4301758, -107.2559433, 560.2904663, -669.8011475, 679.6860962
1: -179.2993469, 679.5592651, -175.5781555, 664.9136963, -844.2130127, 855.1374512
2: -126.6690750, 704.4980469, -124.0427551, 689.7896118, -816.4586792, 828.5407715
3: -309.0639954, 595.2164307, -302.6978760, 582.1488647, -891.2128296, 897.9143066
4: -208.5454254, 603.1550903, -204.1676178, 590.1135254, -798.6589355, 807.3226929

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6263031, upper bound: 743.6349414
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6828174
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -108.6553268, 567.8551025, -112.8455200, 590.1755981, -698.8308105, 680.7006226
1: -177.8773193, 674.1298828, -184.5921631, 700.4442749, -878.3215332, 858.7220459
2: -125.6650238, 698.7783203, -130.5016632, 726.3682861, -852.0333252, 829.2799683
3: -306.5971375, 590.4273071, -318.4997253, 613.2855835, -919.8826904, 908.9270020
4: -206.8818665, 598.3073120, -214.8813171, 621.5140991, -828.3959961, 813.1885986

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6265590, upper bound: 743.6358072
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826289, upper bound: 743.6826289
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -101.4000015, 529.5228882, -127.7504807, 669.7043457, -771.1042480, 657.2731323
1: -166.1188049, 628.5053711, -207.7281189, 794.8952637, -961.0139771, 836.2335205
2: -117.3259125, 651.9213867, -147.7208862, 824.0305176, -941.3564453, 799.6422119
3: -286.3185120, 550.4246216, -359.6256714, 695.5582275, -981.8767090, 910.0502930
4: -193.1430359, 557.9732056, -243.9409332, 704.3534546, -897.4964600, 801.9141235

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820234, upper bound: 743.6822862
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820234, upper bound: 743.6822862
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -100.3938141, 524.0641479, -133.1937103, 696.5796509, -796.9734497, 657.2575684
1: -164.4435730, 622.1174927, -216.3176880, 826.9788818, -991.4224854, 838.4351807
2: -116.1450043, 645.2062378, -153.9565582, 857.1578979, -973.3027344, 799.1627808
3: -283.4144287, 544.7871704, -374.6120300, 724.1007690, -1007.5151978, 919.3991699
4: -191.1868896, 552.2608643, -254.3512726, 732.6951294, -923.8818970, 806.6121216

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820981, upper bound: 743.6817207
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817716, upper bound: 743.6820718
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -126.6743240, 663.5769653, -773.5347900, 701.5103149
1: -180.0325623, 682.4268188, -205.9380188, 787.8771362, -967.9096680, 888.3648682
2: -127.1877441, 707.4198608, -146.4624939, 816.3840942, -943.5718384, 853.8823242
3: -310.3284607, 597.7552490, -356.4977722, 689.8046875, -1000.1330566, 954.2529907
4: -209.4055481, 605.6980591, -241.9350891, 698.1940918, -907.5996094, 847.6331787

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796203, upper bound: 743.6829841
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798457, upper bound: 743.6818401
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.9578705, 574.8359985, -132.0662384, 691.7185059, -801.6763306, 706.9021606
1: -180.0325623, 682.4268188, -214.7357178, 821.2271729, -1001.2597046, 897.1625366
2: -127.1877441, 707.4198608, -152.6842041, 851.0744629, -978.2622070, 860.1040649
3: -310.3284607, 597.7552490, -371.6125488, 719.0783691, -1029.4067383, 969.3677979
4: -209.4055481, 605.6980591, -252.1509399, 727.8773804, -937.2829590, 857.8488159

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796203, upper bound: 743.6845942
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798457, upper bound: 743.6823021
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -126.3043671, 661.6539307, -105.6504974, 551.9562378, -678.2604980, 767.3044434
1: -205.3329620, 785.5828857, -173.0208282, 655.0476685, -860.3806152, 958.6035767
2: -146.0361176, 814.0266113, -122.2005386, 679.5362549, -825.5723877, 936.2271729
3: -355.4576721, 687.7661133, -298.2593384, 573.4670410, -928.9245605, 986.0253296
4: -241.2312317, 696.1493530, -201.1262665, 581.3231812, -822.5544434, 897.2756348

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6154269, upper bound: 743.6215897
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826082, upper bound: 743.6794428
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -125.1267014, 655.2334595, -111.1111984, 581.2710571, -706.3976440, 766.3445435
1: -203.3942261, 777.9520264, -181.8360748, 689.8695068, -893.2637329, 959.7880859
2: -144.6620483, 806.1571655, -128.5153961, 715.3720703, -860.0341187, 934.6725464
3: -352.1151123, 681.0187988, -313.7316895, 603.9739380, -956.0890503, 994.7504883
4: -238.9502411, 689.3666382, -211.5987244, 612.1093140, -851.0595703, 900.9653320

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6147494, upper bound: 743.6225773
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814547, upper bound: 743.6795412
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -131.6514893, 689.5535278, -107.2559433, 560.2904663, -691.9419556, 796.8094482
1: -214.0560455, 818.6406860, -175.5781555, 664.9136963, -878.9696655, 994.2187500
2: -152.2043610, 848.4185181, -124.0427551, 689.7896118, -841.9937744, 972.4613037
3: -370.4444275, 716.7744141, -302.6978760, 582.1488647, -952.5932617, 1019.4721069
4: -251.3567657, 725.5758057, -204.1676178, 590.1135254, -841.4702148, 929.7434082

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744186, upper bound: 743.6706707
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6828753, upper bound: 743.6804406
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -130.7837677, 684.7760010, -112.8455200, 590.1755981, -720.9593506, 797.6215210
1: -212.6269226, 812.9764404, -184.5921631, 700.4442749, -913.0711670, 997.5686035
2: -151.1917114, 842.5675659, -130.5016632, 726.3682861, -877.5599976, 973.0692139
3: -367.9747925, 711.7839355, -318.4997253, 613.2855835, -981.2603149, 1030.2836914
4: -249.6775665, 720.5296631, -214.8813171, 621.5140991, -871.1916504, 935.4109497

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6718418, upper bound: 743.6705829
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804869, upper bound: 743.6806209
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -126.3043671, 661.6539307, -127.7504807, 669.7043457, -796.0085449, 789.4041748
1: -205.3329620, 785.5828857, -207.7281189, 794.8952637, -1000.2282104, 993.3109131
2: -146.0361176, 814.0266113, -147.7208862, 824.0305176, -970.0666504, 961.7474365
3: -355.4576721, 687.7661133, -359.6256714, 695.5582275, -1051.0157471, 1047.3917236
4: -241.2312317, 696.1493530, -243.9409332, 704.3534546, -945.5847168, 940.0902710

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708105, upper bound: 743.6725214
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793550, upper bound: 743.6776899
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6560477, upper bound: 743.6555154
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -125.1267014, 655.2334595, -133.1937103, 696.5796509, -821.7062378, 788.4269409
1: -203.3942261, 777.9520264, -216.3176880, 826.9788818, -1030.3730469, 994.2697144
2: -144.6620483, 806.1571655, -153.9565582, 857.1578979, -1001.8199463, 960.1137085
3: -352.1151123, 681.0187988, -374.6120300, 724.1007690, -1076.2155762, 1055.6308594
4: -238.9502411, 689.3666382, -254.3512726, 732.6951294, -971.6453857, 943.7178955

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6136354, upper bound: 743.6225526
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807930, upper bound: 743.6794176
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -126.6743240, 663.5769653, -795.6431885, 818.3928223
1: -214.7357178, 821.2271729, -205.9380188, 787.8771362, -1002.6128540, 1027.1651611
2: -152.6842041, 851.0744629, -146.4624939, 816.3840942, -969.0682983, 997.5369873
3: -371.6125488, 719.0783691, -356.4977722, 689.8046875, -1061.4171143, 1075.5760498
4: -252.1509399, 727.8773804, -241.9350891, 698.1940918, -950.3448486, 969.8125000

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792723, upper bound: 743.6808580
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797419, upper bound: 743.6811368
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -132.0662384, 691.7185059, -132.0662384, 691.7185059, -823.7847290, 823.7847290
1: -214.7357178, 821.2271729, -214.7357178, 821.2271729, -1035.9626465, 1035.9626465
2: -152.6842041, 851.0744629, -152.6842041, 851.0744629, -1003.7586060, 1003.7586670
3: -371.6125488, 719.0783691, -371.6125488, 719.0783691, -1090.6907959, 1090.6907959
4: -252.1509399, 727.8773804, -252.1509399, 727.8773804, -980.0282593, 980.0282593

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792723, upper bound: 743.6822032
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797419, upper bound: 743.6821719
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -228.2903748, 1221.4112549, -240.2712097, 1280.3796387, -1508.6700439, 1461.6821289
1: -376.2631531, 1449.8553467, -394.3832092, 1520.0517578, -1896.3149414, 1844.2382812
2: -264.9724731, 1499.4958496, -278.7207947, 1572.1082764, -1837.0805664, 1778.2166748
3: -649.9490967, 1271.2828369, -682.5899658, 1333.9282227, -1983.8773193, 1953.8728027
4: -436.7735901, 1285.2982178, -459.7922668, 1348.5299072, -1785.3034668, 1745.0904541

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771749
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770173
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -228.2903748, 1221.4112549, -244.7745209, 1306.3995361, -1534.6899414, 1466.1857910
1: -376.2631531, 1449.8553467, -401.9654541, 1551.0014648, -1927.2646484, 1851.8208008
2: -264.9724731, 1499.4958496, -284.0263367, 1603.6665039, -1868.6387939, 1783.5222168
3: -649.9490967, 1271.2828369, -695.5081787, 1360.9925537, -2010.9416504, 1966.7906494
4: -436.7735901, 1285.2982178, -468.5613403, 1375.3035889, -1812.0771484, 1753.8593750

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771879
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770304
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -231.8300323, 1240.6802979, -240.2576447, 1280.4862061, -1512.3161621, 1480.9379883
1: -381.8343506, 1472.9246826, -394.3895264, 1520.1840820, -1902.0184326, 1867.3142090
2: -269.0559692, 1523.0240479, -278.7161865, 1572.2347412, -1841.2907715, 1801.7402344
3: -659.6599121, 1291.6446533, -682.5869751, 1334.0407715, -1993.7006836, 1974.2316895
4: -443.5063782, 1305.3544922, -459.7908325, 1348.6101074, -1792.1164551, 1765.1452637

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782841, upper bound: 743.6782573
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779870, upper bound: 743.6782591
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -231.8300323, 1240.6802979, -244.7313385, 1306.3363037, -1538.1663818, 1485.4116211
1: -381.8343506, 1472.9246826, -401.9236145, 1550.9343262, -1932.7686768, 1874.8480225
2: -269.0559692, 1523.0240479, -283.9859924, 1603.5853271, -1872.6413574, 1807.0100098
3: -659.6599121, 1291.6446533, -695.4236450, 1360.9248047, -2020.5847168, 1987.0682373
4: -443.5063782, 1305.3544922, -468.5013123, 1375.2053223, -1818.7116699, 1773.8554688

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782841, upper bound: 743.6782573
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779870, upper bound: 743.6782591
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -231.5114441, 1240.6448975, -129.4132843, 677.6606445, -909.1719971, 1370.0581055
1: -381.6169434, 1472.5987549, -210.4622345, 804.3563232, -1185.9731445, 1683.0609131
2: -268.7930908, 1522.6762695, -149.6426239, 833.9105835, -1102.7034912, 1672.3188477
3: -659.2064209, 1291.0607910, -364.3191223, 704.0768433, -1363.2832031, 1655.3798828
4: -443.1143188, 1304.9123535, -247.0837555, 713.0368652, -1156.1510010, 1551.9957275

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6691124, upper bound: 743.6683263
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6631783, upper bound: 743.6615363
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734282, upper bound: 743.6714489
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -230.9785004, 1237.5776367, -135.2377625, 706.5278931, -937.5063477, 1372.8151855
1: -380.7247314, 1468.9672852, -219.6894836, 838.8442383, -1219.5688477, 1688.6567383
2: -268.1645508, 1518.9331055, -156.3261414, 869.4939575, -1137.6584473, 1675.2591553
3: -657.6546021, 1287.8894043, -380.4024658, 734.8089600, -1392.4636230, 1668.2916260
4: -442.0801392, 1301.7291260, -258.2532654, 743.5469971, -1185.6270752, 1559.9824219

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6702104, upper bound: 743.6693669
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6689015, upper bound: 743.6686086
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -234.4634705, 1256.5867920, -124.3867493, 651.4295044, -885.8928833, 1380.9735107
1: -386.2150879, 1491.7841797, -202.2288208, 773.2192993, -1159.4343262, 1694.0129395
2: -272.1780396, 1542.0625000, -143.7606049, 801.6468506, -1073.8249512, 1685.8228760
3: -667.0693359, 1308.0584717, -350.1657715, 676.6793213, -1343.7482910, 1658.2242432
4: -448.6994934, 1321.4244385, -237.4137878, 685.2299194, -1133.9293213, 1558.8382568

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770030, upper bound: 743.6770631
time: 3.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771456, upper bound: 743.6772115
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -234.8689880, 1258.7294922, -129.9771271, 680.2059937, -915.0749512, 1388.7064209
1: -386.8527222, 1494.3305664, -211.4386597, 807.4907227, -1194.3431396, 1705.7692871
2: -272.6446838, 1544.6824951, -150.2943878, 837.0983276, -1109.7430420, 1694.9769287
3: -668.1803589, 1310.3112793, -365.8997803, 707.0587158, -1375.2388916, 1676.2110596
4: -449.4717102, 1323.6857910, -248.1492462, 715.9124146, -1165.3841553, 1571.8349609

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770290, upper bound: 743.6770097
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771712, upper bound: 743.6771581
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -232.3674927, 1245.1796875, -240.2712097, 1280.3796387, -1512.7470703, 1485.4506836
1: -383.0233459, 1477.9774170, -394.3832092, 1520.0517578, -1903.0750732, 1872.3604736
2: -269.7881775, 1528.2460938, -278.7207947, 1572.1082764, -1841.8964844, 1806.9669189
3: -661.6334839, 1295.8068848, -682.5899658, 1333.9282227, -1995.5617676, 1978.3968506
4: -444.7619629, 1309.7084961, -459.7922668, 1348.5299072, -1793.2918701, 1769.5007324

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6731900, upper bound: 743.6732604
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762573, upper bound: 743.6770438
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -232.3674927, 1245.1796875, -244.7745209, 1306.3995361, -1538.7669678, 1489.9542236
1: -383.0233459, 1477.9774170, -401.9654541, 1551.0014648, -1934.0247803, 1879.9427490
2: -269.7881775, 1528.2460938, -284.0263367, 1603.6665039, -1873.4547119, 1812.2724609
3: -661.6334839, 1295.8068848, -695.5081787, 1360.9925537, -2022.6259766, 1991.3146973
4: -444.7619629, 1309.7084961, -468.5613403, 1375.3035889, -1820.0655518, 1778.2697754

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6731900, upper bound: 743.6732604
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762573, upper bound: 743.6770438
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -235.2903442, 1260.9036865, -240.2576447, 1280.4862061, -1515.7763672, 1501.1612549
1: -387.5422058, 1496.9259033, -394.3895264, 1520.1840820, -1907.7263184, 1891.3153076
2: -273.1328430, 1547.3504639, -278.7161865, 1572.2347412, -1845.3675537, 1826.0666504
3: -669.3754883, 1312.6206055, -682.5869751, 1334.0407715, -2003.4160156, 1995.2075195
4: -450.2804260, 1326.0034180, -459.7908325, 1348.6101074, -1798.8905029, 1785.7941895

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782763, upper bound: 743.6779690
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775943, upper bound: 743.6775943
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -235.2903442, 1260.9036865, -244.7313385, 1306.3363037, -1541.6265869, 1505.6350098
1: -387.5422058, 1496.9259033, -401.9236145, 1550.9343262, -1938.4764404, 1898.8491211
2: -273.1328430, 1547.3504639, -283.9859924, 1603.5853271, -1876.7181396, 1831.3364258
3: -669.3754883, 1312.6206055, -695.4236450, 1360.9248047, -2030.2999268, 2008.0441895
4: -450.2804260, 1326.0034180, -468.5013123, 1375.2053223, -1825.4857178, 1794.5045166

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782763, upper bound: 743.6779690
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775943, upper bound: 743.6775943
time: 0.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.97 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6824808, upper bound: 743.6824440
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6824808, upper bound: 743.6824440
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6825021, upper bound: 743.6817793
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6821649, upper bound: 743.6821307
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6263031, upper bound: 743.6349414
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6828174
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6265590, upper bound: 743.6358072
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6826289, upper bound: 743.6826289
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6820234, upper bound: 743.6822862
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6820234, upper bound: 743.6822862
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6820981, upper bound: 743.6817207
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6817716, upper bound: 743.6820718
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6796203, upper bound: 743.6829841
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6798457, upper bound: 743.6818401
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6796203, upper bound: 743.6845942
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6798457, upper bound: 743.6823021
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6154269, upper bound: 743.6215897
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6826082, upper bound: 743.6794428
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6147494, upper bound: 743.6225773
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6814547, upper bound: 743.6795412
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6744186, upper bound: 743.6706707
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6828753, upper bound: 743.6804406
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6718418, upper bound: 743.6705829
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6804869, upper bound: 743.6806209
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6793550, upper bound: 743.6776899
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6560477, upper bound: 743.6555154
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6136354, upper bound: 743.6225526
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6807930, upper bound: 743.6794176
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6792723, upper bound: 743.6808580
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6797419, upper bound: 743.6811368
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6792723, upper bound: 743.6822032
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6797419, upper bound: 743.6821719
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771749
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770173
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6761592, upper bound: 743.6771879
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6763824, upper bound: 743.6770304
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6782841, upper bound: 743.6782573
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6779870, upper bound: 743.6782591
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6782841, upper bound: 743.6782573
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6779870, upper bound: 743.6782591
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6631783, upper bound: 743.6615363
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6734282, upper bound: 743.6714489
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6702104, upper bound: 743.6693669
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6689015, upper bound: 743.6686086
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6770030, upper bound: 743.6770631
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6771456, upper bound: 743.6772115
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6770290, upper bound: 743.6770097
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6771712, upper bound: 743.6771581
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6731900, upper bound: 743.6732604
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6762573, upper bound: 743.6770438
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6731900, upper bound: 743.6732604
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6762573, upper bound: 743.6770438
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6782763, upper bound: 743.6779690
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6775943, upper bound: 743.6775943
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6782763, upper bound: 743.6779690
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.97
Output dim: 0, lower bound: -743.6775943, upper bound: 743.6775943

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -96.9885025, 506.0981140, -105.6504974, 551.9562378, -648.9447021, 611.7485962
1: -158.8952179, 600.4038696, -173.0208282, 655.0476685, -813.9428711, 773.4246826
2: -112.2119446, 623.2306519, -122.2005386, 679.5362549, -791.7481689, 745.4312134
3: -273.8602295, 525.4290771, -298.2593384, 573.4670410, -847.3272705, 823.6882935
4: -184.6551361, 532.9910278, -201.1262665, 581.3231812, -765.9782715, 734.1172485

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6835162, upper bound: 743.6812970
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6826008
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -100.3710098, 523.8244629, -105.6504974, 551.9562378, -652.3272095, 629.4749756
1: -164.2590027, 621.7972412, -173.0208282, 655.0476685, -819.3066406, 794.8180542
2: -116.0657654, 644.9105225, -122.2005386, 679.5362549, -795.6019897, 767.1110840
3: -283.3581238, 544.3154907, -298.2593384, 573.4670410, -856.8250122, 842.5746460
4: -191.0703430, 551.7825928, -201.1262665, 581.3231812, -772.3935547, 752.9088135

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6835162, upper bound: 743.6812970
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6851658, upper bound: 743.6826008
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -89.8007889, 468.9265137, -110.2711182, 576.9017944, -666.7025757, 579.1976318
1: -147.0708618, 556.1876831, -180.4819336, 684.6576538, -831.7285156, 736.6695557
2: -103.8284073, 577.5316162, -127.5492554, 710.0337524, -813.8621826, 705.0808716
3: -253.5540466, 486.3577271, -311.3911743, 599.3527222, -852.9067383, 797.7489014
4: -170.8265991, 493.5342102, -209.9976044, 607.4763794, -778.3029785, 703.5317993

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6646133, upper bound: 743.6634991
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807510, upper bound: 743.6800744
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -95.3904419, 498.1948547, -110.7529831, 579.4051514, -674.7955933, 608.9477539
1: -156.2739868, 591.1506958, -181.2511292, 687.6419067, -843.9158936, 772.4018555
2: -110.3681335, 613.5056763, -128.1015015, 713.0905762, -823.4586182, 741.6071167
3: -269.2639160, 517.2609253, -312.7190857, 601.9974365, -871.2613525, 829.9799805
4: -181.6049347, 524.6630249, -210.9120026, 610.1275024, -791.7324219, 735.5750122

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6648106, upper bound: 743.6641725
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804383, upper bound: 743.6804043
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -110.2634659, 575.6986084, -106.2209091, 554.4555054, -664.7189941, 681.9194946
1: -180.0029907, 683.5120850, -173.8230591, 658.2509155, -838.2539062, 857.3351440
2: -127.2879944, 708.1105347, -122.7999344, 682.3204956, -809.6085205, 830.9104614
3: -310.2442322, 599.2659912, -299.6398010, 576.5797729, -886.8238525, 898.9057617
4: -209.5403748, 607.0921021, -202.1425018, 584.3118286, -793.8519897, 809.2345581

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6846353, upper bound: 743.6825639
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6846353, upper bound: 743.6826978
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.0246124, 569.3483276, -111.5142517, 583.0347290, -692.0593262, 680.8624878
1: -177.9773560, 675.9118042, -182.3666840, 692.0917358, -870.0690918, 858.2784424
2: -125.8491669, 700.3128052, -128.9284668, 717.2835693, -843.1326904, 829.2412720
3: -306.7153931, 592.4528809, -314.6442871, 606.2185059, -912.9338989, 907.0971680
4: -207.1438141, 600.2630615, -212.3188629, 614.1900024, -821.3338013, 812.5819092

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687776, upper bound: 743.6691938
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6808668, upper bound: 743.6808668
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -96.9885025, 506.0981140, -127.7504807, 669.7043457, -766.6927490, 633.8483887
1: -158.8952179, 600.4038696, -207.7281189, 794.8952637, -953.7904053, 808.1319580
2: -112.2119446, 623.2306519, -147.7208862, 824.0305176, -936.2424316, 770.9514771
3: -273.8602295, 525.4290771, -359.6256714, 695.5582275, -969.4184570, 885.0546265
4: -184.6551361, 532.9910278, -243.9409332, 704.3534546, -889.0084839, 776.9319458

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816149, upper bound: 743.6821983
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814869, upper bound: 743.6819679
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -100.3710098, 523.8244629, -127.7504807, 669.7043457, -770.0753174, 651.5747681
1: -164.2590027, 621.7972412, -207.7281189, 794.8952637, -959.1542969, 829.5253296
2: -116.0657654, 644.9105225, -147.7208862, 824.0305176, -940.0962524, 792.6313477
3: -283.3581238, 544.3154907, -359.6256714, 695.5582275, -978.9162598, 903.9410400
4: -191.0703430, 551.7825928, -243.9409332, 704.3534546, -895.4237671, 795.7235107

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816149, upper bound: 743.6821983
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6814869, upper bound: 743.6819679
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -89.8007889, 468.9265137, -132.5339355, 693.0898438, -782.8906250, 601.4604492
1: -147.0708618, 556.1876831, -215.2517090, 822.8289185, -969.8997803, 771.4393311
2: -103.8284073, 577.5316162, -153.1963959, 852.8853760, -956.7138062, 730.7280273
3: -253.5540466, 486.3577271, -372.7752075, 720.4394531, -973.9934692, 859.1328125
4: -170.8265991, 493.5342102, -253.0945129, 729.0106201, -899.8371582, 746.6287231

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6575715, upper bound: 743.6540947
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804021, upper bound: 743.6798215
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -95.3904419, 498.1948547, -132.8732758, 694.9066772, -790.2971191, 631.0680542
1: -156.2739868, 591.1506958, -215.7933502, 824.9838257, -981.2578125, 806.9440308
2: -110.3681335, 613.5056763, -153.5861359, 855.1071777, -965.4751587, 767.0917969
3: -269.2639160, 517.2609253, -373.7065125, 722.3321533, -991.5960693, 890.9674072
4: -181.6049347, 524.6630249, -253.7376404, 730.9199829, -912.5249023, 778.4006348

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775261, upper bound: 743.6789967
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604396, upper bound: 743.6577667
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801218, upper bound: 743.6801421
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -104.7750702, 547.0943604, -126.3043671, 661.6539307, -766.4289551, 673.3987427
1: -171.5263062, 649.2522583, -205.3329620, 785.5828857, -957.1091309, 854.5852051
2: -121.1705399, 673.6063843, -146.0361176, 814.0266113, -935.1971436, 819.6425171
3: -295.6621094, 568.3608398, -355.4576721, 687.7661133, -983.4282227, 923.8184814
4: -199.4253082, 576.2666626, -241.2312317, 696.1493530, -895.5746460, 817.4978638

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6215897, upper bound: 743.6154269
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794428, upper bound: 743.6826082
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -110.6764297, 578.5919189, -125.1267014, 655.2334595, -765.9097290, 703.7184448
1: -181.0428772, 686.6947632, -203.3942261, 777.9520264, -958.9948730, 890.0889893
2: -127.9858551, 712.1240845, -144.6620483, 806.1571655, -934.1430054, 856.7861328
3: -312.3399963, 601.2145996, -352.1151123, 681.0187988, -993.3587036, 953.3297119
4: -210.7366486, 609.3597412, -238.9502411, 689.3666382, -900.1032715, 848.3099976

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6225773, upper bound: 743.6147494
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795412, upper bound: 743.6814547
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -104.7750702, 547.0943604, -131.6514893, 689.5535278, -794.3284912, 678.7458496
1: -171.5263062, 649.2522583, -214.0560455, 818.6406860, -990.1669922, 863.3081665
2: -121.1705399, 673.6063843, -152.2043610, 848.4185181, -969.5890503, 825.8106689
3: -295.6621094, 568.3608398, -370.4444275, 716.7744141, -1012.4365234, 938.8051758
4: -199.4253082, 576.2666626, -251.3567657, 725.5758057, -925.0010986, 827.6233521

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6703751, upper bound: 743.6721453
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803827, upper bound: 743.6828701
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -110.6764297, 578.5919189, -130.7837677, 684.7760010, -795.4523315, 709.3755493
1: -181.0428772, 686.6947632, -212.6269226, 812.9764404, -994.0192871, 899.3216553
2: -127.9858551, 712.1240845, -151.1917114, 842.5675659, -970.5534058, 863.3157959
3: -312.3399963, 601.2145996, -367.9747925, 711.7839355, -1024.1239014, 969.1893921
4: -210.7366486, 609.3597412, -249.6775665, 720.5296631, -931.2662964, 859.0372925

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6600437, upper bound: 743.6598717
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800430, upper bound: 743.6804844
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -121.9425812, 639.0175781, -105.2933578, 550.0979614, -672.0404663, 744.3109131
1: -198.1924744, 758.5976562, -172.4374084, 652.8245850, -851.0170898, 931.0349731
2: -140.9889679, 786.2636108, -121.7876511, 677.2580566, -818.2468872, 908.0512085
3: -343.1169128, 663.7940063, -297.2484436, 571.4950562, -914.6119385, 961.0422974
4: -232.8793030, 672.0563965, -200.4411926, 579.3449097, -812.2242432, 872.4974976

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6760292, upper bound: 743.6729868
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813715, upper bound: 743.6776317
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.7563629, 632.4659424, -110.7529831, 579.4051514, -700.1614990, 743.2188721
1: -196.2399139, 750.8161621, -181.2511292, 687.6419067, -883.8817749, 932.0672607
2: -139.6036530, 778.2445679, -128.1015015, 713.0905762, -852.6942139, 906.3460693
3: -339.7494812, 656.9304199, -312.7190857, 601.9974365, -941.7469482, 969.6495361
4: -230.5777893, 665.1622314, -210.9120026, 610.1275024, -840.7052612, 876.0742188

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6648487, upper bound: 743.6629738
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6800003, upper bound: 743.6781344
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -124.7589188, 653.5538330, -107.0111923, 559.0039062, -683.7627563, 760.5650024
1: -202.6123505, 775.6704102, -175.1728668, 663.3654175, -865.9776611, 950.8432617
2: -144.1412659, 804.3411865, -123.7559662, 688.2156372, -832.3568726, 928.0971680
3: -350.7515869, 678.5160522, -301.9995728, 580.7654419, -931.5170288, 980.5154419
4: -238.0127106, 687.3132935, -203.6910248, 588.7398682, -826.7525635, 891.0043335

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6584203, upper bound: 743.6566203
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6740470, upper bound: 743.6706707
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -128.7117310, 674.2579346, -106.1414413, 554.3510742, -683.0628052, 780.3992920
1: -209.2928772, 800.3945312, -173.7393799, 657.8985596, -867.1914062, 974.1339111
2: -148.7832031, 829.7305908, -122.7412949, 682.4922485, -831.2754517, 952.4718628
3: -362.1069946, 700.4164429, -299.4960632, 575.9573364, -938.0643311, 999.9124756
4: -245.5832214, 709.3179932, -202.0069885, 583.8586426, -829.4417114, 911.3249512

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6714114, upper bound: 743.6683101
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6828753, upper bound: 743.6804406
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -127.7121964, 668.8766479, -111.6200867, 583.7630005, -711.4750977, 780.4966431
1: -207.6510315, 794.0211792, -182.5867004, 692.8096924, -900.4606323, 976.6079102
2: -147.6188660, 823.1315308, -129.0760498, 718.4554443, -866.0743408, 952.2075806
3: -359.2648621, 694.7955322, -315.0136108, 606.5300293, -965.7947998, 1009.8090820
4: -243.6590881, 703.6140137, -212.5090942, 614.7241211, -858.3831787, 916.1231079

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804825, upper bound: 743.6806209
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804825, upper bound: 743.6806209
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -121.5328522, 636.8981323, -127.7504807, 669.7043457, -791.2371826, 764.6483765
1: -197.4739227, 756.0499878, -207.7281189, 794.8952637, -992.3692017, 963.7780762
2: -140.5295715, 783.6213379, -147.7208862, 824.0305176, -964.5600586, 931.3422241
3: -342.0090637, 661.5822144, -359.6256714, 695.5582275, -1037.5672607, 1021.2077637
4: -232.1906433, 669.8446655, -243.9409332, 704.3534546, -936.5440674, 913.7855835

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6560477, upper bound: 743.6555154
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6560477, upper bound: 743.6555154
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.7563629, 632.4659424, -132.8732758, 694.9066772, -815.6630249, 765.3391724
1: -196.2399139, 750.8161621, -215.7933502, 824.9838257, -1021.2236938, 966.6094971
2: -139.6036530, 778.2445679, -153.5861359, 855.1071777, -994.7108154, 931.8306885
3: -339.7494812, 656.9304199, -373.7065125, 722.3321533, -1062.0816650, 1030.6369629
4: -230.5777893, 665.1622314, -253.7376404, 730.9199829, -961.4978027, 918.8997803

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6609164, upper bound: 743.6581422
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793114, upper bound: 743.6778971
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -127.5267563, 668.1494751, -126.3043671, 661.6539307, -789.1806641, 794.4536743
1: -207.2814026, 793.0996094, -205.3329620, 785.5828857, -992.8642578, 998.4325562
2: -147.4360657, 822.1558228, -146.0361176, 814.0266113, -961.4626465, 968.1919556
3: -358.8092651, 694.0598755, -355.4576721, 687.7661133, -1046.5754395, 1049.5173340
4: -243.4863281, 702.8020020, -241.2312317, 696.1493530, -939.6355591, 944.0332031

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6722388, upper bound: 743.6705023
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777269, upper bound: 743.6793550
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6556659, upper bound: 743.6561982
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -133.8376312, 699.4594727, -125.1267014, 655.2334595, -789.0710449, 824.5860596
1: -217.2881012, 830.4815674, -203.3942261, 777.9520264, -995.2401123, 1033.8757324
2: -154.6733246, 860.7138062, -144.6620483, 806.1571655, -960.8303223, 1005.3758545
3: -376.2634277, 727.3377686, -352.1151123, 681.0187988, -1057.2822266, 1079.4527588
4: -255.5643616, 735.8706665, -238.9502411, 689.3666382, -944.9310303, 974.8209229

Time for backsubstitution: 2.10 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=860.0533447265625
rel_dist={0: [-743.6893068054288, 743.6893068054287]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821875, upper bound: 743.6826011
time: 1.06 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805918, upper bound: 743.6805918
time: 0.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.21 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.21
Output dim: 0, lower bound: -743.6821875, upper bound: 743.6826011
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.21
Output dim: 0, lower bound: -743.6805918, upper bound: 743.6805918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -128.5790863, 670.3025513, -135.3650513, 705.3255615, -833.9046021, 805.6676025
1: -210.4309387, 796.4016113, -221.3665009, 837.9066772, -1048.3374023, 1017.7680664
2: -148.6245575, 824.4069214, -156.5181122, 867.5812988, -1016.2058105, 980.9249878
3: -362.4870300, 699.0101929, -381.4162598, 735.6503296, -1098.1373291, 1080.4263916
4: -244.7495728, 707.2666626, -257.7686768, 744.3495483, -989.0991211, 965.0352173

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807758, upper bound: 743.6795852
time: 0.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807638, upper bound: 743.6806666
time: 0.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -250.1162415, 1337.2148438, -132.5029755, 693.2515869, -943.3677979, 1469.7177734
1: -412.2479858, 1588.0107422, -216.6434479, 823.1146851, -1235.3626709, 1804.6540527
2: -290.3746338, 1641.2213135, -153.3491211, 852.6534424, -1143.0279541, 1794.5703125
3: -711.9147949, 1393.7313232, -373.6543274, 721.8125610, -1433.7272949, 1767.3856201
4: -478.8499146, 1407.6657715, -252.4815521, 731.0437012, -1209.8935547, 1660.1473389

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794369, upper bound: 743.6784483
time: 0.67 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796078, upper bound: 743.6796078
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -743.6807758, upper bound: 743.6795852
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -743.6807638, upper bound: 743.6806666
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -743.6794369, upper bound: 743.6784483
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 0, lower bound: -743.6796078, upper bound: 743.6796078

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -121.5711136, 634.5589600, -118.9349518, 621.3196411, -742.8907471, 753.4938354
1: -198.9713287, 753.7140503, -194.5095673, 737.5877686, -936.5590210, 948.2236328
2: -140.5469971, 780.6080322, -137.5653839, 764.6723022, -905.2192993, 918.1734009
3: -342.8704529, 661.0322876, -335.4618835, 646.4867554, -989.3571777, 996.4941406
4: -231.4364166, 669.1576538, -226.5262909, 654.9082642, -886.3446655, 895.6838989

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804674, upper bound: 743.6795704
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6804674, upper bound: 743.6795852
time: 0.75 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -114.2929382, 597.1482544, -142.1371918, 743.8861694, -858.1790771, 739.2854004
1: -186.6672211, 709.1284790, -231.1433105, 883.0995483, -1069.7667236, 940.2716675
2: -132.0549011, 734.5758057, -164.3655548, 915.2902832, -1047.3449707, 898.9412231
3: -321.7898254, 621.4840698, -399.9130554, 773.8182983, -1095.6081543, 1021.3970947
4: -217.4337311, 629.2165527, -271.3944397, 783.1953125, -1000.6290283, 900.6109619

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6651747, upper bound: 743.6646951
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
time: 0.83 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -243.2717438, 1301.8137207, -115.6522903, 606.6144409, -849.8861694, 1417.4659424
1: -401.0203552, 1545.7716064, -189.0507507, 719.6542358, -1120.6745605, 1734.8223877
2: -282.4453735, 1597.7733154, -133.8859100, 746.5598145, -1029.0051270, 1731.6591797
3: -692.6942139, 1356.2403564, -326.4170227, 629.9951782, -1322.6894531, 1682.6573486
4: -465.7908630, 1370.0834961, -220.3940735, 638.9992676, -1104.7901611, 1590.4772949

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6713014, upper bound: 743.6688281
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6591822, upper bound: 743.6598469
time: 0.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -236.2507477, 1264.7819824, -136.9795380, 719.7861328, -956.0368652, 1401.7614746
1: -389.0445251, 1501.6123047, -222.8949127, 854.0061646, -1243.0505371, 1724.5072021
2: -274.2849426, 1552.2906494, -158.6326447, 885.6903687, -1159.9753418, 1710.9230957
3: -672.3327026, 1317.3229980, -385.8038330, 747.5045166, -1419.8371582, 1703.1268311
4: -452.3524780, 1330.8757324, -261.8527222, 757.4526978, -1209.8051758, 1592.7283936

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786659, upper bound: 743.6789254
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786643, upper bound: 743.6786643
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.51 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6804674, upper bound: 743.6795704
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6804674, upper bound: 743.6795852
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6651747, upper bound: 743.6646951
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6713014, upper bound: 743.6688281
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6591822, upper bound: 743.6598469
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6786659, upper bound: 743.6789254
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 0, lower bound: -743.6786643, upper bound: 743.6786643

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -118.9349518, 621.3196411, -734.2932739, 709.7319336
1: -184.9509888, 701.4193115, -194.5095673, 737.5877686, -922.5386963, 895.9288940
2: -130.6758118, 727.0247803, -137.5653839, 764.6723022, -895.3480835, 864.5900879
3: -318.8685303, 614.4975586, -335.4618835, 646.4867554, -965.3552856, 949.9594727
4: -215.1691284, 622.4899902, -226.5262909, 654.9082642, -870.0773315, 849.0161743

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795704
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795704
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -134.4156647, 704.3609619, -118.9349518, 621.3196411, -755.7351685, 823.2958984
1: -218.5928497, 836.1904907, -194.5095673, 737.5877686, -956.1806030, 1030.7000732
2: -155.4117889, 866.5617676, -137.5653839, 764.6723022, -920.0841064, 1004.1270142
3: -378.2995605, 732.1653442, -335.4618835, 646.4867554, -1024.7862549, 1067.6270752
4: -256.6361694, 741.0963745, -226.5262909, 654.9082642, -911.5443115, 967.6226196

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795852
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795852
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -109.8864517, 575.3594971, -138.7057495, 724.7770996, -834.6633911, 714.0651855
1: -179.2736969, 682.9730225, -225.4969025, 860.6778564, -1039.9515381, 908.4699097
2: -126.8475266, 707.5144043, -160.3337097, 891.7111206, -1018.5586548, 867.8480835
3: -308.9889221, 598.3204956, -390.0628967, 754.7375488, -1063.7264404, 988.3834229
4: -208.7905121, 606.0291138, -264.7478333, 763.6119385, -972.4024658, 870.7769775

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -228.4421387, 1222.7379150, -134.1076050, 704.6690674, -933.1111450, 1356.8454590
1: -376.2122192, 1451.4896240, -218.1237793, 835.9437256, -1212.1558838, 1669.6130371
2: -265.1391296, 1501.0238037, -155.2670288, 867.1760254, -1132.3151855, 1656.2907715
3: -650.0893555, 1272.7851562, -377.5998230, 731.4227905, -1381.5122070, 1650.3850098
4: -437.1325989, 1286.5235596, -256.2620544, 741.4137573, -1178.5463867, 1542.7856445

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6677281, upper bound: 743.6665989
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6629584, upper bound: 743.6629114
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -232.5255280, 1246.4826660, -133.7921600, 703.0712891, -935.5968018, 1380.2747803
1: -382.9930420, 1479.6049805, -217.7366028, 834.1026001, -1217.0957031, 1697.3415527
2: -269.9559021, 1529.7435303, -154.9269714, 865.2407837, -1135.1966553, 1684.6704102
3: -661.8043213, 1297.3220215, -376.8312378, 729.8064575, -1391.6105957, 1674.1533203
4: -445.1285400, 1310.9323730, -255.6695404, 739.7870483, -1184.9155273, 1566.6019287

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695861, upper bound: 743.6684459
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6659764, upper bound: 743.6659764
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795704
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795704
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795852
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6806856, upper bound: 743.6795852
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6802469, upper bound: 743.6785506
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6677281, upper bound: 743.6665989
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6629584, upper bound: 743.6629114
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6695861, upper bound: 743.6684459
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.70
Output dim: 0, lower bound: -743.6659764, upper bound: 743.6659764

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -112.9736404, 590.7969971, -703.7706299, 703.7706299
1: -184.9509888, 701.4193115, -184.9509888, 701.4193115, -886.3702393, 886.3703003
2: -130.6758118, 727.0247803, -130.6758118, 727.0247803, -857.7005615, 857.7005615
3: -318.8685303, 614.4975586, -318.8685303, 614.4975586, -933.3660278, 933.3660278
4: -215.1691284, 622.4899902, -215.1691284, 622.4899902, -837.6589966, 837.6589966

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5684304, upper bound: 743.5603701
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5691235, upper bound: 743.5605995
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.9736404, 590.7969971, -232.8589478, 1248.5032959, -1361.4769287, 823.6558838
1: -184.9509888, 701.4193115, -384.0163879, 1482.1339111, -1667.0849609, 1085.4355469
2: -130.6758118, 727.0247803, -270.4130554, 1532.3221436, -1662.9979248, 997.4378662
3: -318.8685303, 614.4975586, -663.5385132, 1299.6317139, -1618.5002441, 1278.0360107
4: -215.1691284, 622.4899902, -445.9592896, 1313.3482666, -1528.5173340, 1068.4489746

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5684304, upper bound: 743.5603701
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5691235, upper bound: 743.5605995
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -134.4156647, 704.3609619, -112.9736404, 590.7969971, -725.2126465, 817.3345947
1: -218.5928497, 836.1904907, -184.9509888, 701.4193115, -920.0121460, 1021.1414795
2: -155.4117889, 866.5617676, -130.6758118, 727.0247803, -882.4365845, 997.2374878
3: -378.2995605, 732.1653442, -318.8685303, 614.4975586, -992.7971191, 1051.0339355
4: -256.6361694, 741.0963745, -215.1691284, 622.4899902, -879.1259766, 956.2654419

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5595914
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5609544
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -134.4156647, 704.3609619, -232.8150787, 1248.2844238, -1382.7000732, 937.1760254
1: -218.5928497, 836.1904907, -383.9443359, 1481.8747559, -1700.4676514, 1220.1347656
2: -155.4117889, 866.5617676, -270.3623352, 1532.0515137, -1687.4632568, 1136.9240723
3: -378.2995605, 732.1653442, -663.4143677, 1299.4016113, -1677.7011719, 1395.5797119
4: -256.6361694, 741.0963745, -445.8751831, 1313.1137695, -1569.4902344, 1186.9714355

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5595914
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5609544
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.8864517, 575.3594971, -132.8022919, 694.2027588, -804.0891724, 708.1616821
1: -179.2736969, 682.9730225, -215.9902039, 824.4985352, -1003.7722168, 898.9632568
2: -126.8475266, 707.5144043, -153.5099640, 854.0543213, -980.9018555, 861.0243530
3: -308.9889221, 598.3204956, -373.7120972, 722.8700562, -1031.8590088, 972.0325928
4: -208.7905121, 606.0291138, -253.5276947, 731.3139648, -940.1044922, 859.5567017

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6785506
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6784548
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.8864517, 575.3594971, -245.2544556, 1307.7990723, -1417.6855469, 820.6139526
1: -179.2736969, 682.9730225, -402.6631775, 1553.0058594, -1732.2795410, 1085.6362305
2: -126.8475266, 707.5144043, -284.5502625, 1605.3333740, -1732.1806641, 992.0646973
3: -308.9889221, 598.3204956, -696.7287598, 1363.6282959, -1672.6171875, 1295.0491943
4: -208.7905121, 606.0291138, -469.5535583, 1377.5902100, -1586.3806152, 1075.5826416

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6785506
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6784548
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.65 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5684304, upper bound: 743.5603701
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5691235, upper bound: 743.5605995
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5684304, upper bound: 743.5603701
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5691235, upper bound: 743.5605995
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5595914
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5609544
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5595914
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.5676264, upper bound: 743.5609544
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6785506
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6784548
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6785506
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 0, lower bound: -743.6798948, upper bound: 743.6784548

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.5535049, 587.8195190, -132.8022919, 694.2027588, -806.7562866, 720.6217041
1: -183.7729950, 697.9183960, -215.9902039, 824.4985352, -1008.2715454, 913.9085693
2: -129.9520111, 722.9826050, -153.5099640, 854.0543213, -984.0063477, 876.4925537
3: -316.7823486, 611.9615479, -373.7120972, 722.8700562, -1039.6523438, 985.6736450
4: -213.9270020, 619.8825684, -253.5276947, 731.3139648, -945.2409668, 873.4101562

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6809000, upper bound: 743.6813099
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6836904, upper bound: 743.6836904
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -126.3022079, 661.6420898, -132.8022919, 694.2027588, -820.5049438, 794.4442139
1: -205.2046204, 785.4786377, -215.9902039, 824.4985352, -1029.7030029, 1001.4688721
2: -145.9328613, 813.9240112, -153.5099640, 854.0543213, -999.9871826, 967.4339600
3: -355.0084839, 687.8810425, -373.7120972, 722.8700562, -1077.8785400, 1061.5928955
4: -240.9271393, 696.1613770, -253.5276947, 731.3139648, -972.2410889, 949.6890259

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6809000, upper bound: 743.6813099
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6836904, upper bound: 743.6836904
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.5535049, 587.8195190, -245.2352600, 1307.7025146, -1420.2559814, 833.0548096
1: -183.7729950, 697.9183960, -402.6316833, 1552.8912354, -1736.6641846, 1100.5500488
2: -129.9520111, 722.9826050, -284.5281982, 1605.2145996, -1735.1665039, 1007.5108032
3: -316.7823486, 611.9615479, -696.6744995, 1363.5261230, -1680.3084717, 1308.6359863
4: -213.9270020, 619.8825684, -469.5166931, 1377.4870605, -1591.4139404, 1089.3992920

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791099, upper bound: 743.6781254
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6621714, upper bound: 743.6630867
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6513281, upper bound: 743.6510984
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -126.3022079, 661.6420898, -245.1838531, 1307.4442139, -1433.7462158, 906.8259277
1: -205.2046204, 785.4786377, -402.5472412, 1552.5843506, -1757.7888184, 1188.0258789
2: -145.9328613, 813.9240112, -284.4688416, 1604.8957520, -1750.8286133, 1098.3928223
3: -355.0084839, 687.8810425, -696.5291748, 1363.2530518, -1718.2614746, 1384.4099121
4: -240.9271393, 696.1613770, -469.4180298, 1377.2111816, -1618.1378174, 1165.5793457

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6791099, upper bound: 743.6780999
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6621714, upper bound: 743.6626742
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6513281, upper bound: 743.6510984
time: 1.11 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.54 seconds
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6809000, upper bound: 743.6813099
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6836904, upper bound: 743.6836904
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6809000, upper bound: 743.6813099
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6836904, upper bound: 743.6836904
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6621714, upper bound: 743.6630867
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6513281, upper bound: 743.6510984
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6621714, upper bound: 743.6626742
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.54
Output dim: 0, lower bound: -743.6513281, upper bound: 743.6510984

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -108.2882156, 565.4182129, -124.2383499, 649.1452026, -757.4332886, 689.6565552
1: -176.9326172, 671.3320923, -201.9607849, 771.0812378, -948.0137939, 873.2928467
2: -125.0448303, 695.4693604, -143.6025696, 798.5954590, -923.6402588, 839.0718384
3: -304.9407654, 588.6151733, -349.5443420, 675.9639282, -980.9046631, 938.1595459
4: -205.8247681, 596.3399658, -237.2444153, 683.8510742, -889.6758423, 833.5843506

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784040, upper bound: 743.6803391
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794659, upper bound: 743.6809159
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -111.7841110, 583.7898560, -130.2130737, 680.3433838, -792.1275024, 714.0029297
1: -182.5046539, 693.1295776, -211.7438965, 808.0780029, -990.5826416, 904.8734131
2: -129.0562897, 718.0343628, -150.5024414, 837.0573730, -966.1136475, 868.5367432
3: -314.5763550, 607.7368774, -366.3440857, 708.4655151, -1023.0418701, 974.0809326
4: -212.4529572, 615.6199341, -248.5749817, 716.7762451, -929.2291870, 864.1948853

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6822256, upper bound: 743.6834220
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6820642, upper bound: 743.6821194
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -122.7702408, 643.4505005, -124.2383499, 649.1452026, -771.9154053, 767.6888428
1: -199.5637817, 763.8261108, -201.9607849, 771.0812378, -970.6450195, 965.7868042
2: -141.8893433, 791.5953369, -143.6025696, 798.5954590, -940.4848022, 935.1978760
3: -345.2917175, 668.7427979, -349.5443420, 675.9639282, -1021.2556152, 1018.2871094
4: -234.2440338, 676.9312744, -237.2444153, 683.8510742, -918.0950928, 914.1756592

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6233642, upper bound: 743.6312839
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799171, upper bound: 743.6806856
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -125.4033661, 656.8510132, -130.2130737, 680.3433838, -805.7467041, 787.0640869
1: -203.7227478, 779.8095703, -211.7438965, 808.0780029, -1011.8007202, 991.5534058
2: -144.8871460, 808.0402222, -150.5024414, 837.0573730, -981.9445190, 958.5426636
3: -352.4350586, 682.9262085, -366.3440857, 708.4655151, -1060.9001465, 1049.2702637
4: -239.2116089, 691.1299438, -248.5749817, 716.7762451, -955.9878540, 939.7048950

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6810574, upper bound: 743.6812525
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826522, upper bound: 743.6826522
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.57 seconds
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6784040, upper bound: 743.6803391
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6794659, upper bound: 743.6809159
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6822256, upper bound: 743.6834220
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6820642, upper bound: 743.6821194
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6233642, upper bound: 743.6312839
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6799171, upper bound: 743.6806856
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6810574, upper bound: 743.6812525
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.57
Output dim: 0, lower bound: -743.6826522, upper bound: 743.6826522

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -103.6830978, 540.2709961, -122.2435837, 638.8383179, -742.5214233, 662.5145874
1: -169.3316956, 641.4143677, -198.6887360, 758.7716064, -928.1032715, 840.1030884
2: -119.6813126, 664.6749878, -141.2972565, 785.9461670, -905.6275024, 805.9722290
3: -291.8728027, 562.2522583, -343.9260864, 664.9910278, -956.8638306, 906.1781616
4: -196.9387360, 569.8983765, -233.4347534, 672.8603516, -869.7990723, 803.3331299

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 25

Time for candidate selection: 11.24 seconds

### Candidate
type: B, layer: 3, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6783779, upper bound: 743.6801618
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -106.5561676, 556.8811035, -121.4102936, 633.9031982, -740.4593506, 678.2913818
1: -173.9851379, 660.9297485, -197.3059692, 752.9697876, -926.9549561, 858.2357178
2: -123.0389175, 685.0297852, -140.3136292, 779.9202271, -902.9591675, 825.3433838
3: -300.0370483, 578.8858032, -341.5194397, 659.9509277, -959.9877319, 920.4052734
4: -202.4703369, 586.7277832, -231.8021240, 667.7235107, -870.1938477, 818.5299072

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6010791, upper bound: 743.5961577
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6790333, upper bound: 743.6804950
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.2263794, 558.9567261, -128.1019592, 669.3328857, -776.5592041, 687.0586548
1: -175.0039825, 663.5709229, -208.2766266, 794.9259644, -969.9299316, 871.8475342
2: -123.7531738, 687.6325073, -148.0582886, 823.5532227, -947.3063965, 835.6907959
3: -301.6783447, 581.6857910, -360.3928223, 696.7415771, -998.4198608, 942.0784912
4: -203.6665649, 589.5092163, -244.5283966, 705.0620117, -908.7285767, 834.0375366

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6703467, upper bound: 743.6725397
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802121, upper bound: 743.6814752
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -111.3873215, 581.8513794, -127.9668503, 668.1965332, -779.5837402, 709.8181152
1: -181.6612244, 690.6502686, -208.0534668, 793.6489868, -975.3099976, 898.7036743
2: -128.5666656, 715.6653442, -147.8880157, 822.1629639, -950.7296143, 863.5533447
3: -313.3125305, 605.1284790, -359.9759216, 695.7496338, -1009.0620728, 965.1043701
4: -211.6306458, 612.9989014, -244.2412109, 703.9054565, -915.5360718, 857.2400513

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6698336, upper bound: 743.6703455
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801795, upper bound: 743.6800315
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -120.6823959, 632.6312866, -121.1301956, 632.4087524, -753.0910645, 753.7612915
1: -196.1441956, 750.9697876, -196.8929901, 751.1452026, -947.2893677, 947.8627930
2: -139.4269867, 778.2990112, -139.9789581, 778.1623535, -917.5893555, 918.2779541
3: -339.2666931, 657.4843140, -340.7604065, 658.2886353, -997.5552979, 998.2446289
4: -230.0656891, 665.4216919, -231.1894989, 666.2713623, -896.3369751, 896.6112061

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5946673, upper bound: 743.5894151
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795977, upper bound: 743.6804059
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -118.8659668, 622.3664551, -127.2204514, 664.7358398, -783.6018066, 749.5869141
1: -192.8065491, 738.6988525, -206.7638550, 789.4254761, -982.2320557, 945.4627075
2: -137.2072144, 765.7896118, -146.9990387, 817.9313965, -955.1383667, 912.7886353
3: -333.6643982, 646.3602295, -357.7800903, 691.8069458, -1025.4711914, 1004.1403198
4: -226.5031891, 654.5895996, -242.7727966, 700.1538696, -926.6569824, 897.3623047

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6711768, upper bound: 743.6703769
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709178, upper bound: 743.6701985
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -122.9754715, 644.4849854, -127.3756104, 665.1851807, -788.1605225, 771.8605957
1: -199.7925568, 765.0316162, -207.1579285, 790.0305786, -989.8231201, 972.1895752
2: -142.0426331, 792.8778076, -147.2010498, 818.5266113, -960.5692139, 940.0788574
3: -345.5006714, 669.8426514, -358.3482666, 692.5999756, -1038.1005859, 1028.1907959
4: -234.3589478, 677.8869019, -243.0368347, 700.8199463, -935.1788940, 920.9237061

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801578, upper bound: 743.6800521
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798822, upper bound: 743.6798934
time: 0.85 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.92 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6783779, upper bound: 743.6801618
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6010791, upper bound: 743.5961577
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6790333, upper bound: 743.6804950
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6703467, upper bound: 743.6725397
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6802121, upper bound: 743.6814752
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6698336, upper bound: 743.6703455
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6801795, upper bound: 743.6800315
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.5946673, upper bound: 743.5894151
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6795977, upper bound: 743.6804059
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6711768, upper bound: 743.6703769
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6709178, upper bound: 743.6701985
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6801578, upper bound: 743.6800521
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -743.6798822, upper bound: 743.6798934

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -103.6830978, 540.2709961, -106.9548340, 547.9552002, -651.6383057, 647.2258301
1: -169.3316956, 641.4143677, -173.4595032, 653.3713379, -822.7030029, 814.8739014
2: -119.6813126, 664.6749878, -123.4143372, 673.6066284, -793.2879639, 788.0893555
3: -291.8728027, 562.2522583, -299.8485413, 578.9887085, -870.8615112, 862.1006470
4: -196.9387360, 569.8983765, -204.3113556, 583.3242188, -780.2629395, 774.2097168

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779913, upper bound: 743.6798924
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782448, upper bound: 743.6801548
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -103.6830978, 540.2709961, -117.3608398, 616.0281372, -719.7112427, 657.6318359
1: -169.3316956, 641.4143677, -190.8022308, 731.2055054, -900.5372314, 832.2165527
2: -119.6813126, 664.6749878, -135.7741699, 757.7407227, -877.4220581, 800.4491577
3: -291.8728027, 562.2522583, -330.4240112, 640.0795288, -931.9523315, 892.6762085
4: -196.9387360, 569.8983765, -224.5089874, 647.8059082, -844.7446289, 794.4073486

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6729744, upper bound: 743.6767760
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 25

Time for candidate selection: 11.54 seconds

### Candidate
type: A, layer: 3, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -104.2929535, 545.1140747, -117.0230560, 611.0989990, -715.3919067, 662.1371460
1: -170.2792511, 646.8815308, -190.1148224, 725.7759399, -896.0550537, 836.9963379
2: -120.4201431, 670.6151733, -135.2342377, 751.9508667, -872.3710327, 805.8494263
3: -293.6359558, 566.4163818, -329.1007080, 635.7837524, -929.4196777, 895.5169678
4: -198.1301727, 574.2164917, -223.3852692, 643.4497681, -841.5799561, 797.6017456

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6404249, upper bound: 743.6473984
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776514, upper bound: 743.6791857
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -104.1523285, 543.0599976, -125.1951065, 654.2939453, -758.4461670, 668.2551270
1: -169.9497070, 644.6269531, -203.5682983, 776.9693604, -946.9190674, 848.1952515
2: -120.1644821, 668.0747681, -144.6759491, 805.1530762, -925.3175659, 812.7507324
3: -292.8976746, 564.8279419, -352.1497498, 680.7689819, -973.6666260, 916.9776611
4: -197.7020416, 572.5327759, -238.8057098, 689.0353394, -886.7373047, 811.3383789

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801566, upper bound: 743.6812262
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6796012, upper bound: 743.6807996
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -108.6439896, 567.7659912, -125.2702408, 654.4338989, -763.0778198, 693.0362549
1: -177.2133484, 673.8403931, -203.7144470, 777.2029419, -954.4162598, 877.5547485
2: -125.3870010, 698.3715210, -144.7550201, 805.3081665, -930.6950684, 843.1265259
3: -305.6136780, 590.1860962, -352.3522034, 681.1976929, -986.8113403, 942.5382690
4: -206.3541412, 598.0546875, -238.9298248, 689.1994019, -895.5535278, 836.9844971

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801547, upper bound: 743.6796240
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798696, upper bound: 743.6797926
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -118.7333069, 622.4210815, -116.7610397, 609.6529541, -728.3861084, 739.1820679
1: -192.9465485, 738.7943115, -189.7266693, 724.0159912, -916.9625244, 928.5209961
2: -137.1725006, 765.7911377, -134.9234314, 750.2645264, -887.4370117, 900.7144775
3: -333.7458496, 646.6643677, -328.3816223, 634.1965942, -967.9424438, 975.0460205
4: -226.3259735, 654.5969849, -222.8151703, 642.0723877, -868.3983765, 877.4121094

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788688, upper bound: 743.6796170
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795860, upper bound: 743.6803352
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -120.8226089, 633.0490723, -122.7987061, 641.2289429, -762.0514526, 755.8477173
1: -196.2329712, 751.4011230, -199.6256561, 761.4134521, -957.6464233, 951.0267944
2: -139.5439606, 778.8626709, -141.8968353, 789.1612549, -928.7052002, 920.7595215
3: -339.3977966, 657.6563721, -345.4416504, 666.9645386, -1006.3623047, 1003.0979004
4: -230.2260742, 665.7871094, -234.2608337, 675.3733521, -905.5994263, 900.0479126

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770804, upper bound: 743.6762797
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6764437, upper bound: 743.6756263
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701828, upper bound: 743.6686386
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -120.5224762, 631.4532471, -129.1353455, 672.9604492, -793.4829102, 760.5885620
1: -195.7806091, 749.5223999, -209.7020721, 799.3232422, -995.1038818, 959.2244263
2: -139.1978302, 776.8997803, -149.1842957, 828.2308350, -967.4284058, 926.0840454
3: -338.5714111, 656.1098633, -362.9882812, 700.8439941, -1039.4151611, 1019.0981445
4: -229.6475525, 664.0378418, -246.4568787, 708.9054565, -938.5529785, 910.4946899

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6706812, upper bound: 743.6715480
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6654559, upper bound: 743.6655035
time: 0.70 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.96 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6779913, upper bound: 743.6798924
IS_A1_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6782448, upper bound: 743.6801548
IS_A1_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
IS_A1_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6763931, upper bound: 743.6799394
IS_A1_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6404249, upper bound: 743.6473984
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6776514, upper bound: 743.6791857
IS_A1_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6801566, upper bound: 743.6812262
IS_A1_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6796012, upper bound: 743.6807996
IS_A1_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6801547, upper bound: 743.6796240
IS_A1_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6798696, upper bound: 743.6797926
IS_A1_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6788688, upper bound: 743.6796170
IS_A1_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6795860, upper bound: 743.6803352
IS_A1_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6764437, upper bound: 743.6756263
IS_A1_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6701828, upper bound: 743.6686386
IS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6706812, upper bound: 743.6715480
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.96
Output dim: 0, lower bound: -743.6654559, upper bound: 743.6655035

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -98.8066635, 514.7426758, -104.5144501, 534.9624634, -633.7691040, 619.2571411
1: -161.3678894, 610.9003296, -169.4770966, 637.8856201, -799.2534180, 780.3774414
2: -114.0396881, 633.3759155, -120.5946274, 657.7006226, -771.7402954, 753.9705200
3: -278.1366577, 535.1267700, -292.9967346, 565.2755127, -843.4120483, 828.1235352
4: -187.5753937, 542.8269653, -199.6466827, 569.6121826, -757.1875610, 742.4736328

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6778569, upper bound: 743.6795999
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 25

Time for candidate selection: 12.26 seconds

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743380, upper bound: 743.6777684
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775307, upper bound: 743.6796402
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -101.1788025, 527.5548706, -104.8373260, 536.7848511, -637.9636230, 632.3922119
1: -165.2015381, 626.0507202, -170.0108948, 640.0503540, -805.2518921, 796.0616455
2: -116.8274765, 649.3641357, -120.9664307, 659.9274902, -776.7549438, 770.3305664
3: -284.7787476, 548.6115723, -293.9079590, 567.1568604, -851.9356079, 842.5194092
4: -192.2169342, 556.3201294, -200.2637329, 571.4836426, -763.7005615, 756.5838623

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780482, upper bound: 743.6798536
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 25

Time for candidate selection: 12.29 seconds

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743946, upper bound: 743.6780436
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777709, upper bound: 743.6799094
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -96.7643280, 499.7228699, -117.3608398, 616.0281372, -712.7924805, 617.0837402
1: -157.7897949, 594.0247803, -190.8022308, 731.2055054, -888.9953003, 784.8270264
2: -111.5709686, 614.9988403, -135.7741699, 757.7407227, -869.3117065, 750.7730103
3: -271.9071960, 522.5017700, -330.4240112, 640.0795288, -911.9866943, 852.9257812
4: -183.8138123, 528.9987183, -224.5089874, 647.8059082, -831.6197510, 753.5076904

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762881, upper bound: 743.6796699
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744914, upper bound: 743.6765870
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -100.7389984, 526.4280396, -117.3608398, 616.0281372, -716.7671509, 643.7888794
1: -164.6297455, 624.7213135, -190.8022308, 731.2055054, -895.8352661, 815.5234375
2: -116.3866272, 647.6108398, -135.7741699, 757.7407227, -874.1273193, 783.3850098
3: -283.8333130, 547.3147583, -330.4240112, 640.0795288, -923.9128418, 877.7387695
4: -191.6205292, 554.8408203, -224.5089874, 647.8059082, -839.4264526, 779.3497925

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762881, upper bound: 743.6796699
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744914, upper bound: 743.6765870
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -102.1643906, 535.6278687, -114.0562286, 595.1069946, -697.2713623, 649.6840210
1: -166.6730347, 635.5159302, -185.2834320, 706.7360840, -873.4091187, 820.7993164
2: -117.9259109, 658.6922607, -131.7800903, 732.4379883, -850.3638916, 790.4723511
3: -287.3461914, 555.9965820, -320.7155457, 618.9240723, -906.2701416, 876.7121582
4: -193.9746857, 563.4969482, -217.6159058, 626.6870728, -820.6617432, 781.1127930

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6702206, upper bound: 743.6725996
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770647, upper bound: 743.6783773
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776514, upper bound: 743.6791628
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -94.1639023, 489.9299011, -123.5144424, 645.4087524, -739.5725708, 613.4443359
1: -153.4744110, 581.4916382, -200.8594360, 766.4022827, -919.8767090, 782.3510132
2: -108.5047455, 602.8825073, -142.7365875, 794.2837524, -902.7884521, 745.6190796
3: -264.6234436, 509.2904968, -347.4766846, 671.4004517, -936.0238647, 856.7672119
4: -178.4528809, 516.5706787, -235.5940704, 679.6434326, -858.0963135, 752.1646118

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763171, upper bound: 743.6764995
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801566, upper bound: 743.6812262
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -99.0399551, 516.3248901, -123.2677612, 644.2430420, -743.2830200, 639.5926514
1: -161.5665436, 612.7414551, -200.4116516, 764.9798584, -926.5463867, 813.1530151
2: -114.2459335, 635.3184204, -142.4484863, 792.8397217, -907.0856323, 777.7668457
3: -278.4212646, 536.5488281, -346.7039795, 670.1077881, -948.5289307, 883.2528076
4: -187.8923950, 544.1619263, -235.1145325, 678.3708496, -866.2632446, 779.2764893

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751712, upper bound: 743.6761208
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6682327, upper bound: 743.6698932
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -98.5851135, 514.1384277, -123.5712051, 645.4635620, -744.0487061, 637.7096558
1: -160.5891266, 610.0873413, -200.9665527, 766.5343018, -927.1234131, 811.0538940
2: -113.6319580, 632.5002441, -142.7928467, 794.3296509, -907.9616089, 775.2930298
3: -277.0696716, 534.1500244, -347.6203613, 671.7296143, -948.7993164, 881.7703857
4: -186.9441071, 541.5297852, -235.6814270, 679.7059326, -866.6500244, 777.2111816

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771361, upper bound: 743.6766804
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6801547, upper bound: 743.6796240
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -103.4380341, 540.5734253, -123.3536072, 644.3963623, -747.8344116, 663.9270020
1: -168.7017365, 641.3815308, -200.5740814, 765.2330933, -933.9348145, 841.9556274
2: -119.3647461, 665.0621338, -142.5390930, 793.0165405, -912.3812866, 807.6011963
3: -290.9015808, 561.4066772, -346.9338379, 670.5621338, -961.4637451, 908.3402100
4: -196.3690948, 569.1924438, -235.2581787, 678.5636597, -874.9327393, 804.4505615

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6682781, upper bound: 743.6676539
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6636253, upper bound: 743.6633431
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -113.8418579, 596.4632568, -114.4976959, 597.7648926, -711.6067505, 710.9608765
1: -184.9423981, 707.8392944, -186.0356903, 709.8308105, -894.7731934, 893.8750000
2: -131.5144958, 734.0165405, -132.3091431, 735.7009888, -867.2154541, 866.3255615
3: -319.9654846, 619.0969849, -322.0304871, 621.5991821, -941.5645752, 941.1274414
4: -216.9555969, 627.1946411, -218.4891815, 629.4920044, -846.4476318, 845.6838379

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767290, upper bound: 743.6767684
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6492777, upper bound: 743.6469480
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -115.5211716, 606.6237793, -114.7842484, 599.3131714, -714.8343506, 721.4080200
1: -187.4761200, 719.8009644, -186.5324554, 711.6715088, -899.1476440, 906.3334351
2: -133.4274902, 746.3000488, -132.6469879, 737.5989380, -871.0264282, 878.9470215
3: -324.4160156, 629.2719116, -322.8726196, 623.2428589, -947.6588745, 952.1445312
4: -220.2434692, 637.3926392, -219.0517883, 631.1282349, -851.3714600, 856.4442749

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6790280, upper bound: 743.6796930
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6505496, upper bound: 743.6486773
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -117.2255402, 614.6641235, -120.6374054, 630.2476196, -747.4731445, 735.3015137
1: -190.4830475, 729.5172729, -196.1681519, 748.3103027, -938.7932739, 925.6853638
2: -135.4348450, 756.2741699, -139.4232330, 775.6613770, -911.0961914, 895.6973877
3: -329.4708862, 638.2944946, -339.4631042, 655.3262939, -984.7971802, 977.7573853
4: -223.4595490, 646.3390503, -230.1700592, 663.7087402, -887.1682739, 876.5090942

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741961, upper bound: 743.6735831
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762791, upper bound: 743.6753815
time: 0.68 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.88 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6743380, upper bound: 743.6777684
IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6775307, upper bound: 743.6796402
IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6743946, upper bound: 743.6780436
IS_A1_B2_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6777709, upper bound: 743.6799094
IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6762881, upper bound: 743.6796699
IS_A1_B2_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6744914, upper bound: 743.6765870
IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6762881, upper bound: 743.6796699
IS_A1_B2_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6744914, upper bound: 743.6765870
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6770647, upper bound: 743.6783773
IS_A1_B2_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6776514, upper bound: 743.6791628
IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6763171, upper bound: 743.6764995
IS_A1_B2_A2_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6801566, upper bound: 743.6812262
IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6751712, upper bound: 743.6761208
IS_A1_B2_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6682327, upper bound: 743.6698932
IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6771361, upper bound: 743.6766804
IS_A1_B2_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6801547, upper bound: 743.6796240
IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6682781, upper bound: 743.6676539
IS_A1_B2_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6636253, upper bound: 743.6633431
IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6767290, upper bound: 743.6767684
IS_A1_B2_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6492777, upper bound: 743.6469480
IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6790280, upper bound: 743.6796930
IS_A1_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6505496, upper bound: 743.6486773
IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6741961, upper bound: 743.6735831
IS_A1_B2_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.88
Output dim: 0, lower bound: -743.6762791, upper bound: 743.6753815

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -98.8066635, 514.7426758, -92.9012756, 474.7951965, -573.6018066, 607.6439209
1: -161.3678894, 610.9003296, -150.4499207, 566.0700073, -727.4377441, 761.3502197
2: -114.0396881, 633.3759155, -106.6948776, 583.9417725, -697.9814453, 740.0708008
3: -278.1366577, 535.1267700, -259.5195007, 501.0173340, -779.1539307, 794.6462402
4: -187.5753937, 542.8269653, -176.3803558, 504.8978882, -692.4732666, 719.2073364

Time for backsubstitution: 1.86 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=860.0533447265625
rel_dist={0: [-743.6887393664056, 743.6887393664056]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1121.90 seconds
