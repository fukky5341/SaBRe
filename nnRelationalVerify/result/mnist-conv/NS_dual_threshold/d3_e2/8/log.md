## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.370427409


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.4328885, -5.0901341, -6.4328885, -5.0901341, -1.2011700, 1.2011700)
1: (-13.9913034, -12.7177610, -13.9913034, -12.7177610, -0.8999453, 0.8999455)
2: (-5.9237208, -4.6270599, -5.9237208, -4.6270599, -1.0896006, 1.0896006)
3: (-8.4110985, -7.2623472, -8.4110985, -7.2623472, -0.8056455, 0.8056452)
4: (-11.0883484, -9.6427240, -11.0883484, -9.6427240, -0.9431391, 0.9431391)
5: (0.0829567, 1.1426759, 0.0829567, 1.1426759, -0.9501886, 0.9501886)
6: (-4.6835279, -3.2830725, -4.6835279, -3.2830725, -0.8429096, 0.8429098)
7: (-11.3278637, -9.9057770, -11.3278637, -9.9057770, -1.0050030, 1.0050030)
8: (6.9650993, 7.9060383, 6.9650993, 7.9060383, -0.6467774, 0.6467774)
9: (-5.0575700, -3.9295335, -5.0575700, -3.9295335, -0.6523471, 0.6523471)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.22 + 32.93 = 56.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3741690, upper bound: 0.3741700

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: B, layer: 1, pos: 5859
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741162, upper bound: 0.3723378
time: 3.10 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741684, upper bound: 0.3741683
time: 3.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.53
Output dim: 8, lower bound: -0.3741162, upper bound: 0.3723378
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.53
Output dim: 8, lower bound: -0.3741684, upper bound: 0.3741683

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.4289141, -5.0903602, -6.4315681, -5.0902090, -1.1965704, 1.1990318
1: -13.9911976, -12.7207947, -13.9912691, -12.7187662, -0.8982859, 0.8962612
2: -5.9235783, -4.6293335, -5.9236746, -4.6278133, -1.0887170, 1.0871844
3: -8.3978271, -7.2625570, -8.4067039, -7.2624154, -0.7918715, 0.8006234
4: -11.0882196, -9.6472206, -11.0883055, -9.6442146, -0.9408913, 0.9377785
5: 0.0862083, 1.1426196, 0.0840347, 1.1426573, -0.9463205, 0.9484134
6: -4.6772947, -3.2831826, -4.6814618, -3.2831111, -0.8351719, 0.8394589
7: -11.3269043, -9.9062395, -11.3275442, -9.9059277, -1.0003428, 1.0007520
8: 6.9660530, 7.9025559, 6.9654150, 7.9048858, -0.6445363, 0.6429994
9: -5.0574317, -3.9319105, -5.0575256, -3.9303229, -0.6505635, 0.6489787

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718098, upper bound: 0.3718507
time: 3.18 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741156, upper bound: 0.3723375
time: 3.04 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.4331870, -5.0864692, -6.4328871, -5.0901346, -1.1999612, 1.2048416
1: -13.9943619, -12.7176094, -13.9913044, -12.7177610, -0.9026847, 0.8989553
2: -5.9261918, -4.6264286, -5.9237194, -4.6270599, -1.0920134, 1.0894942
3: -8.4112215, -7.2486649, -8.4110947, -7.2623463, -0.7992148, 0.8146343
4: -11.0928583, -9.6422052, -11.0883455, -9.6427259, -0.9473019, 0.9421382
5: 0.0827411, 1.1453738, 0.0829582, 1.1426769, -0.9494681, 0.9526629
6: -4.6838055, -3.2764759, -4.6835237, -3.2830710, -0.8413341, 0.8486409
7: -11.3287945, -9.9045582, -11.3278627, -9.9057770, -1.0028558, 1.0078087
8: 6.9612737, 7.9062099, 6.9651003, 7.9060383, -0.6507235, 0.6458027
9: -5.0599542, -3.9292009, -5.0575705, -3.9295344, -0.6542122, 0.6522443

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718621, upper bound: 0.3736821
time: 3.25 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741678, upper bound: 0.3741688
time: 3.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.09 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 8, lower bound: -0.3718098, upper bound: 0.3718507
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 8, lower bound: -0.3741156, upper bound: 0.3723375
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 8, lower bound: -0.3718621, upper bound: 0.3736821
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.09
Output dim: 8, lower bound: -0.3741678, upper bound: 0.3741688

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -6.4256654, -5.0926218, -6.4309406, -5.0905991, -1.1937962, 1.1962452
1: -13.9904547, -12.7249231, -13.9911690, -12.7195683, -0.8955946, 0.8924487
2: -5.9211712, -4.6411390, -5.9234171, -4.6301179, -1.0847683, 1.0751123
3: -8.3959484, -7.2748008, -8.4065561, -7.2648363, -0.7878599, 0.7883425
4: -11.0864906, -9.6500769, -11.0879993, -9.6447392, -0.9383497, 0.9345400
5: 0.0903821, 1.1416533, 0.0848491, 1.1424687, -0.9422746, 0.9463067
6: -4.6758885, -3.2953758, -4.6814003, -3.2855129, -0.8314996, 0.8276124
7: -11.3224087, -9.9079590, -11.3266783, -9.9061804, -0.9955058, 0.9976950
8: 6.9771800, 7.9018469, 6.9676228, 7.9048653, -0.6326559, 0.6391914
9: -5.0503488, -3.9333394, -5.0561590, -3.9304605, -0.6433325, 0.6462069

Time for backsubstitution: 23.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
time: 3.50 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
time: 3.64 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4315681, -5.0902090, -1.1966877, 1.1988559
1: -13.9912004, -12.7207975, -13.9912691, -12.7187662, -0.8978910, 0.8987212
2: -5.9235792, -4.6293344, -5.9236746, -4.6278133, -1.0887165, 1.0829387
3: -8.3978262, -7.2625589, -8.4067039, -7.2624154, -0.7918711, 0.7927110
4: -11.0882206, -9.6472206, -11.0883055, -9.6442146, -0.9408908, 0.9413900
5: 0.0862069, 1.1426194, 0.0840347, 1.1426573, -0.9508414, 0.9484134
6: -4.6772938, -3.2831852, -4.6814618, -3.2831111, -0.8351719, 0.8310194
7: -11.3269062, -9.9062405, -11.3275442, -9.9059277, -1.0003424, 1.0018806
8: 6.9660549, 7.9025564, 6.9654150, 7.9048858, -0.6424892, 0.6426942
9: -5.0574307, -3.9319105, -5.0575256, -3.9303229, -0.6465459, 0.6489789

Time for backsubstitution: 23.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
time: 3.38 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
time: 3.56 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -6.4299326, -5.0887218, -6.4322591, -5.0905252, -1.1971841, 1.2020626
1: -13.9936228, -12.7217331, -13.9912033, -12.7185621, -0.8999968, 0.8951435
2: -5.9237876, -4.6382308, -5.9234643, -4.6293654, -1.0880623, 1.0774288
3: -8.4093437, -7.2609043, -8.4109507, -7.2647657, -0.7952042, 0.8023312
4: -11.0911274, -9.6450682, -11.0880413, -9.6432495, -0.9447603, 0.9388881
5: 0.0869572, 1.1444083, 0.0837715, 1.1424878, -0.9454308, 0.9505496
6: -4.6823950, -3.2886591, -4.6834641, -3.2854753, -0.8376622, 0.8367970
7: -11.3243074, -9.9062643, -11.3269968, -9.9060307, -0.9980216, 1.0047641
8: 6.9723988, 7.9054956, 6.9673114, 7.9060168, -0.6388497, 0.6419919
9: -5.0528674, -3.9306281, -5.0562057, -3.9296720, -0.6469827, 0.6494751

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
time: 3.13 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
time: 3.16 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -6.4331875, -5.0864711, -6.4328871, -5.0901346, -1.2000790, 1.2046652
1: -13.9943619, -12.7176094, -13.9913044, -12.7177610, -0.9022903, 0.9014170
2: -5.9261918, -4.6264296, -5.9237194, -4.6270599, -1.0920129, 1.0852480
3: -8.4112206, -7.2486658, -8.4110947, -7.2623463, -0.7992148, 0.8067122
4: -11.0928593, -9.6422081, -11.0883455, -9.6427259, -0.9473019, 0.9457493
5: 0.0827421, 1.1453743, 0.0829582, 1.1426769, -0.9539742, 0.9526620
6: -4.6838045, -3.2764754, -4.6835237, -3.2830710, -0.8413341, 0.8402014
7: -11.3287954, -9.9045591, -11.3278627, -9.9057770, -1.0028553, 1.0089374
8: 6.9612741, 7.9062104, 6.9651003, 7.9060383, -0.6486509, 0.6454985
9: -5.0599523, -3.9292006, -5.0575705, -3.9295344, -0.6501956, 0.6522439

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741165
time: 3.14 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741690
time: 3.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.90 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741165
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741690

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -6.4256654, -5.0926218, -6.4282866, -5.0907536, -1.1933331, 1.1933208
1: -13.9904547, -12.7249231, -13.9911003, -12.7215977, -0.8932381, 0.8921180
2: -5.9211712, -4.6411390, -5.9233222, -4.6316385, -1.0831733, 1.0750504
3: -8.3959484, -7.2748008, -8.3976803, -7.2649770, -0.7875147, 0.7792447
4: -11.0864906, -9.6500769, -11.0879116, -9.6477451, -0.9348760, 0.9341846
5: 0.0903821, 1.1416533, 0.0870193, 1.1424296, -0.9419031, 0.9438496
6: -4.6758885, -3.2953758, -4.6772337, -3.2855847, -0.8308313, 0.8226566
7: -11.3224087, -9.9079590, -11.3260393, -9.9064960, -0.9934692, 0.9952497
8: 6.9771800, 7.9018469, 6.9682570, 7.9025364, -0.6303735, 0.6384408
9: -5.0503488, -3.9333394, -5.0560656, -3.9320476, -0.6412439, 0.6457022

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3700318
time: 3.33 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
time: 3.36 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -6.4256654, -5.0926218, -6.4325590, -5.0868611, -1.1976986, 1.1975541
1: -13.9904547, -12.7249231, -13.9942617, -12.7184105, -0.8964643, 0.8953533
2: -5.9211712, -4.6411390, -5.9259357, -4.6287317, -1.0862494, 1.0775547
3: -8.3959484, -7.2748008, -8.4110756, -7.2510834, -0.7911737, 0.7926171
4: -11.0864906, -9.6500769, -11.0925512, -9.6427317, -0.9398813, 0.9388802
5: 0.0903821, 1.1416533, 0.0835600, 1.1451856, -0.9449329, 0.9472809
6: -4.6758885, -3.2953758, -4.6837444, -3.2788749, -0.8332801, 0.8292806
7: -11.3224087, -9.9079590, -11.3279295, -9.9048119, -0.9953423, 0.9967594
8: 6.9771800, 7.9018469, 6.9634809, 7.9061894, -0.6340175, 0.6410687
9: -5.0503488, -3.9333394, -5.0585880, -3.9293373, -0.6439886, 0.6483216

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3700318
time: 3.30 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
time: 3.42 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4289141, -5.0903602, -1.1962223, 1.1959314
1: -13.9912004, -12.7207975, -13.9911976, -12.7207947, -0.8955345, 0.8983881
2: -5.9235792, -4.6293344, -5.9235783, -4.6293335, -1.0871220, 1.0828757
3: -8.3978262, -7.2625589, -8.3978271, -7.2625570, -0.7915258, 0.7836134
4: -11.0882206, -9.6472206, -11.0882196, -9.6472206, -0.9374223, 0.9410346
5: 0.0862069, 1.1426194, 0.0862083, 1.1426196, -0.9504700, 0.9459486
6: -4.6772938, -3.2831852, -4.6772947, -3.2831826, -0.8345027, 0.8260639
7: -11.3269062, -9.9062405, -11.3269043, -9.9062395, -0.9983068, 0.9994364
8: 6.9660549, 7.9025564, 6.9660530, 7.9025559, -0.6402059, 0.6419487
9: -5.0574307, -3.9319105, -5.0574317, -3.9319105, -0.6444578, 0.6484752

Time for backsubstitution: 22.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3700314
time: 3.27 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3707032
time: 3.28 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4331870, -5.0864692, -1.2005944, 1.2001653
1: -13.9912004, -12.7207975, -13.9943619, -12.7176094, -0.8987603, 0.9016242
2: -5.9235792, -4.6293344, -5.9261918, -4.6264286, -1.0901966, 1.0853801
3: -8.3978262, -7.2625589, -8.4112215, -7.2486649, -0.7999411, 0.7969868
4: -11.0882206, -9.6472206, -11.0928583, -9.6422052, -0.9424391, 0.9457300
5: 0.0862069, 1.1426194, 0.0827411, 1.1453738, -0.9535007, 0.9493971
6: -4.6772938, -3.2831852, -4.6838055, -3.2764759, -0.8412364, 0.8326871
7: -11.3269062, -9.9062405, -11.3287945, -9.9045582, -1.0001779, 1.0009460
8: 6.9660549, 7.9025564, 6.9612737, 7.9062099, -0.6438458, 0.6470063
9: -5.0574307, -3.9319105, -5.0599542, -3.9292009, -0.6472027, 0.6510940

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3700307
time: 5.89 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
time: 3.41 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -6.4299326, -5.0887218, -6.4282866, -5.0907536, -1.1975598, 1.1976933
1: -13.9936228, -12.7217331, -13.9911003, -12.7215977, -0.8964763, 0.8953438
2: -5.9237876, -4.6382308, -5.9233222, -4.6316385, -1.0856752, 1.0781307
3: -8.4093437, -7.2609043, -8.3976803, -7.2649770, -0.7975411, 0.7887268
4: -11.0911274, -9.6450682, -11.0879116, -9.6477451, -0.9395704, 0.9391692
5: 0.0869572, 1.1444083, 0.0870193, 1.1424296, -0.9453282, 0.9468794
6: -4.6823950, -3.2886591, -4.6772337, -3.2855847, -0.8366950, 0.8293924
7: -11.3243074, -9.9062643, -11.3260393, -9.9064960, -0.9949808, 0.9971333
8: 6.9723988, 7.9054956, 6.9682570, 7.9025364, -0.6354377, 0.6415682
9: -5.0528674, -3.9306281, -5.0560656, -3.9320476, -0.6438627, 0.6484466

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718109
time: 3.31 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
time: 3.22 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.4299326, -5.0887218, -6.4325590, -5.0868611, -1.1974859, 1.1974745
1: -13.9936228, -12.7217331, -13.9942617, -12.7184105, -0.8963718, 0.8952549
2: -5.9237876, -4.6382308, -5.9259357, -4.6287317, -1.0863338, 1.0782175
3: -8.4093437, -7.2609043, -8.4110756, -7.2510834, -0.7954693, 0.7871938
4: -11.0911274, -9.6450682, -11.0925512, -9.6427317, -0.9397316, 0.9390347
5: 0.0869572, 1.1444083, 0.0835600, 1.1451856, -0.9457464, 0.9476724
6: -4.6823950, -3.2886591, -4.6837444, -3.2788749, -0.8379278, 0.8297522
7: -11.3243074, -9.9062643, -11.3279295, -9.9048119, -1.0034475, 1.0052371
8: 6.9723988, 7.9054956, 6.9634809, 7.9061894, -0.6354079, 0.6434746
9: -5.0528674, -3.9306281, -5.0585880, -3.9293373, -0.6452122, 0.6496720

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718110
time: 3.37 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
time: 3.61 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.4331875, -5.0864711, -6.4289141, -5.0903602, -1.2004547, 1.2002964
1: -13.9943619, -12.7176094, -13.9911976, -12.7207947, -0.8987703, 0.9016135
2: -5.9261918, -4.6264296, -5.9235783, -4.6293335, -1.0896268, 1.0859504
3: -8.4112206, -7.2486658, -8.3978271, -7.2625570, -0.8048992, 0.7931077
4: -11.0928593, -9.6422081, -11.0882196, -9.6472206, -0.9421182, 0.9460502
5: 0.0827421, 1.1453743, 0.0862083, 1.1426196, -0.9539027, 0.9489784
6: -4.6838045, -3.2764754, -4.6772947, -3.2831826, -0.8411264, 0.8327971
7: -11.3287954, -9.9045591, -11.3269043, -9.9062395, -0.9998155, 1.0013075
8: 6.9612741, 7.9062104, 6.9660530, 7.9025559, -0.6452384, 0.6455944
9: -5.0599523, -3.9292006, -5.0574317, -3.9319105, -0.6470773, 0.6512198

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3718098
time: 4.03 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3724822
time: 3.22 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.4331875, -5.0864711, -6.4331870, -5.0864692, -1.2003822, 1.2000861
1: -13.9943619, -12.7176094, -13.9943619, -12.7176094, -0.8986726, 0.9015262
2: -5.9261918, -4.6264296, -5.9261918, -4.6264286, -1.0902834, 1.0860372
3: -8.4112206, -7.2486658, -8.4112215, -7.2486649, -0.7994823, 0.7915697
4: -11.0928593, -9.6422081, -11.0928583, -9.6422052, -0.9422841, 0.9458954
5: 0.0827421, 1.1453743, 0.0827411, 1.1453738, -0.9542890, 0.9497819
6: -4.6838045, -3.2764754, -4.6838055, -3.2764759, -0.8416004, 0.8331609
7: -11.3287954, -9.9045591, -11.3287945, -9.9045582, -1.0082812, 1.0094104
8: 6.9612741, 7.9062104, 6.9612737, 7.9062099, -0.6452048, 0.6469786
9: -5.0599523, -3.9292006, -5.0599542, -3.9292009, -0.6484251, 0.6524417

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3718621
time: 5.00 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718630
time: 3.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.60 seconds
NS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3700318
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
NS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3700318
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3700314
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3707032
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3700307
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718109
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718110
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3718098
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3724822
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718496, upper bound: 0.3718621
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.60
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718630

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.4256654, -5.0926218, -6.4289131, -5.0903616, -1.1935163, 1.1938338
1: -13.9904547, -12.7249231, -13.9912004, -12.7207975, -0.8945837, 0.8918655
2: -5.9211712, -4.6411390, -5.9235792, -4.6293344, -1.0855165, 1.0751443
3: -8.3959484, -7.2748008, -8.3978262, -7.2625589, -0.7899237, 0.7793305
4: -11.0864906, -9.6500769, -11.0882206, -9.6472206, -0.9349384, 0.9351382
5: 0.0903821, 1.1416533, 0.0862069, 1.1426194, -0.9421797, 0.9449105
6: -4.6758885, -3.2953758, -4.6772938, -3.2831852, -0.8330185, 0.8227048
7: -11.3224087, -9.9079590, -11.3269062, -9.9062405, -0.9936671, 0.9963160
8: 6.9771800, 7.9018469, 6.9660549, 7.9025564, -0.6301064, 0.6405888
9: -5.0503488, -3.9333394, -5.0574307, -3.9319105, -0.6413765, 0.6470630

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3717893
time: 3.53 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700300, upper bound: 0.3718503
time: 3.49 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.4256654, -5.0926218, -6.4331875, -5.0864711, -1.1978807, 1.1980677
1: -13.9904547, -12.7249231, -13.9943619, -12.7176094, -0.8978095, 0.8951008
2: -5.9211712, -4.6411390, -5.9261918, -4.6264296, -1.0885911, 1.0776491
3: -8.3959484, -7.2748008, -8.4112206, -7.2486658, -0.7913864, 0.7927041
4: -11.0864906, -9.6500769, -11.0928593, -9.6422081, -0.9399538, 0.9398341
5: 0.0903821, 1.1416533, 0.0827421, 1.1453743, -0.9452095, 0.9483595
6: -4.6758885, -3.2953758, -4.6838045, -3.2764754, -0.8334248, 0.8293290
7: -11.3224087, -9.9079590, -11.3287954, -9.9045591, -0.9955392, 0.9978247
8: 6.9771800, 7.9018469, 6.9612741, 7.9062104, -0.6337519, 0.6420842
9: -5.0503488, -3.9333394, -5.0599523, -3.9292006, -0.6441216, 0.6496828

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3711621, upper bound: 0.3717894
time: 3.17 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718090, upper bound: 0.3718502
time: 3.51 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4256654, -5.0926218, -1.1938343, 1.1935158
1: -13.9912004, -12.7207975, -13.9904547, -12.7249231, -0.8918657, 0.8945837
2: -5.9235792, -4.6293344, -5.9211712, -4.6411390, -1.0751443, 1.0855165
3: -8.3978262, -7.2625589, -8.3959484, -7.2748008, -0.7793303, 0.7899237
4: -11.0882206, -9.6472206, -11.0864906, -9.6500769, -0.9351382, 0.9349382
5: 0.0862069, 1.1426194, 0.0903821, 1.1416533, -0.9449110, 0.9421797
6: -4.6772938, -3.2831852, -4.6758885, -3.2953758, -0.8227043, 0.8330185
7: -11.3269062, -9.9062405, -11.3224087, -9.9079590, -0.9963160, 0.9936671
8: 6.9660549, 7.9025564, 6.9771800, 7.9018469, -0.6405888, 0.6301064
9: -5.0574307, -3.9319105, -5.0503488, -3.9333394, -0.6470633, 0.6413767

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3717882, upper bound: 0.3693833
time: 3.84 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718491, upper bound: 0.3700309
time: 3.19 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4289131, -5.0903616, -1.1962242, 1.1962237
1: -13.9912004, -12.7207975, -13.9912004, -12.7207975, -0.8983874, 0.8983874
2: -5.9235792, -4.6293344, -5.9235792, -4.6293344, -1.0828753, 1.0828753
3: -8.3978262, -7.2625589, -8.3978262, -7.2625589, -0.7836137, 0.7836137
4: -11.0882206, -9.6472206, -11.0882206, -9.6472206, -0.9410343, 0.9410343
5: 0.0862069, 1.1426194, 0.0862069, 1.1426194, -0.9504700, 0.9504700
6: -4.6772938, -3.2831852, -4.6772938, -3.2831852, -0.8260632, 0.8260632
7: -11.3269062, -9.9062405, -11.3269062, -9.9062405, -0.9994359, 0.9994359
8: 6.9660549, 7.9025564, 6.9660549, 7.9025564, -0.6402054, 0.6402054
9: -5.0574307, -3.9319105, -5.0574307, -3.9319105, -0.6444578, 0.6444576

Time for backsubstitution: 22.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3712023, upper bound: 0.3699701
time: 3.08 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718493, upper bound: 0.3707027
time: 3.19 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.4289131, -5.0903616, -6.4299326, -5.0887218, -1.1982059, 1.1977434
1: -13.9912004, -12.7207975, -13.9936228, -12.7217331, -0.8950911, 0.8978219
2: -5.9235792, -4.6293344, -5.9237876, -4.6382308, -1.0782251, 1.0880179
3: -8.3978262, -7.2625589, -8.4093437, -7.2609043, -0.7877223, 0.7977536
4: -11.0882206, -9.6472206, -11.0911274, -9.6450682, -0.9401226, 0.9396327
5: 0.0862069, 1.1426194, 0.0869572, 1.1444083, -0.9479408, 0.9456048
6: -4.6772938, -3.2831852, -4.6823950, -3.2886591, -0.8294406, 0.8368382
7: -11.3269062, -9.9062405, -11.3243074, -9.9062643, -0.9981990, 0.9951787
8: 6.9660549, 7.9025564, 6.9723988, 7.9054956, -0.6425877, 0.6351707
9: -5.0574307, -3.9319105, -5.0528674, -3.9306281, -0.6498075, 0.6439954

Time for backsubstitution: 22.32 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.15 + 563.23 = 619.38 seconds
