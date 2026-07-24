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
execution time: IAR + RelationalAnalysis = 22.51 + 32.66 = 55.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3741690, upper bound: 0.3741700

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5859
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5859

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741162, upper bound: 0.3723378
time: 2.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3741684, upper bound: 0.3741683
time: 3.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.27
Output dim: 8, lower bound: -0.3741162, upper bound: 0.3723378
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.27
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

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3723378
time: 3.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3723378
time: 3.28 seconds

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

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5859
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 5859

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3741171
time: 3.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3741693
time: 3.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.07 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3723378
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3723378
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3741171
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.07
Output dim: 8, lower bound: -0.3723369, upper bound: 0.3741693

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.4289141, -5.0903602, -6.4289141, -5.0903602, -1.1961074, 1.1961074
1: -13.9911976, -12.7207947, -13.9911976, -12.7207947, -0.8959303, 0.8959298
2: -5.9235783, -4.6293335, -5.9235783, -4.6293335, -1.0871224, 1.0871224
3: -8.3978271, -7.2625570, -8.3978271, -7.2625570, -0.7915258, 0.7915258
4: -11.0882196, -9.6472206, -11.0882196, -9.6472206, -0.9374232, 0.9374232
5: 0.0862083, 1.1426196, 0.0862083, 1.1426196, -0.9459486, 0.9459486
6: -4.6772947, -3.2831826, -4.6772947, -3.2831826, -0.8345027, 0.8345027
7: -11.3269043, -9.9062395, -11.3269043, -9.9062395, -0.9983072, 0.9983072
8: 6.9660530, 7.9025559, 6.9660530, 7.9025559, -0.6422539, 0.6422536
9: -5.0574317, -3.9319105, -5.0574317, -3.9319105, -0.6484752, 0.6484752

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
time: 3.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
time: 3.26 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.4289141, -5.0903602, -6.4331870, -5.0864692, -1.2004728, 1.2003417
1: -13.9911976, -12.7207947, -13.9943619, -12.7176094, -0.8991551, 0.8991644
2: -5.9235783, -4.6293335, -5.9261918, -4.6264286, -1.0901971, 1.0896273
3: -8.3978271, -7.2625570, -8.4112215, -7.2486649, -0.8010299, 0.8048990
4: -11.0882196, -9.6472206, -11.0928583, -9.6422052, -0.9424391, 0.9421186
5: 0.0862083, 1.1426196, 0.0827411, 1.1453738, -0.9489794, 0.9493976
6: -4.6772947, -3.2831826, -4.6838055, -3.2764759, -0.8412359, 0.8411264
7: -11.3269043, -9.9062395, -11.3287945, -9.9045582, -1.0001788, 0.9998159
8: 6.9660530, 7.9025559, 6.9612737, 7.9062099, -0.6458988, 0.6473114
9: -5.0574317, -3.9319105, -5.0599542, -3.9292009, -0.6512203, 0.6510940

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
time: 3.34 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
time: 3.42 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.4331870, -5.0864692, -6.4289141, -5.0903602, -1.2003417, 1.2004728
1: -13.9943619, -12.7176094, -13.9911976, -12.7207947, -0.8991642, 0.8991554
2: -5.9261918, -4.6264286, -5.9235783, -4.6293335, -1.0896273, 1.0901971
3: -8.4112215, -7.2486649, -8.3978271, -7.2625570, -0.8048992, 0.8010299
4: -11.0928583, -9.6422052, -11.0882196, -9.6472206, -0.9421186, 0.9424393
5: 0.0827411, 1.1453738, 0.0862083, 1.1426196, -0.9493976, 0.9489794
6: -4.6838055, -3.2764759, -4.6772947, -3.2831826, -0.8411264, 0.8412361
7: -11.3287945, -9.9045582, -11.3269043, -9.9062395, -0.9998159, 1.0001788
8: 6.9612737, 7.9062099, 6.9660530, 7.9025559, -0.6473114, 0.6458988
9: -5.0599542, -3.9292009, -5.0574317, -3.9319105, -0.6510940, 0.6512203

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741165
time: 3.35 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.4331870, -5.0864692, -6.4331870, -5.0864692, -1.2002630, 1.2002630
1: -13.9943619, -12.7176094, -13.9943619, -12.7176094, -0.8990669, 0.8990667
2: -5.9261918, -4.6264286, -5.9261918, -4.6264286, -1.0902843, 1.0902839
3: -8.4112215, -7.2486649, -8.4112215, -7.2486649, -0.7994823, 0.7994821
4: -11.0928583, -9.6422052, -11.0928583, -9.6422052, -0.9422846, 0.9422846
5: 0.0827411, 1.1453738, 0.0827411, 1.1453738, -0.9497828, 0.9497828
6: -4.6838055, -3.2764759, -4.6838055, -3.2764759, -0.8416004, 0.8416004
7: -11.3287945, -9.9045582, -11.3287945, -9.9045582, -1.0082817, 1.0082817
8: 6.9612737, 7.9062099, 6.9612737, 7.9062099, -0.6472831, 0.6472828
9: -5.0599542, -3.9292009, -5.0599542, -3.9292009, -0.6524420, 0.6524417

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
time: 3.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741690
time: 3.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.45 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718508
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3718506
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3723375
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736298
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741165
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3700305, upper bound: 0.3736301
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.45
Output dim: 8, lower bound: -0.3723363, upper bound: 0.3741690

## BFS NS instance: NS_A1_B1_A1

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

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 945

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3717893
time: 3.65 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700299, upper bound: 0.3718503
time: 3.31 seconds

## BFS NS instance: NS_A1_B1_A2

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

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
time: 3.29 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
time: 3.22 seconds

## BFS NS instance: NS_A1_B2_A1

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

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 945

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3711621, upper bound: 0.3717894
time: 3.18 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718090, upper bound: 0.3718502
time: 3.52 seconds

## BFS NS instance: NS_A1_B2_A2

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

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3736289, upper bound: 0.3700314
time: 3.32 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3736289, upper bound: 0.3700313
time: 3.34 seconds

## BFS NS instance: NS_A2_B1_A1

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

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3735684
time: 3.71 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3700299, upper bound: 0.3736293
time: 3.25 seconds

## BFS NS instance: NS_A2_B1_A2

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

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718105
time: 3.20 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718105
time: 3.40 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 945

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 5745

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3694803, upper bound: 0.3736210
time: 3.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3701272, upper bound: 0.3736816
time: 3.70 seconds

## BFS NS instance: NS_A2_B2_A2

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

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 5745
type: B, layer: 1, pos: 945

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3719471, upper bound: 0.3718630
time: 3.50 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3719471, upper bound: 0.3718628
time: 3.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3717893
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3700299, upper bound: 0.3718503
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3700314
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3711621, upper bound: 0.3717894
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3718090, upper bound: 0.3718502
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3736289, upper bound: 0.3700314
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3736289, upper bound: 0.3700313
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3735684
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3700299, upper bound: 0.3736293
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718105
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3718498, upper bound: 0.3718105
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3694803, upper bound: 0.3736210
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3701272, upper bound: 0.3736816
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3719471, upper bound: 0.3718630
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.67
Output dim: 8, lower bound: -0.3719471, upper bound: 0.3718628

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.4243078, -5.0926876, -6.4232259, -5.0909910, -1.1894145, 1.1864038
1: -13.9900246, -12.7249336, -13.9894934, -12.7216425, -0.8926692, 0.8904300
2: -5.9211197, -4.6422291, -5.9231277, -4.6357040, -1.0786781, 1.0732269
3: -8.3955355, -7.2749395, -8.3961544, -7.2654958, -0.7842960, 0.7759717
4: -11.0863876, -9.6523132, -11.0875444, -9.6560831, -0.9257140, 0.9303913
5: 0.0905720, 1.1393939, 0.0877488, 1.1339967, -0.9332228, 0.9407539
6: -4.6715417, -3.2954633, -4.6609983, -3.2859209, -0.8261776, 0.8063447
7: -11.3223562, -9.9110031, -11.3258400, -9.9178543, -0.9819407, 0.9920645
8: 6.9775276, 7.9014106, 6.9695745, 7.9009051, -0.6283550, 0.6365101
9: -5.0489578, -3.9334123, -5.0508690, -3.9323232, -0.6395769, 0.6404028

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3712032
time: 3.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3693830, upper bound: 0.3717893
time: 3.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.4256625, -5.0926218, -6.4289722, -5.0844955, -1.1976857, 1.1937556
1: -13.9904537, -12.7249222, -13.9911804, -12.7192259, -0.8957362, 0.8916962
2: -5.9211712, -4.6411414, -5.9286280, -4.6306524, -1.0827193, 1.0798059
3: -8.3959484, -7.2748022, -8.3982468, -7.2635083, -0.7870579, 0.7809348
4: -11.0864887, -9.6500816, -11.0986423, -9.6474514, -0.9331331, 0.9438035
5: 0.0903831, 1.1416510, 0.0746477, 1.1424866, -0.9390531, 0.9529197
6: -4.6758842, -3.2953753, -4.6772676, -3.2606843, -0.8340595, 0.8167703
7: -11.3224096, -9.9079647, -11.3421564, -9.9062786, -0.9895549, 1.0073874
8: 6.9771805, 7.9018464, 6.9659462, 7.9025569, -0.6303020, 0.6404157
9: -5.0503464, -3.9333386, -5.0560946, -3.9242957, -0.6490211, 0.6440212

Time for backsubstitution: 22.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3699691, upper bound: 0.3712033
time: 3.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3699691, upper bound: 0.3712033
time: 3.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1

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

Time for backsubstitution: 22.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3717882, upper bound: 0.3693833
time: 4.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718491, upper bound: 0.3700309
time: 3.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2

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

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5745
type: A, layer: 1, pos: 945

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5745

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3717885, upper bound: 0.3693840
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3718494, upper bound: 0.3700309
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.4243078, -5.0926876, -6.4274979, -5.0870953, -1.1937819, 1.1906362
1: -13.9900246, -12.7249336, -13.9926567, -12.7184525, -0.8958950, 0.8936648
2: -5.9211197, -4.6422291, -5.9257417, -4.6327982, -1.0817547, 1.0757318
3: -8.3955355, -7.2749395, -8.4095488, -7.2515998, -0.7876861, 0.7893443
4: -11.0863876, -9.6523132, -11.0921812, -9.6510735, -0.9307094, 0.9350858
5: 0.0905720, 1.1393939, 0.0842896, 1.1367501, -0.9362535, 0.9441833
6: -4.6715417, -3.2954633, -4.6675081, -3.2792087, -0.8268137, 0.8129683
7: -11.3223562, -9.9110031, -11.3277369, -9.9161701, -0.9838152, 0.9935699
8: 6.9775276, 7.9014106, 6.9647894, 7.9045582, -0.6319993, 0.6389936
9: -5.0489578, -3.9334123, -5.0533905, -3.9296114, -0.6423218, 0.6430213

Time for backsubstitution: 22.51 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.17 + 558.32 = 613.48 seconds
