## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Result
execution time: IAR + LP analysis = 1.80 + 1.78 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065


# Binary Search by BASE starts (time budget: 1196.42 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=232.61239624023438
rel_dist={3: [-187.8886779558913, 187.8886779558913]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=232.61239624023438
rel_dist={3: [-187.88282977162675, 187.88282977162675]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=232.61239624023438
rel_dist={3: [-187.8796990044812, 187.87969900448115]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=232.61239624023438
rel_dist={3: [-187.87806065812612, 187.87806065812612]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=232.61239624023438
rel_dist={3: [-187.87722761678833, 187.87722761678833]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=232.61239624023438
rel_dist={3: [-187.87681109620016, 187.87681109620019]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=232.61239624023438
rel_dist={3: [-187.87660283606715, 187.87660283606715]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=232.61239624023438
rel_dist={3: [-187.87649870699573, 187.87649870699568]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=232.61239624023438
rel_dist={3: [-187.87644665103846, 187.87644665103846]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=232.61239624023438
rel_dist={3: [-187.87642062420437, 187.87642062420434]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=232.61239624023438
rel_dist={3: [-187.87640761299804, 187.87640761299804]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=232.61239624023438
rel_dist={3: [-187.8764011114574, 187.87640150941382]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=232.61239624023438
rel_dist={3: [-187.87639785304535, 187.87639806237894]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=232.61239624023438
rel_dist={3: [-187.87640340339, 187.87639952386695]}

## Binary Search Result
Binary search time: 59.83 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1136.59 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8813359, upper bound: 187.6535146
time: 0.74 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6590011
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.52 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -187.8813359, upper bound: 187.6535146
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -187.6590011, upper bound: 187.6590011

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -149.6440735, 126.7424088, -119.7027283, 108.6696014, -258.3136597, 246.4451294
1: -117.3338928, 118.4335785, -93.7178497, 101.6046600, -218.9385529, 212.1514282
2: -169.7016296, 131.6250763, -135.5732574, 113.5433578, -283.2449341, 267.1983337
3: -63.3496017, 169.2627869, -54.4398117, 137.3926697, -200.7422791, 223.7026062
4: -188.6523895, 133.4867859, -150.9911346, 114.0707092, -302.7230835, 284.4779053

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6315550
time: 0.66 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
time: 0.64 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -149.5794983, 126.7029190, -184.0042267, 156.9276886, -306.5072021, 310.7071533
1: -117.2834854, 118.3959808, -144.4059448, 147.8805542, -265.1640320, 262.8019104
2: -169.6293488, 131.5840454, -208.6540070, 163.4298706, -333.0590820, 340.2380371
3: -63.3297729, 169.1976776, -80.1929092, 207.0820923, -270.4118652, 249.3905945
4: -188.5724487, 133.4454041, -232.1647949, 164.1955109, -352.7679138, 365.6101990

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6400201
time: 0.71 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6396430
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.31 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6315550
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 3, lower bound: -187.6316936, upper bound: 187.6400201
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6396430

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.7027283, 108.6696014, -228.6269226, 228.4495850
1: -93.8448410, 101.7164536, -93.7178497, 101.6046600, -195.4494934, 195.4342957
2: -135.7941437, 113.6401825, -135.5732574, 113.5433578, -249.3374786, 249.2134399
3: -54.4526558, 137.4407501, -54.4398117, 137.3926697, -191.8453217, 191.8805542
4: -151.2667084, 114.1601410, -150.9911346, 114.0707092, -265.3374023, 265.1512756

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6311779
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6311779
time: 0.62 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -119.6233597, 108.6178360, -290.7781372, 273.2838745
1: -143.0398865, 144.6755829, -93.6554718, 101.5565948, -244.5964813, 238.3310089
2: -206.5966339, 159.9870605, -135.4834137, 113.4910812, -320.0877075, 295.4704590
3: -78.3434219, 204.8266907, -54.4135170, 137.3085632, -215.6519470, 259.2402039
4: -229.7526093, 160.7959442, -150.8911438, 114.0153961, -343.7680054, 311.6870728

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
time: 0.59 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
time: 0.61 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -119.8824997, 108.7030869, -184.0042267, 156.9276886, -276.8101807, 292.7073059
1: -93.7861786, 101.6748047, -144.4059448, 147.8805542, -241.6667328, 246.0807495
2: -135.7100830, 113.5954590, -208.6540070, 163.4298706, -299.1398926, 322.2494507
3: -54.4305954, 137.3644867, -80.1929092, 207.0820923, -261.5126953, 217.5573883
4: -151.1734619, 114.1141815, -232.1647949, 164.1955109, -315.3689575, 346.2789917

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6323670
time: 0.59 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
time: 0.59 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -182.1177826, 153.6374054, -183.9449921, 156.8887482, -339.0065308, 337.5823669
1: -143.0064392, 144.6536407, -144.3596649, 147.8446198, -290.8509827, 289.0132446
2: -206.5487366, 159.9633484, -208.5877686, 163.3905334, -369.9391479, 368.5511169
3: -78.3319092, 204.7839813, -80.1730652, 207.0210114, -285.3529053, 284.9569702
4: -229.7001953, 160.7719421, -232.0910187, 164.1533051, -393.8533936, 392.8628845

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6319900
time: 0.58 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6392715
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.09 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6311779
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6309287, upper bound: 187.6311779
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6289280, upper bound: 187.6323670
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6319900
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6392715

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -93.5899734, 92.8750916, -212.8324127, 202.3368378
1: -93.8448410, 101.7164536, -73.1107025, 87.0040359, -180.8488770, 174.8271484
2: -135.7941437, 113.6401825, -105.9443512, 97.7353897, -233.5295410, 219.5845337
3: -54.4526558, 137.4407501, -47.2811356, 109.5639648, -164.0166168, 184.7218933
4: -151.2667084, 114.1601410, -118.2672424, 97.3865814, -248.6532898, 232.4273682

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
time: 0.57 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
time: 0.58 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -148.4843750, 134.0473938, -254.0047150, 257.2312622
1: -93.8448410, 101.7164536, -116.2964935, 126.5755463, -220.4203796, 218.0129395
2: -135.7941437, 113.6401825, -168.2982788, 140.5842438, -276.3783875, 281.9384766
3: -54.4526558, 137.4407501, -68.7259674, 169.2932587, -223.7459106, 206.1667023
4: -151.2667084, 114.1601410, -187.5333405, 139.8242035, -291.0908813, 301.6934814

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -119.6233597, 108.6178360, -257.9017639, 254.3126678
1: -116.9905624, 127.1221771, -93.6554718, 101.5565948, -218.5471497, 220.7776184
2: -169.2326202, 141.1778259, -135.4834137, 113.4910812, -282.7236938, 276.6612244
3: -69.0598907, 170.1712341, -54.4135170, 137.3085632, -206.3684387, 224.5847473
4: -188.5170746, 140.5437622, -150.8911438, 114.0153961, -302.5324707, 291.4349060

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -119.6233597, 108.6178360, -323.3409424, 305.5965576
1: -168.6910553, 175.9269562, -93.6554718, 101.5565948, -270.2476501, 269.5824280
2: -243.5518188, 193.9766388, -135.4834137, 113.4910812, -357.0429077, 329.4600525
3: -95.6573029, 241.0182190, -54.4135170, 137.3085632, -232.3889923, 295.4316711
4: -271.0773621, 193.9729767, -150.8911438, 114.0153961, -385.0927124, 344.8641357

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -116.4163284, 105.8416977, -117.7962265, 111.3588486, -227.7751617, 223.6379242
1: -91.0683441, 98.9942398, -92.3771362, 105.3934631, -196.4618073, 191.3713684
2: -131.7942505, 110.5843964, -133.9635315, 116.2873993, -248.0816498, 244.5478973
3: -52.9523697, 133.6402130, -56.3376808, 137.7161255, -190.6684875, 189.9778748
4: -146.8222351, 111.1372223, -149.6368866, 115.8507843, -262.6730347, 260.7740784

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
time: 0.83 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -119.8824997, 108.7030869, -179.9100342, 154.1128998, -273.9953918, 288.6131287
1: -93.7861786, 101.6748047, -141.1715698, 145.1972504, -238.9834290, 242.8463745
2: -135.7100830, 113.5954590, -204.0370789, 160.5492554, -296.2593079, 317.6325378
3: -54.4305954, 137.3644867, -78.8057022, 202.6504364, -257.0810242, 216.1701508
4: -151.1734619, 114.1141815, -227.0235748, 161.2098389, -312.3833008, 341.1377258

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6312977
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -178.2720490, 150.4557800, -117.7430115, 111.3235092, -289.5954895, 268.1987610
1: -139.9651794, 141.6920624, -92.3358231, 105.3603516, -245.3254852, 234.0278931
2: -202.1978149, 156.6364746, -133.9037476, 116.2510834, -318.4489136, 290.5402222
3: -76.6667099, 200.6600494, -56.3193588, 137.6603546, -214.3270416, 256.9794006
4: -224.8847961, 157.4253845, -149.5703735, 115.8125458, -340.6973267, 306.9957275

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6312251
time: 0.67 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -182.1177826, 153.6374054, -179.8515015, 154.0743866, -336.1921692, 333.4888916
1: -143.0064392, 144.6536407, -141.1258392, 145.1615906, -288.1679688, 285.7794495
2: -206.5487366, 159.9633484, -203.9714966, 160.5103149, -367.0590210, 363.9348450
3: -78.3319092, 204.7839813, -78.7860641, 202.5899963, -280.9218750, 283.5700378
4: -229.7001953, 160.7719421, -226.9505768, 161.1680298, -390.8682251, 387.7225037

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6309206
time: 0.71 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6309206
time: 0.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.12 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6306268, upper bound: 187.6315550
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6311779
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6312977
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6312251
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6309206
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -187.6392715, upper bound: 187.6309206

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -93.5899734, 92.8750916, -186.4650574, 186.4650574
1: -73.1107025, 87.0040359, -73.1107025, 87.0040359, -160.1147461, 160.1147308
2: -105.9443512, 97.7353897, -105.9443512, 97.7353897, -203.6797485, 203.6797485
3: -47.2811356, 109.5639648, -47.2811356, 109.5639648, -156.8450928, 156.8450928
4: -118.2672424, 97.3865814, -118.2672424, 97.3865814, -215.6537933, 215.6538086

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217075, upper bound: 187.6291626
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8241248, upper bound: 187.6312736
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -93.5899734, 92.8750916, -242.8496857, 230.9306641
1: -117.3068619, 129.7829132, -73.1107025, 87.0040359, -204.3108978, 202.8936005
2: -169.9014282, 144.0322723, -105.9443512, 97.7353897, -267.6368103, 249.9765320
3: -70.6789017, 171.0912170, -47.2811356, 109.5639648, -180.2428589, 218.3723450
4: -189.4996490, 143.3141022, -118.2672424, 97.3865814, -286.8861694, 261.5813599

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217075, upper bound: 187.6291626
time: 0.58 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8241248, upper bound: 187.6312736
time: 0.59 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -148.4843750, 134.0473938, -227.6373596, 241.3594666
1: -73.1107025, 87.0040359, -116.2964935, 126.5755463, -199.6862488, 203.3005371
2: -105.9443512, 97.7353897, -168.2982788, 140.5842438, -246.5285950, 266.0336609
3: -47.2811356, 109.5639648, -68.7259674, 169.2932587, -216.5744019, 178.2899323
4: -118.2672424, 97.3865814, -187.5333405, 139.8242035, -258.0914001, 284.9199219

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.58 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -148.4843750, 134.0473938, -284.0219727, 285.8250732
1: -117.3068619, 129.7829132, -116.2964935, 126.5755463, -243.8824158, 246.0794067
2: -169.9014282, 144.0322723, -168.2982788, 140.5842438, -310.4856567, 312.3305054
3: -70.6789017, 171.0912170, -68.7259674, 169.2932587, -239.9721680, 239.8171692
4: -189.4996490, 143.3141022, -187.5333405, 139.8242035, -329.3237915, 330.8474426

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -93.5899734, 92.8750916, -242.1590271, 228.2793274
1: -116.9905624, 127.1221771, -73.1107025, 87.0040359, -203.9945984, 200.2328796
2: -169.2326202, 141.1778259, -105.9443512, 97.7353897, -266.9680176, 247.1221313
3: -69.0598907, 170.1712341, -47.2811356, 109.5639648, -178.6238556, 217.4523621
4: -188.5170746, 140.5437622, -118.2672424, 97.3865814, -285.9035950, 258.8109436

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -148.4843750, 134.0473938, -283.3312988, 283.1737366
1: -116.9905624, 127.1221771, -116.2964935, 126.5755463, -243.5661011, 243.4186707
2: -169.2326202, 141.1778259, -168.2982788, 140.5842438, -309.8168640, 309.4760437
3: -69.0598907, 170.1712341, -68.7259674, 169.2932587, -238.3531494, 238.8971558
4: -188.5170746, 140.5437622, -187.5333405, 139.8242035, -328.3412476, 328.0770874

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
time: 0.59 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
time: 0.59 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -93.5899734, 92.8750916, -307.5981750, 279.5631714
1: -168.6910553, 175.9269562, -73.1107025, 87.0040359, -255.6950989, 249.0376282
2: -243.5518188, 193.9766388, -105.9443512, 97.7353897, -341.2871704, 299.9209900
3: -95.6573029, 241.0182190, -47.2811356, 109.5639648, -204.5025635, 288.2993469
4: -271.0773621, 193.9729767, -118.2672424, 97.3865814, -368.4638977, 312.2401733

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -148.4843750, 134.0473938, -348.7705078, 334.4575806
1: -168.6910553, 175.9269562, -116.2964935, 126.5755463, -295.2665710, 292.2234192
2: -243.5518188, 193.9766388, -168.2982788, 140.5842438, -384.1360474, 362.2749023
3: -95.6573029, 241.0182190, -68.7259674, 169.2932587, -264.4364014, 309.7441406
4: -271.0773621, 193.9729767, -187.5333405, 139.8242035, -410.9014893, 381.5063171

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -117.7962265, 111.3588486, -201.4859467, 207.8058624
1: -70.3955536, 84.3041153, -92.3771362, 105.3934631, -175.7890167, 176.6812439
2: -102.0280762, 94.7010880, -133.9635315, 116.2873993, -218.3154755, 228.6646118
3: -45.7114983, 105.8294601, -56.3376808, 137.7161255, -183.4276276, 162.1671448
4: -113.9134064, 94.3664169, -149.6368866, 115.8507843, -229.7641907, 244.0032654

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6290496
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
time: 0.56 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -146.5204315, 134.4402618, -117.7962265, 111.3588486, -257.8792725, 252.2364502
1: -114.5982819, 127.0411911, -92.3771362, 105.3934631, -219.9917450, 219.4183044
2: -165.9999237, 140.9418030, -133.9635315, 116.2873993, -282.2872620, 274.9053345
3: -69.1322937, 167.3540039, -56.3376808, 137.7161255, -206.8483887, 223.6916504
4: -185.1622620, 140.2452240, -149.6368866, 115.8507843, -301.0130615, 289.8821106

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.57 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -119.8824997, 108.7030869, -146.5121613, 135.0994873, -254.9819946, 255.2152405
1: -93.7861786, 101.6748047, -114.6307831, 127.5979538, -221.3841248, 216.3055878
2: -135.7100830, 113.5954590, -166.0353241, 141.6752014, -277.3852234, 279.6307678
3: -54.4305954, 137.3644867, -69.5864105, 167.3457031, -221.7762604, 206.9508667
4: -151.1734619, 114.1141815, -185.1463776, 140.9630737, -292.1365356, 299.2605591

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312977
time: 0.55 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312736
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -119.8824997, 108.7030869, -210.2807465, 183.0075378, -302.8900452, 318.9838257
1: -93.7861786, 101.6748047, -165.1631775, 173.1105804, -266.8967590, 266.8379517
2: -135.7100830, 113.5954590, -238.5625763, 190.9392853, -326.6493225, 352.1580200
3: -54.4305954, 137.3644867, -94.2064362, 236.2539673, -290.6845093, 230.9414825
4: -151.1734619, 114.1141815, -265.5162659, 190.8201752, -341.9936523, 379.6304321

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.59 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -145.5118103, 131.5657959, -117.7430115, 111.3235092, -256.8353271, 249.3088074
1: -114.0084686, 124.1955643, -92.3358231, 105.3603516, -219.3687439, 216.5313416
2: -164.9636383, 137.8805237, -133.9037476, 116.2510834, -281.2147217, 271.7842712
3: -67.3987885, 166.0986328, -56.3193588, 137.6603546, -205.0591278, 222.4179688
4: -183.7941284, 137.2277069, -149.5703735, 115.8125458, -299.6066895, 286.7980957

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6279076
time: 0.62 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -210.6589203, 182.7044373, -117.7430115, 111.3235092, -321.9824219, 300.4474487
1: -165.3976288, 172.8678131, -92.3358231, 105.3603516, -270.7579651, 265.2036133
2: -238.9599762, 190.5442200, -133.9037476, 116.2510834, -355.2110596, 324.4479675
3: -93.9248505, 236.6527405, -56.3193588, 137.6603546, -230.8038788, 292.9721069
4: -265.9977722, 190.5023956, -149.5703735, 115.8125458, -381.8103027, 340.0727234

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -182.1177826, 153.6374054, -146.5121613, 135.0994873, -317.2172852, 300.1495667
1: -143.0064392, 144.6536407, -114.6307831, 127.5979538, -270.6044006, 259.2844238
2: -206.5487366, 159.9633484, -166.0353241, 141.6752014, -348.2238159, 325.9986572
3: -78.3319092, 204.7839813, -69.5864105, 167.3457031, -245.6776123, 274.3703613
4: -229.7001953, 160.7719421, -185.1463776, 140.9630737, -370.6632385, 345.9183044

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301557
time: 0.64 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6309206
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -182.1177826, 153.6374054, -210.2807465, 183.0075378, -365.1253052, 363.9181519
1: -143.0064392, 144.6536407, -165.1631775, 173.1105804, -316.1170044, 309.8168030
2: -206.5487366, 159.9633484, -238.5625763, 190.9392853, -397.4879761, 398.5259399
3: -78.3319092, 204.7839813, -94.2064362, 236.2539673, -314.5858765, 298.5309143
4: -229.7001953, 160.7719421, -265.5162659, 190.8201752, -420.5203552, 426.2882080

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
time: 0.66 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6309206
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.21 seconds
IS_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.8217075, upper bound: 187.6291626
IS_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.8241248, upper bound: 187.6312736
IS_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.8217075, upper bound: 187.6291626
IS_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.8241248, upper bound: 187.6312736
IS_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
IS_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6290496
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6323670
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312977
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312736
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6279076
IS_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6278615
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301557
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6309206
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6309206

## BFS IS instance: IS_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -90.1271210, 90.0096359, -130.9906311, 144.9062347
1: -32.1188278, 51.3675804, -70.3955536, 84.3041153, -116.4229126, 121.7631302
2: -47.0364113, 57.8066292, -102.0280762, 94.7010880, -141.7375031, 159.8347015
3: -27.2729225, 55.0080185, -45.7114983, 105.8294601, -133.1023712, 100.7195053
4: -53.0424118, 57.1621284, -113.9134064, 94.3664169, -147.4088287, 171.0755005

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8195608, upper bound: 187.7732911
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8216798, upper bound: 187.8241248
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -93.5899734, 92.8750916, -182.3230438, 183.4718781
1: -69.8412399, 84.1715393, -73.1107025, 87.0040359, -156.8452454, 157.2822418
2: -101.2739258, 94.6007919, -105.9443512, 97.7353897, -199.0093079, 200.5451355
3: -45.8811989, 105.0105896, -47.2811356, 109.5639648, -155.4451599, 152.2917175
4: -113.0568390, 94.2403870, -118.2672424, 97.3865814, -210.4434204, 212.5076294

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446681, upper bound: 187.7838627
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8209597
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -90.1271210, 90.0096359, -180.0393219, 185.8730469
1: -70.5223999, 90.7102203, -70.3955536, 84.3041153, -154.8264923, 161.1057434
2: -102.4902802, 100.3906708, -102.0280762, 94.7010880, -197.1913757, 202.4187317
3: -48.7527466, 108.2179565, -45.7114983, 105.8294601, -154.5822144, 153.9294586
4: -114.9058533, 99.2772675, -113.9134064, 94.3664169, -209.2722778, 213.1906738

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4333480
time: 0.72 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6291626
time: 0.64 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -93.5899734, 92.8750916, -238.9216614, 228.3237305
1: -114.2232513, 127.2852173, -73.1107025, 87.0040359, -201.2272644, 200.3959198
2: -165.4884644, 141.3374176, -105.9443512, 97.7353897, -263.2238464, 247.2817535
3: -69.3965378, 166.8359833, -47.2811356, 109.5639648, -178.9605103, 214.1171112
4: -184.5755615, 140.5543518, -118.2672424, 97.3865814, -281.9620667, 258.8214722

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6247483
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6312531
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -88.6077271, 92.8103714, -182.9374847, 178.6173706
1: -70.3955536, 84.3041153, -69.5185242, 87.8645935, -158.2601166, 153.8226318
2: -102.0280762, 94.7010880, -100.9003143, 97.3328018, -199.3608551, 195.6014099
3: -45.7114983, 105.8294601, -47.0367928, 106.5991135, -152.3106079, 152.8662567
4: -113.9134064, 94.3664169, -113.0341187, 96.2283783, -210.1417236, 207.4005280

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5275814, upper bound: 187.7334832
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5249229, upper bound: 187.7697195
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8122145
time: 0.70 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -144.2667847, 131.2299805, -224.8199463, 237.1418610
1: -73.1107025, 87.0040359, -112.9824219, 123.8839874, -196.9946899, 199.9864502
2: -105.9443512, 97.7353897, -163.5588074, 137.6753693, -243.6196899, 261.2941895
3: -47.2811356, 109.5639648, -67.3619995, 164.7313690, -212.0124969, 176.9259644
4: -118.2672424, 97.3865814, -182.2491913, 136.8414307, -255.1086426, 279.6357727

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -144.9881134, 131.1335144, -221.1632080, 240.7340240
1: -70.5223999, 90.7102203, -113.5550079, 123.8287811, -194.3511200, 204.2652130
2: -102.4902802, 100.3906708, -164.3510590, 137.4807434, -239.9710236, 264.7417297
3: -48.7527466, 108.2179565, -67.1747513, 165.5218964, -214.2746429, 175.3927002
4: -114.9058533, 99.2772675, -183.1476288, 136.7434082, -251.6492462, 282.4248962

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4332142
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6291800
time: 0.66 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -148.4843750, 134.0473938, -280.0939636, 283.2181396
1: -114.2232513, 127.2852173, -116.2964935, 126.5755463, -240.7987823, 243.5817108
2: -165.4884644, 141.3374176, -168.2982788, 140.5842438, -306.0726929, 309.6356812
3: -69.3965378, 166.8359833, -68.7259674, 169.2932587, -238.6897888, 235.5618896
4: -184.5755615, 140.5543518, -187.5333405, 139.8242035, -324.3997192, 328.0877075

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4213064, upper bound: 187.6256369
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301412, upper bound: 187.6312657
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -90.1271210, 90.0096359, -178.6173706, 182.9375000
1: -69.5185242, 87.8645935, -70.3955536, 84.3041153, -153.8226318, 158.2601013
2: -100.9003143, 97.3328018, -102.0280762, 94.7010880, -195.6014099, 199.3608551
3: -47.0367928, 106.5991135, -45.7114983, 105.8294601, -152.8662567, 152.3106079
4: -113.0341187, 96.2283783, -113.9134064, 94.3664169, -207.4005127, 210.1417389

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7334832, upper bound: 187.5275814
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7697195, upper bound: 187.5249229
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8122145, upper bound: 187.5295517
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -93.5899734, 92.8750916, -237.6843567, 225.2490234
1: -113.4549103, 124.2506790, -73.1107025, 87.0040359, -200.4589539, 197.3613892
2: -164.1941833, 138.0722961, -105.9443512, 97.7353897, -261.9295654, 244.0166168
3: -67.5839310, 165.3264618, -47.2811356, 109.5639648, -177.1478882, 212.6076050
4: -182.9160309, 137.3219604, -118.2672424, 97.3865814, -280.3026123, 255.5891571

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4446807, upper bound: 187.6238230
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8209721, upper bound: 187.6301112
time: 0.64 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -144.9881134, 131.1335144, -219.7412415, 237.7984772
1: -69.5185242, 87.8645935, -113.5550079, 123.8287811, -193.3472748, 201.4195709
2: -100.9003143, 97.3328018, -164.3510590, 137.4807434, -238.3810577, 261.6838684
3: -47.0367928, 106.5991135, -67.1747513, 165.5218964, -212.5586700, 173.7738647
4: -113.0341187, 96.2283783, -183.1476288, 136.7434082, -249.7775269, 279.3759766

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259274, upper bound: 187.3663697
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252269, upper bound: 187.5068514
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -148.4843750, 134.0473938, -278.8566589, 280.1434326
1: -113.4549103, 124.2506790, -116.2964935, 126.5755463, -240.0304565, 240.5471802
2: -164.1941833, 138.0722961, -168.2982788, 140.5842438, -304.7784424, 306.3705750
3: -67.5839310, 165.3264618, -68.7259674, 169.2932587, -236.8771973, 234.0524139
4: -182.9160309, 137.3219604, -187.5333405, 139.8242035, -322.7402344, 324.8552246

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -90.1271210, 90.0096359, -233.9671021, 226.8527527
1: -112.9016113, 130.6958466, -70.3955536, 84.3041153, -197.2056885, 199.3443451
2: -163.7802124, 143.6897125, -102.0280762, 94.7010880, -258.4812927, 243.7314453
3: -70.0736237, 166.9585724, -45.7114983, 105.8294601, -173.9596710, 212.6700745
4: -182.9930725, 142.4557343, -113.9134064, 94.3664169, -277.3594666, 255.4165039

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4457535, upper bound: 187.5322084
time: 0.57 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8220734, upper bound: 187.6288834
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -210.2340088, 182.9533081, -93.5899734, 92.8750916, -303.1090698, 276.5432739
1: -165.1261597, 173.0587158, -73.1107025, 87.0040359, -252.1301880, 246.1694183
2: -238.5093231, 190.8821259, -105.9443512, 97.7353897, -336.2447205, 296.8264771
3: -94.1762238, 236.2008667, -47.2811356, 109.5639648, -202.9751282, 283.4819946
4: -265.4568787, 190.7641907, -118.2672424, 97.3865814, -362.8434143, 309.0314026

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4530635, upper bound: 187.6244627
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8293550, upper bound: 187.6308760
time: 0.72 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -144.9881134, 131.1335144, -275.0910034, 282.1108704
1: -112.9016113, 130.6958466, -113.5550079, 123.8287811, -236.7303467, 242.8147278
2: -163.7802124, 143.6897125, -164.3510590, 137.4807434, -301.2609558, 306.4789429
3: -70.0736237, 166.9585724, -67.1747513, 165.5218964, -233.8621521, 234.1333313
4: -182.9930725, 142.4557343, -183.1476288, 136.7434082, -319.7364807, 325.1866455

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4223904, upper bound: 187.5429743
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6288834
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -148.4843750, 134.0473938, -344.3281250, 331.4919128
1: -165.1631775, 173.1105804, -116.2964935, 126.5755463, -291.7387085, 289.4070740
2: -238.5625763, 190.9392853, -168.2982788, 140.5842438, -379.1468201, 359.2375488
3: -94.2064362, 236.2539673, -68.7259674, 169.2932587, -262.9326477, 304.9798584
4: -265.5162659, 190.8201752, -187.5333405, 139.8242035, -405.3404541, 378.3535156

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6315027, upper bound: 187.4829529
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308760
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -90.0296860, 95.7459259, -185.8730469, 180.0393219
1: -70.3955536, 84.3041153, -70.5223999, 90.7102203, -161.1057587, 154.8264771
2: -102.0280762, 94.7010880, -102.4902802, 100.3906708, -202.4187317, 197.1913757
3: -45.7114983, 105.8294601, -48.7527466, 108.2179565, -153.9294434, 154.5821838
4: -113.9134064, 94.3664169, -114.9058533, 99.2772675, -213.1906738, 209.2722778

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4226656, upper bound: 187.8198218
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289039, upper bound: 187.8218635
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -143.9209900, 137.5663147, -226.8291016, 233.9306183
1: -70.3955536, 84.3041153, -112.8730698, 130.6749878, -199.3232880, 197.1771851
2: -102.0280762, 94.7010880, -163.7390137, 143.6699066, -243.7113495, 258.4400940
3: -45.7114983, 105.8294601, -70.0665512, 166.9166107, -212.6281128, 173.9524994
4: -113.9134064, 94.3664169, -182.9464722, 142.4316864, -255.3923950, 277.3128967

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4226656, upper bound: 187.8218585
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289039, upper bound: 187.8252147
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.5204315, 134.4402618, -90.0296860, 95.7459259, -242.2663574, 224.4699402
1: -114.5982819, 127.0411911, -70.5223999, 90.7102203, -205.3085022, 197.5635681
2: -165.9999237, 140.9418030, -102.4902802, 100.3906708, -266.3905640, 243.4320679
3: -69.1322937, 167.3540039, -48.7527466, 108.2179565, -177.3502197, 216.1067200
4: -185.1622620, 140.2452240, -114.9058533, 99.2772675, -284.4395142, 255.1510620

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3740021, upper bound: 187.6270404
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260053, upper bound: 187.6263399
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.5204315, 134.4402618, -143.9209900, 137.5663147, -283.5631104, 278.3612061
1: -114.5982819, 127.0411911, -112.8730698, 130.6749878, -243.8008423, 239.9142609
2: -165.9999237, 140.9418030, -163.7390137, 143.6699066, -308.0452576, 304.6808167
3: -69.1322937, 167.3540039, -70.0665512, 166.9166107, -236.0488586, 235.6338196
4: -185.1622620, 140.2452240, -182.9464722, 142.4316864, -327.1165161, 323.1917114

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
time: 0.76 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6280304
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -146.5121613, 135.0994873, -228.6894531, 239.3872528
1: -73.1107025, 87.0040359, -114.6307831, 127.5979538, -200.7086487, 201.6348267
2: -105.9443512, 97.7353897, -166.0353241, 141.6752014, -247.6195221, 263.7707214
3: -47.2811356, 109.5639648, -69.5864105, 167.3457031, -214.6268311, 179.1503601
4: -118.2672424, 97.3865814, -185.1463776, 140.9630737, -259.2302551, 282.5328979

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4352136
time: 0.61 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6291444
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -146.5121613, 135.0994873, -285.0740967, 283.8528442
1: -117.3068619, 129.7829132, -114.6307831, 127.5979538, -244.9048157, 244.4136963
2: -169.9014282, 144.0322723, -166.0353241, 141.6752014, -311.5765991, 310.0675354
3: -70.6789017, 171.0912170, -69.5864105, 167.3457031, -238.0245972, 240.6776123
4: -189.4996490, 143.3141022, -185.1463776, 140.9630737, -330.4627075, 328.4604797

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6291626
time: 0.61 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6295361
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -210.2340088, 182.9533081, -276.5432434, 303.1090698
1: -73.1107025, 87.0040359, -165.1261597, 173.0587158, -246.1694183, 252.1301880
2: -105.9443512, 97.7353897, -238.5093231, 190.8821259, -296.8264771, 336.2446899
3: -47.2811356, 109.5639648, -94.1762238, 236.2008667, -283.4819946, 202.9750977
4: -118.2672424, 97.3865814, -265.4568787, 190.7641907, -309.0313721, 362.8434448

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4212764, upper bound: 187.6326386
time: 0.78 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301112, upper bound: 187.6312657
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -210.2807465, 183.0075378, -332.9821167, 347.6214600
1: -117.3068619, 129.7829132, -165.1631775, 173.1105804, -290.4174500, 294.9461060
2: -169.9014282, 144.0322723, -238.5625763, 190.9392853, -360.8406982, 382.5947876
3: -70.6789017, 171.0912170, -94.2064362, 236.2539673, -306.9328003, 264.6764526
4: -189.4996490, 143.3141022, -265.5162659, 190.8201752, -380.3198242, 408.8303833

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4212764, upper bound: 187.6256369
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301112, upper bound: 187.6312657
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -145.5118103, 131.5657959, -90.0296860, 95.7459259, -241.2577362, 221.5954590
1: -114.0084686, 124.1955643, -70.5223999, 90.7102203, -204.7186432, 194.7179413
2: -164.9636383, 137.8805237, -102.4902802, 100.3906708, -265.3542480, 240.3708038
3: -67.3987885, 166.0986328, -48.7527466, 108.2179565, -175.6167297, 214.8513489
4: -183.7941284, 137.2277069, -114.9058533, 99.2772675, -283.0714111, 252.1335602

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225319, upper bound: 187.6258287
time: 0.59 seconds

## Relational analysis of IS_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6278704
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -145.5118103, 131.5657959, -143.9209900, 137.5663147, -282.6158752, 275.4867249
1: -114.0084686, 124.1955643, -112.8730698, 130.6749878, -243.2540588, 237.0686188
2: -164.9636383, 137.8805237, -163.7390137, 143.6699066, -307.0756226, 301.6195374
3: -67.3987885, 166.0986328, -70.0665512, 166.9166107, -234.3153992, 234.4335175
4: -183.7941284, 137.2277069, -182.9464722, 142.4316864, -325.8156738, 320.1741943

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225319, upper bound: 187.6258287
time: 0.59 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6278704
time: 0.61 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -210.6589203, 182.7044373, -90.0296860, 95.7459259, -306.4048462, 272.7341309
1: -165.3976288, 172.8678131, -70.5223999, 90.7102203, -256.1078186, 243.3901978
2: -238.9599762, 190.5442200, -102.4902802, 100.3906708, -339.3506470, 293.0344849
3: -93.9248505, 236.6527405, -48.7527466, 108.2179565, -201.1913910, 285.4054871
4: -265.9977722, 190.5023956, -114.9058533, 99.2772675, -365.2750244, 305.4081421

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
time: 0.70 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6278615
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -210.6589203, 182.7044373, -143.9209900, 137.5663147, -348.0799255, 326.6253662
1: -165.3976288, 172.8678131, -112.8730698, 130.6749878, -294.8795776, 285.7408752
2: -238.9599762, 190.5442200, -163.7390137, 143.6699066, -381.3220520, 354.2832336
3: -93.9248505, 236.6527405, -70.0665512, 166.9166107, -260.0750732, 305.0944214
4: -265.9977722, 190.5023956, -182.9464722, 142.4316864, -408.4028625, 373.4488525

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
time: 0.77 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6278615
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -146.5121613, 135.0994873, -284.3834229, 281.2014160
1: -116.9905624, 127.1221771, -114.6307831, 127.5979538, -244.5885010, 241.7529297
2: -169.2326202, 141.1778259, -166.0353241, 141.6752014, -310.9077759, 307.2131042
3: -69.0598907, 170.1712341, -69.5864105, 167.3457031, -236.4055939, 239.7575989
4: -188.5170746, 140.5437622, -185.1463776, 140.9630737, -329.4801331, 325.6901245

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256369, upper bound: 187.4213064
time: 0.61 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312657, upper bound: 187.6301412
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -146.5121613, 135.0994873, -349.8226013, 332.4853516
1: -168.6910553, 175.9269562, -114.6307831, 127.5979538, -296.2889404, 290.5576782
2: -243.5518188, 193.9766388, -166.0353241, 141.6752014, -385.2269592, 360.0119629
3: -95.6573029, 241.0182190, -69.5864105, 167.3457031, -262.1953430, 310.6046143
4: -271.0773621, 193.9729767, -185.1463776, 140.9630737, -412.0404053, 379.1193237

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6289272
time: 0.62 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6292015
time: 0.60 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -210.2807465, 183.0075378, -332.2914429, 344.9700623
1: -116.9905624, 127.1221771, -165.1631775, 173.1105804, -290.1011353, 292.2853088
2: -169.2326202, 141.1778259, -238.5625763, 190.9392853, -360.1719055, 379.7403564
3: -69.0598907, 170.1712341, -94.2064362, 236.2539673, -305.3138123, 263.8127747
4: -188.5170746, 140.5437622, -265.5162659, 190.8201752, -379.3372498, 406.0599976

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_A1_B1

### Relational analysis result of IS_B2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4190813, upper bound: 187.4076027
time: 0.69 seconds

## Relational analysis of IS_B2_A2_B2_B2_A1_B2

### Relational analysis result of IS_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6301111
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -210.2807465, 183.0075378, -397.7306213, 396.2539673
1: -168.6910553, 175.9269562, -165.1631775, 173.1105804, -341.8016052, 341.0900574
2: -243.5518188, 193.9766388, -238.5625763, 190.9392853, -434.4910889, 432.5391846
3: -95.6573029, 241.0182190, -94.2064362, 236.2539673, -331.2742920, 334.7620850
4: -271.0773621, 193.9729767, -265.5162659, 190.8201752, -461.8975220, 459.4892273

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_B2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4190815, upper bound: 187.6212472
time: 0.60 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6309060
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.23 seconds
IS_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8195608, upper bound: 187.7732911
IS_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8216798, upper bound: 187.8241248
IS_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4446681, upper bound: 187.7838627
IS_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8209595, upper bound: 187.8209597
IS_B1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8198218, upper bound: 187.4333480
IS_B1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8218635, upper bound: 187.6291626
IS_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4446981, upper bound: 187.6247483
IS_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8209895, upper bound: 187.6312531
IS_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5249229, upper bound: 187.7697195
IS_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5295517, upper bound: 187.8122145
IS_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6238230, upper bound: 187.4446807
IS_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301112, upper bound: 187.8209721
IS_B1_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6258287, upper bound: 187.4332142
IS_B1_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6278704, upper bound: 187.6291800
IS_B1_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4213064, upper bound: 187.6256369
IS_B1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301412, upper bound: 187.6312657
IS_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.7697195, upper bound: 187.5249229
IS_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8122145, upper bound: 187.5295517
IS_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4446807, upper bound: 187.6238230
IS_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8209721, upper bound: 187.6301112
IS_B1_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6259274, upper bound: 187.3663697
IS_B1_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6252269, upper bound: 187.5068514
IS_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
IS_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
IS_B1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4457535, upper bound: 187.5322084
IS_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8220734, upper bound: 187.6288834
IS_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4530635, upper bound: 187.6244627
IS_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.8293550, upper bound: 187.6308760
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4223904, upper bound: 187.5429743
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6288834
IS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6315027, upper bound: 187.4829529
IS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308760
IS_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4226656, upper bound: 187.8198218
IS_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6289039, upper bound: 187.8218635
IS_B2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4226656, upper bound: 187.8218585
IS_B2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6289039, upper bound: 187.8252147
IS_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.3740021, upper bound: 187.6270404
IS_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6260053, upper bound: 187.6263399
IS_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
IS_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6266799, upper bound: 187.6280304
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4352136
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6291444
IS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6291626
IS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6267453, upper bound: 187.6295361
IS_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4212764, upper bound: 187.6326386
IS_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301112, upper bound: 187.6312657
IS_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4212764, upper bound: 187.6256369
IS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301112, upper bound: 187.6312657
IS_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4225319, upper bound: 187.6258287
IS_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6278704
IS_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4225319, upper bound: 187.6258287
IS_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6278704
IS_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
IS_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6278615
IS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
IS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6278615
IS_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6256369, upper bound: 187.4213064
IS_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6312657, upper bound: 187.6301412
IS_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6289272
IS_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6292015
IS_B2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4190813, upper bound: 187.4076027
IS_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6301111
IS_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.4190815, upper bound: 187.6212472
IS_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 3, lower bound: -187.6301238, upper bound: 187.6309060

## BFS IS instance: IS_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -32.7674904, 48.4766617, -89.3119202, 89.4319458, -122.1994324, 137.7885742
1: -25.7796745, 45.4064064, -69.7544098, 83.7592239, -109.5388947, 115.1608124
2: -37.8850098, 51.2374420, -101.1050262, 94.0961609, -131.9811707, 152.3424683
3: -24.2111740, 46.3258553, -45.4068604, 104.9753876, -129.1865540, 91.7327042
4: -42.9179688, 50.6116943, -112.8916779, 93.7653809, -136.6833191, 163.5033722

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8136326, upper bound: 186.8731229
time: 0.64 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8135479, upper bound: 187.7704506
time: 0.77 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -40.8582802, 54.6809502, -90.1271210, 90.0096359, -130.8679199, 144.8080750
1: -32.0204277, 51.2760010, -70.3955536, 84.3041153, -116.3245392, 121.6715546
2: -46.8952942, 57.7043457, -102.0280762, 94.7010880, -141.5963745, 159.7324219
3: -27.2256889, 54.8681870, -45.7114983, 105.8294601, -133.0551453, 100.5796738
4: -52.8865433, 57.0566254, -113.9134064, 94.3664169, -147.2529602, 170.9699554

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8159557, upper bound: 187.3900516
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149491, upper bound: 187.8172813
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -88.5961609, 89.3543777, -78.3078232, 83.3920822, -171.9881897, 167.6622009
1: -69.1682358, 83.6756744, -61.0082550, 78.1903000, -147.3585358, 144.6839294
2: -100.3089142, 94.0714035, -88.5467072, 88.3264847, -188.6354065, 182.6180725
3: -45.6402168, 104.0936050, -42.9643326, 92.9597931, -138.6000061, 147.0579071
4: -111.9885635, 93.6896210, -99.0556564, 87.5924988, -199.5809937, 192.7452698

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.2425294, upper bound: 187.7161606
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4421850, upper bound: 187.7255157
time: 0.58 seconds

## BFS IS instance: IS_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -93.0833359, 92.5407257, -181.9887085, 182.9652405
1: -69.8412399, 84.1715393, -72.7047272, 86.7016907, -156.5429382, 156.8762207
2: -101.2739258, 94.6007919, -105.3617554, 97.4237671, -198.6976471, 199.9625549
3: -45.8811989, 105.0105896, -47.1332130, 109.0123520, -154.8935547, 152.1437988
4: -113.0568390, 94.2403870, -117.6394958, 97.0536499, -210.1104889, 211.8798828

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8184764, upper bound: 187.7321685
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_B1_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7298451, upper bound: 187.7298453
time: 0.73 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -75.1740952, 86.4655151, -89.3119202, 89.4319458, -164.6060486, 175.7774353
1: -59.0501709, 81.8904648, -69.7544098, 83.7592239, -142.8093719, 151.6448517
2: -86.0124969, 90.7257843, -101.1050262, 94.0961609, -180.1086578, 191.8308105
3: -43.9294243, 93.0511627, -45.4068604, 104.9753876, -148.8447876, 138.4580078
4: -96.5924683, 89.4440918, -112.8916779, 93.7653809, -190.3578186, 202.3357697

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8139011, upper bound: 186.8647431
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8138139, upper bound: 187.4298343
time: 0.66 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -89.8806458, 95.6513596, -90.1271210, 90.0096359, -179.8902740, 185.7784729
1: -70.4029846, 90.6203918, -70.3955536, 84.3041153, -154.7070923, 161.0158997
2: -102.3191757, 100.2926941, -102.0280762, 94.7010880, -197.0202637, 202.3207092
3: -48.7057953, 108.0552597, -45.7114983, 105.8294601, -154.5352478, 153.7667542
4: -114.7167358, 99.1772308, -113.9134064, 94.3664169, -209.0831604, 213.0906067

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7345315, upper bound: 187.6271923
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8161505, upper bound: 187.3743339
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B1_A2_A1_A2_A2

### Relational analysis result of IS_B1_A1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151150, upper bound: 187.6263037
time: 0.63 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -144.9912872, 134.1121521, -78.3078232, 83.3920822, -228.3833618, 212.4199829
1: -113.3895874, 126.7003326, -61.0082550, 78.1903000, -191.5798950, 187.7085876
2: -164.2908630, 140.7146149, -88.5467072, 88.3264847, -252.6173401, 229.2612915
3: -69.0959320, 165.7116699, -42.9643326, 92.9597931, -162.0557251, 208.6759796
4: -183.2507172, 139.9004822, -99.0556564, 87.5924988, -270.8432007, 238.9561462

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4445361, upper bound: 187.6242238
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3442679, upper bound: 187.6239750
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4399181, upper bound: 187.6213623
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -93.0833359, 92.5407257, -238.5872955, 227.8171082
1: -114.2232513, 127.2852173, -72.7047272, 86.7016907, -200.9249420, 199.9899292
2: -165.4884644, 141.3374176, -105.3617554, 97.4237671, -262.9122314, 246.6991577
3: -69.3965378, 166.8359833, -47.1332130, 109.0123520, -178.4088898, 213.9691772
4: -184.5755615, 140.5543518, -117.6394958, 97.0536499, -281.6292114, 258.1938477

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7321962, upper bound: 187.6287700
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6821948, upper bound: 187.6297634
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8201621, upper bound: 187.6304231
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -87.3030930, 87.7531204, -88.6077271, 92.8103714, -180.1134644, 176.3608398
1: -68.0981674, 82.2644958, -69.5185242, 87.8645935, -155.9627533, 151.7830200
2: -98.7406158, 92.4605713, -100.9003143, 97.3328018, -196.0734100, 193.3608704
3: -44.4350815, 102.5194550, -47.0367928, 106.5991135, -151.0341797, 149.5562439
4: -110.2698212, 91.9975433, -113.0341187, 96.2283783, -206.4981842, 205.0316620

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3628795, upper bound: 187.7645102
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5033612, upper bound: 187.7638097
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -89.1020126, 89.3148727, -88.6077271, 92.8103714, -181.9123230, 177.9226074
1: -69.5746841, 83.6795959, -69.5185242, 87.8645935, -157.4392548, 153.1981201
2: -100.8616104, 94.0344391, -100.9003143, 97.3328018, -198.1944122, 194.9347534
3: -45.4045029, 104.7132263, -47.0367928, 106.5991135, -152.0036163, 151.7500153
4: -112.6212540, 93.6422577, -113.0341187, 96.2283783, -208.8495941, 206.6763763

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5275814, upper bound: 187.7250532
time: 0.64 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663626, upper bound: 187.8075819
time: 0.63 seconds

## Relational analysis of IS_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068443, upper bound: 187.8068814
time: 0.77 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.3078232, 83.3920822, -143.2077789, 130.6004639, -208.9082794, 226.5998535
1: -61.0082550, 78.1903000, -112.1456070, 123.2922058, -184.3004608, 190.3359070
2: -88.5467072, 88.3264847, -162.3571167, 137.0434875, -225.5901489, 250.6835938
3: -42.9643326, 92.9597931, -67.0538864, 163.6034393, -206.5677643, 160.0136719
4: -99.0556564, 87.5924988, -180.9197388, 136.1773224, -235.2329712, 268.5122070

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5326621, upper bound: 187.2429417
time: 0.63 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5374616, upper bound: 187.4425973
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -93.0833359, 92.5407257, -144.2667847, 131.2299805, -224.3133240, 236.8075104
1: -72.7047272, 86.7016907, -112.9824219, 123.8839874, -196.5887146, 199.6841125
2: -105.3617554, 97.4237671, -163.5588074, 137.6753693, -243.0370941, 260.9825745
3: -47.1332130, 109.0123520, -67.3619995, 164.7313690, -211.8645782, 176.3743439
4: -117.6394958, 97.0536499, -182.2491913, 136.8414307, -254.4809265, 279.3028564

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276281, upper bound: 187.7321553
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5417912, upper bound: 187.7302574
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -75.1740952, 86.4655151, -144.1082916, 130.5227966, -205.6968994, 230.5737915
1: -59.0501709, 81.8904648, -112.8615570, 123.2553558, -182.3054962, 194.7520142
2: -86.0124969, 90.7257843, -163.3523560, 136.8441010, -222.8565979, 254.0781403
3: -43.9294243, 93.0511627, -66.8554077, 164.5908813, -208.5202942, 159.9065704
4: -96.5924683, 89.4440918, -182.0408783, 136.1013794, -232.6938171, 271.4849243

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A2_A1_A1_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6243825, upper bound: 186.8645395
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A1_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6242979, upper bound: 187.4296048
time: 0.70 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -89.8806458, 95.6513596, -144.9881134, 131.1335144, -221.0141602, 240.6394653
1: -70.4029846, 90.6203918, -113.5550079, 123.8287811, -194.2317657, 204.1753693
2: -102.3191757, 100.2926941, -164.3510590, 137.4807434, -239.7999115, 264.6437378
3: -48.7057953, 108.0552597, -67.1747513, 165.5218964, -214.2276917, 175.2300110
4: -114.7167358, 99.1772308, -183.1476288, 136.7434082, -251.4601288, 282.3248291

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269004, upper bound: 187.3743426
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2_A2_A1_A2_A2

### Relational analysis result of IS_B1_A1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258650, upper bound: 187.6263123
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -144.9912872, 134.1121521, -131.2125702, 123.7671967, -268.7584534, 265.3247070
1: -113.3895874, 126.7003326, -102.6090622, 116.8737335, -230.2633209, 229.3093872
2: -164.2908630, 140.7146149, -148.5732117, 130.2702484, -294.5610962, 289.2878418
3: -69.0959320, 165.7116699, -63.8431320, 150.6193237, -219.7152557, 229.5548096
4: -183.2507172, 139.9004822, -165.7654877, 129.0743713, -312.3250732, 305.6659546

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3305720, upper bound: 187.6248284
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4181828, upper bound: 187.6220154
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -147.9475708, 133.7212982, -279.7678833, 282.6813354
1: -114.2232513, 127.2852173, -115.8660278, 126.2766418, -240.4998932, 243.1512451
2: -165.4884644, 141.3374176, -167.6802826, 140.2731628, -305.7616272, 309.0177002
3: -69.3965378, 166.8359833, -68.5721893, 168.7137146, -238.1102600, 235.4081573
4: -184.5755615, 140.5543518, -186.8633575, 139.4981537, -324.0736694, 327.4177246

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B2_A2_A2_B2_B1

### Relational analysis result of IS_B1_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295611, upper bound: 187.6280304
time: 0.68 seconds

## Relational analysis of IS_B1_A1_B2_A2_A2_B2_B2

### Relational analysis result of IS_B1_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295612, upper bound: 187.6287787
time: 0.72 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -87.3030930, 87.7531204, -176.3608398, 180.1134644
1: -69.5185242, 87.8645935, -68.0981674, 82.2644958, -151.7830048, 155.9627533
2: -100.9003143, 97.3328018, -98.7406158, 92.4605713, -193.3608704, 196.0733948
3: -47.0367928, 106.5991135, -44.4350815, 102.5194550, -149.5562439, 151.0341949
4: -113.0341187, 96.2283783, -110.2698212, 91.9975433, -205.0316620, 206.4981995

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7645102, upper bound: 187.3628795
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7638097, upper bound: 187.5033612
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -89.1020126, 89.3148727, -177.9226074, 181.9123230
1: -69.5185242, 87.8645935, -69.5746841, 83.6795959, -153.1981201, 157.4392548
2: -100.9003143, 97.3328018, -100.8616104, 94.0344391, -194.9347534, 198.1944122
3: -47.0367928, 106.5991135, -45.4045029, 104.7132263, -151.7500153, 152.0036011
4: -113.0341187, 96.2283783, -112.6212540, 93.6422577, -206.6763763, 208.8496246

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7250532, upper bound: 187.5275814
time: 0.61 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8075819, upper bound: 187.3663626
time: 0.58 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8068814, upper bound: 187.5068443
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -143.6822815, 130.9752960, -78.3078232, 83.3920822, -227.0743408, 209.2831116
1: -112.5612793, 123.6121674, -61.0082550, 78.1903000, -190.7515869, 184.6204224
2: -162.9146423, 137.3910522, -88.5467072, 88.3264847, -251.2411041, 225.9377136
3: -67.2502441, 164.1228790, -42.9643326, 92.9597931, -160.2100372, 207.0871735
4: -181.5015106, 136.5980835, -99.0556564, 87.5924988, -269.0939941, 235.6537476

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.2429417, upper bound: 187.5326621
time: 0.71 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4425973, upper bound: 187.5374616
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -93.0833359, 92.5407257, -237.3499908, 224.7423706
1: -113.4549103, 124.2506790, -72.7047272, 86.7016907, -200.1566010, 196.9554138
2: -164.1941833, 138.0722961, -105.3617554, 97.4237671, -261.6179504, 243.4340515
3: -67.5839310, 165.3264618, -47.1332130, 109.0123520, -176.5962830, 212.4596710
4: -182.9160309, 137.3219604, -117.6394958, 97.0536499, -279.9696655, 254.9614563

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7321553, upper bound: 187.6276281
time: 0.60 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7302574, upper bound: 187.5417912
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -79.2864609, 88.0104218, -144.9881134, 131.1335144, -210.4199829, 232.9985199
1: -62.3454056, 83.3233795, -113.5550079, 123.8287811, -186.1741486, 196.8783722
2: -90.5679855, 92.4689178, -164.3510590, 137.4807434, -228.0487366, 256.8199768
3: -44.6927567, 97.4578094, -67.1747513, 165.5218964, -210.2146454, 164.6325684
4: -101.7387695, 91.1533737, -183.1476288, 136.7434082, -238.4821472, 274.3009338

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5047471, upper bound: 187.3635645
time: 0.57 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5047471, upper bound: 187.3663697
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -85.4159851, 90.9214935, -144.9881134, 131.1335144, -216.5494843, 235.9096069
1: -67.0133209, 86.0685425, -113.5550079, 123.8287811, -190.8421021, 199.6235504
2: -97.3060455, 95.3715820, -164.3510590, 137.4807434, -234.7867889, 259.7226257
3: -46.1031952, 103.3089905, -67.1747513, 165.5218964, -211.6250916, 170.4837341
4: -109.0886841, 94.2118530, -183.1476288, 136.7434082, -245.8320770, 277.3594055

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5040466, upper bound: 187.5040462
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5040466, upper bound: 187.5068514
time: 0.69 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -88.6077271, 92.8103714, -237.6196289, 220.2667694
1: -113.4549103, 124.2506790, -69.5185242, 87.8645935, -201.3195038, 193.7691956
2: -164.1941833, 138.0722961, -100.9003143, 97.3328018, -261.5269775, 238.9726105
3: -67.5839310, 165.3264618, -47.0367928, 106.5991135, -174.1830444, 212.3632507
4: -182.9160309, 137.3219604, -113.0341187, 96.2283783, -279.1444092, 250.3560791

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663783, upper bound: 187.6259274
time: 0.71 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068600, upper bound: 187.6252269
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -144.2667847, 131.2299805, -276.0392456, 275.9258423
1: -113.4549103, 124.2506790, -112.9824219, 123.8839874, -237.3388977, 237.2330933
2: -164.1941833, 138.0722961, -163.5588074, 137.6753693, -301.8695374, 301.6311035
3: -67.5839310, 165.3264618, -67.3619995, 164.7313690, -232.3153076, 232.6884613
4: -182.9160309, 137.3219604, -182.2491913, 136.8414307, -319.7574463, 319.5711060

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3663784, upper bound: 187.6270831
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068601, upper bound: 187.6259756
time: 0.60 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -89.6254959, 89.6727753, -233.6302490, 226.3312225
1: -112.9016113, 130.6958466, -69.9929199, 83.9996567, -196.9012604, 198.9374695
2: -163.7802124, 143.6897125, -101.4494705, 94.3850708, -258.1652832, 243.1539764
3: -70.0736237, 166.9585724, -45.5639267, 105.2798157, -173.4190216, 212.5224762
4: -182.9930725, 142.4557343, -113.2918549, 94.0299683, -277.0230103, 254.7894592

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8187137, upper bound: 187.4226656
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8220699, upper bound: 187.6288834
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -209.0707550, 182.2571411, -78.3078232, 83.3920822, -292.4628296, 260.5649719
1: -164.2199249, 172.4071960, -61.0082550, 78.1903000, -242.4102173, 233.4154510
2: -237.1934967, 190.1849213, -88.5467072, 88.3264847, -325.5199890, 278.7315979
3: -93.8339157, 234.9738007, -42.9643326, 92.9597931, -185.8665466, 277.9381409
4: -263.9979858, 190.0255127, -99.0556564, 87.5924988, -351.5904846, 289.0811462

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4526768, upper bound: 187.6239382
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4465258, upper bound: 187.5384040
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -210.2322845, 182.9512939, -93.0833359, 92.5407257, -302.7730103, 276.0346069
1: -165.1247864, 173.0568390, -72.7047272, 86.7016907, -251.8264771, 245.7615662
2: -238.5073242, 190.8800201, -105.3617554, 97.4237671, -335.9310608, 296.2417603
3: -94.1751175, 236.1989441, -47.1332130, 109.0123520, -202.4325562, 283.3321533
4: -265.4546814, 190.7621460, -117.6394958, 97.0536499, -362.5083008, 308.4016418

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7401595, upper bound: 187.6283930
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7341859, upper bound: 187.5427335
time: 0.86 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -142.9640198, 136.9809723, -127.7120361, 120.8518143, -263.8158264, 263.9223633
1: -112.1127090, 130.1181793, -99.8437500, 114.1122055, -226.2249146, 228.3862457
2: -162.6429138, 143.0717163, -144.6075134, 127.1671677, -289.8100891, 286.0080872
3: -69.7774734, 165.8832245, -62.2917862, 146.8193359, -214.7074432, 228.1749878
4: -181.7416077, 141.8115082, -161.3662262, 125.9870377, -307.7286377, 302.5970459

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4222415, upper bound: 187.5425418
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4204353, upper bound: 187.3847469
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4177371, upper bound: 187.5384362
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -144.4419403, 130.8016663, -274.7590942, 281.5367126
1: -112.9016113, 130.6958466, -113.1171951, 123.5252380, -236.4268188, 242.3685913
2: -163.7802124, 143.6897125, -163.7219391, 137.1648102, -300.9450073, 305.8506775
3: -70.0736237, 166.9585724, -67.0178909, 164.9322357, -233.2821503, 233.9764557
4: -182.9930725, 142.4557343, -182.4673767, 136.4104919, -319.4035645, 324.4938354

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278654, upper bound: 187.4225207
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312216, upper bound: 187.6288834
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -189.2039032, 170.2929993, -147.4202881, 133.4157104, -322.6195984, 317.7132263
1: -148.4229126, 161.1784210, -115.4561005, 125.9795303, -274.4024048, 276.5023499
2: -214.2863617, 178.2115936, -167.0916138, 139.9495087, -354.2358093, 345.1659851
3: -88.0816040, 213.3966370, -68.4211807, 168.1601868, -254.6332245, 281.8178101
4: -238.8560486, 177.4226837, -186.1973267, 139.1570892, -378.0130005, 363.6199951

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5192431, upper bound: 187.4796415
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5192431, upper bound: 187.4829518
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -209.4516602, 182.5380554, -148.4843750, 134.0473938, -343.4990234, 331.0224304
1: -164.4951782, 172.6763000, -116.2964935, 126.5755463, -291.0707397, 288.9727783
2: -237.6165009, 190.4843597, -168.2982788, 140.5842438, -378.2007446, 358.7825623
3: -93.9804077, 235.3852997, -68.7259674, 169.2932587, -262.6427307, 304.1112671
4: -264.4883728, 190.3352966, -187.5333405, 139.8242035, -404.3125610, 377.8686523

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276533
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6284155
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -89.3119202, 89.4319458, -75.1740952, 86.4655151, -175.7774353, 164.6060486
1: -69.7544098, 83.7592239, -59.0501709, 81.8904648, -151.6448364, 142.8093719
2: -101.1050262, 94.0961609, -86.0124969, 90.7257843, -191.8308105, 180.1086578
3: -45.4068604, 104.9753876, -43.9294243, 93.0511627, -138.4579926, 148.8448029
4: -112.8916779, 93.7653809, -96.5924683, 89.4440918, -202.3357697, 190.3578186

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -186.8647431, upper bound: 187.8139011
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4298343, upper bound: 187.8138139
time: 0.55 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -89.8806458, 95.6513596, -185.7784729, 179.8902588
1: -70.3955536, 84.3041153, -70.4029846, 90.6203918, -161.0158997, 154.7070923
2: -102.0280762, 94.7010880, -102.3191757, 100.2926941, -202.3207245, 197.0202484
3: -45.7114983, 105.8294601, -48.7057953, 108.0552597, -153.7667542, 154.5352478
4: -113.9134064, 94.3664169, -114.7167358, 99.1772308, -213.0906067, 209.0831604

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271923, upper bound: 187.7345315
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3743339, upper bound: 187.8161505
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263037, upper bound: 187.8151150
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -89.3119202, 89.4319458, -126.8857346, 127.0289917, -214.4331970, 216.3176880
1: -69.7544098, 83.7592239, -99.5447693, 120.7034454, -187.8314514, 183.3039856
2: -101.1050262, 94.0961609, -144.3884735, 132.6780396, -230.7034302, 238.4846344
3: -45.4068604, 104.9753876, -64.6005859, 148.9533081, -194.3601685, 167.2675476
4: -112.8916779, 93.7653809, -161.6483459, 131.3171997, -242.1422119, 255.4136963

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3996900, upper bound: 187.7345629
time: 0.57 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3891996, upper bound: 187.7942825
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4226656, upper bound: 187.8060612
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3940644, upper bound: 187.8131916
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -90.1271210, 90.0096359, -143.7625427, 137.4667053, -226.6902008, 233.7721863
1: -70.3955536, 84.3041153, -112.7459106, 130.5810547, -199.1880493, 197.0500183
2: -102.0280762, 94.7010880, -163.5565948, 143.5669098, -243.5667725, 258.2576599
3: -45.7114983, 105.8294601, -70.0163727, 166.7445679, -212.4560699, 173.8869476
4: -113.9134064, 94.3664169, -182.7444916, 142.3251953, -255.2456970, 277.1109009

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5322084, upper bound: 187.4457535
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288834, upper bound: 187.8220699
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -146.5204315, 134.4402618, -80.3507690, 90.7024994, -237.2229309, 214.7910309
1: -114.5982819, 127.0411911, -63.1035271, 85.9116058, -200.5098877, 190.1446991
2: -165.9999237, 140.9418030, -91.7921677, 95.2717209, -261.2716370, 232.7339478
3: -69.1322937, 167.3540039, -46.2828140, 98.7244492, -167.8567047, 213.5657349
4: -185.1622620, 140.2452240, -103.1749496, 93.9415741, -279.1037903, 243.4201050

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -186.8630234, upper bound: 187.6254956
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3741240, upper bound: 187.6278187
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -146.5204315, 134.4402618, -87.0424194, 93.9333038, -240.4537201, 221.4826813
1: -114.5982819, 127.0411911, -68.1751099, 88.9729767, -203.5712585, 195.2162781
2: -165.9999237, 140.9418030, -99.1208801, 98.4945221, -264.4944153, 240.0626526
3: -69.1322937, 167.3540039, -47.8467789, 105.1401672, -174.2724152, 215.1929016
4: -185.1622620, 140.2452240, -111.2113953, 97.3413239, -282.5035095, 251.4566193

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4299513, upper bound: 187.6254109
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263031, upper bound: 187.6268122
time: 0.62 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -143.9209900, 137.5663147, -226.6192474, 239.6669159
1: -70.5223999, 90.7102203, -112.8730698, 130.6749878, -199.4515228, 203.5832825
2: -102.4902802, 100.3906708, -163.7390137, 143.6699066, -244.2500458, 264.1296997
3: -48.7527466, 108.2179565, -70.0665512, 166.9166107, -215.6693420, 176.3728790
4: -114.9058533, 99.2772675, -182.9464722, 142.4316864, -256.4761047, 282.2237549

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249471, upper bound: 187.3719446
time: 0.93 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6238118, upper bound: 187.6234985
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -143.9209900, 137.5663147, -282.8721924, 278.6547241
1: -114.2232513, 127.2852173, -112.8730698, 130.6749878, -243.2799683, 240.1582947
2: -165.4884644, 141.3374176, -163.7390137, 143.6699066, -307.2042847, 305.0764160
3: -69.3965378, 166.8359833, -70.0665512, 166.9166107, -236.3131409, 234.8898163
4: -184.5755615, 140.5543518, -182.9464722, 142.4316864, -326.2274780, 323.5008240

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8581898, upper bound: 187.6713976
time: 0.66 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.51 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -187.8581898, upper bound: 187.6713976
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -149.6440735, 126.7424088, -119.9573212, 108.7468872, -258.3909607, 246.6997375
1: -117.3338928, 118.4335785, -93.8448410, 101.7164536, -219.0503540, 212.2784119
2: -169.7016296, 131.6250763, -135.7941437, 113.6401825, -283.3417969, 267.4191895
3: -63.3496017, 169.2627869, -54.4526558, 137.4407501, -200.7903442, 223.7154388
4: -188.6523895, 133.4867859, -151.2667084, 114.1601410, -302.8124695, 284.7534180

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
time: 0.71 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
time: 0.61 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -148.4383087, 125.9597473, -182.1602936, 153.6605072, -302.0988159, 308.1200562
1: -116.3920288, 117.7056580, -143.0398865, 144.6755829, -261.0675964, 260.7455444
2: -168.3444672, 130.8322754, -206.5966339, 159.9870605, -328.3315430, 337.4288940
3: -62.9577904, 168.0063171, -78.3434219, 204.8266907, -267.7844849, 246.3497314
4: -187.1445160, 132.6473389, -229.7526093, 160.7959442, -347.9404602, 362.3999023

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6367566
time: 0.69 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6375295
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.17 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6367566
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6375295

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311384, upper bound: 187.6299901
time: 0.65 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6307839
time: 0.71 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -119.9573212, 108.7468872, -290.9071655, 273.6178284
1: -143.0398865, 144.6755829, -93.8448410, 101.7164536, -244.7563477, 238.5203857
2: -206.5966339, 159.9870605, -135.7941437, 113.6401825, -320.2368164, 295.7811584
3: -78.3434219, 204.8266907, -54.4526558, 137.4407501, -215.7841339, 259.2793579
4: -229.7526093, 160.7959442, -151.2667084, 114.1601410, -343.9127502, 312.0626221

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311384, upper bound: 187.6299901
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6307839
time: 0.75 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -118.5167923, 107.8890457, -182.1602936, 153.6605072, -272.1773071, 290.0493469
1: -92.7873993, 100.8808746, -143.0398865, 144.6755829, -237.4629364, 243.9207611
2: -134.2325439, 112.7549667, -206.5966339, 159.9870605, -294.2196045, 319.3515930
3: -54.0469742, 136.1377563, -78.3434219, 204.8266907, -258.8736572, 214.4811554
4: -149.4989014, 113.2376175, -229.7526093, 160.7959442, -310.2948608, 342.9902344

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6300054
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6300054
time: 0.86 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -182.9971924, 156.2641907, -181.5190277, 153.3090973, -336.3062744, 337.7831726
1: -143.6196594, 147.2672119, -142.5338440, 144.3416901, -287.9613647, 289.8010254
2: -207.5287476, 162.7586975, -205.8718872, 159.6268616, -367.1555786, 368.6305847
3: -79.8542099, 206.0446167, -78.1683502, 204.1779022, -284.0321045, 284.2129517
4: -230.9113464, 163.4759369, -228.9604187, 160.4309540, -391.3422852, 392.4363403

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6375295
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6307839
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.33 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6311384, upper bound: 187.6299901
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6307839
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6311384, upper bound: 187.6299901
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6307839
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6300054
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6300054
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6375295
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.33
Output dim: 3, lower bound: -187.6307839, upper bound: 187.6307839

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -119.9573212, 108.7468872, -202.3368530, 212.8324127
1: -73.1107025, 87.0040359, -93.8448410, 101.7164536, -174.8271484, 180.8488770
2: -105.9443512, 97.7353897, -135.7941437, 113.6401825, -219.5845337, 233.5295410
3: -47.2811356, 109.5639648, -54.4526558, 137.4407501, -184.7218933, 164.0166168
4: -118.2672424, 97.3865814, -151.2667084, 114.1601410, -232.4273529, 248.6532745

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311390, upper bound: 187.6311390
time: 0.65 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311390, upper bound: 187.6311568
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -119.0552597, 108.2171707, -258.1917725, 256.3959656
1: -117.3068619, 129.7829132, -93.1325226, 101.2112274, -218.5180969, 222.9154053
2: -169.9014282, 144.0322723, -134.7739410, 113.0966644, -282.9980774, 278.8061523
3: -70.6789017, 171.0912170, -54.1846466, 136.5119934, -207.1908875, 225.2758636
4: -189.4996490, 143.3141022, -150.1368103, 113.6082687, -303.1079102, 293.4509277

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6311390
time: 0.64 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6311568
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -119.9573212, 108.7468872, -258.0307922, 254.6466827
1: -116.9905624, 127.1221771, -93.8448410, 101.7164536, -218.7070160, 220.9670105
2: -169.2326202, 141.1778259, -135.7941437, 113.6401825, -282.8728027, 276.9718933
3: -69.0598907, 170.1712341, -54.4526558, 137.4407501, -206.5006256, 224.6238861
4: -188.5170746, 140.5437622, -151.2667084, 114.1601410, -302.6772156, 291.8104248

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311433, upper bound: 187.6299901
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311433, upper bound: 187.6299901
time: 0.74 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -119.0552597, 108.2171707, -322.9401245, 305.0284729
1: -168.6910553, 175.9269562, -93.1325226, 101.2112274, -269.9022827, 269.0594482
2: -243.5518188, 193.9766388, -134.7739410, 113.0966644, -356.6484680, 328.7505493
3: -95.6573029, 241.0182190, -54.1846466, 136.5119934, -231.5868530, 295.2027893
4: -271.0773621, 193.9729767, -150.1368103, 113.6082687, -384.6856079, 344.1097412

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6379026, upper bound: 187.6307642
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6379026, upper bound: 187.6307839
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -118.5167923, 107.8890457, -149.2839355, 134.6893616, -253.2061005, 257.1729431
1: -92.7873993, 100.8808746, -116.9905624, 127.1221771, -219.9095764, 217.8714294
2: -134.2325439, 112.7549667, -169.2326202, 141.1778259, -275.4103394, 281.9875793
3: -54.0469742, 136.1377563, -69.0598907, 170.1712341, -224.2182007, 205.1976471
4: -149.4989014, 113.2376175, -188.5170746, 140.5437622, -290.0426636, 301.7546997

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6300054
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6299901
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -118.5167923, 107.8890457, -214.7231140, 185.9732056, -304.4899902, 322.6120911
1: -92.7873993, 100.8808746, -168.6910553, 175.9269562, -268.7143250, 269.5718994
2: -134.2325439, 112.7549667, -243.5518188, 193.9766388, -328.2091675, 356.3067932
3: -54.0469742, 136.1377563, -95.6573029, 241.0182190, -295.0651855, 231.2200470
4: -149.4989014, 113.2376175, -271.0773621, 193.9729767, -343.4718628, 384.3149719

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6300054
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6299901
time: 0.79 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -150.6682129, 137.8992462, -181.5190277, 153.3090973, -303.9772949, 319.4181824
1: -117.9103165, 130.2588501, -142.5338440, 144.3416901, -262.2520142, 272.7926941
2: -170.7127686, 144.5482178, -205.8718872, 159.6268616, -330.3396301, 350.4201050
3: -70.9683914, 171.8507385, -78.1683502, 204.1779022, -275.1463013, 250.0190887
4: -190.3519287, 143.9389343, -228.9604187, 160.4309540, -350.7828979, 372.8993530

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307684
time: 0.60 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6308252
time: 0.69 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -181.5190277, 153.3090973, -368.0322266, 367.4922485
1: -168.6910553, 175.9269562, -142.5338440, 144.3416901, -313.0327454, 318.4607849
2: -243.5518188, 193.9766388, -205.8718872, 159.6268616, -403.1786804, 399.8485107
3: -95.6573029, 241.0182190, -78.1683502, 204.1779022, -299.4284973, 319.1865540
4: -271.0773621, 193.9729767, -228.9604187, 160.4309540, -431.5083008, 422.9333801

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307636
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307839
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.10 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311390, upper bound: 187.6311390
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311390, upper bound: 187.6311568
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6311390
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311568, upper bound: 187.6311568
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311433, upper bound: 187.6299901
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6311433, upper bound: 187.6299901
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6379026, upper bound: 187.6307642
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6379026, upper bound: 187.6307839
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6300054
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6299901
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6300054
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299942, upper bound: 187.6299901
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307684
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6308252
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307636
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 3, lower bound: -187.6299901, upper bound: 187.6307839

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -93.5899734, 92.8750916, -186.4650574, 186.4650574
1: -73.1107025, 87.0040359, -73.1107025, 87.0040359, -160.1147461, 160.1147308
2: -105.9443512, 97.7353897, -105.9443512, 97.7353897, -203.6797485, 203.6797485
3: -47.2811356, 109.5639648, -47.2811356, 109.5639648, -156.8450928, 156.8450928
4: -118.2672424, 97.3865814, -118.2672424, 97.3865814, -215.6537933, 215.6538086

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279461, upper bound: 187.8012999
time: 0.67 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279461, upper bound: 187.8012999
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -149.9745941, 137.3406982, -230.9306641, 242.8496857
1: -73.1107025, 87.0040359, -117.3068619, 129.7829132, -202.8936157, 204.3108826
2: -105.9443512, 97.7353897, -169.9014282, 144.0322723, -249.9765625, 267.6368103
3: -47.2811356, 109.5639648, -70.6789017, 171.0912170, -218.3723450, 180.2428589
4: -118.2672424, 97.3865814, -189.4996490, 143.3141022, -261.5813293, 286.8862000

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285558, upper bound: 187.7987681
time: 0.66 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309205, upper bound: 187.8012999
time: 0.75 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -93.5899734, 92.8750916, -242.8496857, 230.9306641
1: -117.3068619, 129.7829132, -73.1107025, 87.0040359, -204.3108978, 202.8936005
2: -169.9014282, 144.0322723, -105.9443512, 97.7353897, -267.6368103, 249.9765320
3: -70.6789017, 171.0912170, -47.2811356, 109.5639648, -180.2428589, 218.3723450
4: -189.4996490, 143.3141022, -118.2672424, 97.3865814, -286.8861694, 261.5813599

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
time: 0.59 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -149.9745941, 137.3406982, -149.9745941, 137.3406982, -287.3153076, 287.3153076
1: -117.3068619, 129.7829132, -117.3068619, 129.7829132, -247.0897827, 247.0897675
2: -169.9014282, 144.0322723, -169.9014282, 144.0322723, -313.9336853, 313.9336548
3: -70.6789017, 171.0912170, -70.6789017, 171.0912170, -241.7701111, 241.7701111
4: -189.4996490, 143.3141022, -189.4996490, 143.3141022, -332.8137512, 332.8137512

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
time: 0.61 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -93.5899734, 92.8750916, -242.1590271, 228.2793274
1: -116.9905624, 127.1221771, -73.1107025, 87.0040359, -203.9945984, 200.2328796
2: -169.2326202, 141.1778259, -105.9443512, 97.7353897, -266.9680176, 247.1221313
3: -69.0598907, 170.1712341, -47.2811356, 109.5639648, -178.6238556, 217.4523621
4: -188.5170746, 140.5437622, -118.2672424, 97.3865814, -285.9035950, 258.8109436

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274614, upper bound: 187.5290925
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309568, upper bound: 187.6293981
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -149.9745941, 137.3406982, -286.6246338, 284.6639404
1: -116.9905624, 127.1221771, -117.3068619, 129.7829132, -246.7734680, 244.4290009
2: -169.2326202, 141.1778259, -169.9014282, 144.0322723, -313.2648926, 311.0792236
3: -69.0598907, 170.1712341, -70.6789017, 171.0912170, -240.1510925, 240.8501129
4: -188.5170746, 140.5437622, -189.4996490, 143.3141022, -331.8311768, 330.0433960

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286071, upper bound: 187.6269352
time: 0.57 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309569, upper bound: 187.6293981
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -214.5149994, 185.7349548, -93.5899734, 92.8750916, -307.3900757, 279.3248901
1: -168.5262756, 175.6992188, -73.1107025, 87.0040359, -255.5303040, 248.8099213
2: -243.3150024, 193.7258453, -105.9443512, 97.7353897, -341.0503845, 299.6701965
3: -95.5249634, 240.7821655, -47.2811356, 109.5639648, -204.3985291, 288.0632935
4: -270.8131104, 193.7270813, -118.2672424, 97.3865814, -368.1996765, 311.9942627

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6370516, upper bound: 187.6303796
time: 0.83 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -149.9745941, 137.3406982, -352.0638123, 335.9478149
1: -168.6910553, 175.9269562, -117.3068619, 129.7829132, -298.4739685, 293.2337952
2: -243.5518188, 193.9766388, -169.9014282, 144.0322723, -387.5840454, 363.8780518
3: -95.6573029, 241.0182190, -70.6789017, 171.0912170, -266.1802063, 311.6970520
4: -271.0773621, 193.9729767, -189.4996490, 143.3141022, -414.3914795, 383.4726257

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
time: 1.01 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6370516, upper bound: 187.6303796
time: 0.79 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -149.2839355, 134.6893616, -228.2793121, 242.1590271
1: -73.1107025, 87.0040359, -116.9905624, 127.1221771, -200.2328796, 203.9945831
2: -105.9443512, 97.7353897, -169.2326202, 141.1778259, -247.1221771, 266.9680176
3: -47.2811356, 109.5639648, -69.0598907, 170.1712341, -217.4523621, 178.6238556
4: -118.2672424, 97.3865814, -188.5170746, 140.5437622, -258.8109131, 285.9036255

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5293691, upper bound: 187.6263038
time: 0.60 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297745, upper bound: 187.6298520
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -148.4843750, 134.0473938, -149.2839355, 134.6893616, -283.1737366, 283.3313293
1: -116.2964935, 126.5755463, -116.9905624, 127.1221771, -243.4186707, 243.5661011
2: -168.2982788, 140.5842438, -169.2326202, 141.1778259, -309.4760742, 309.8168640
3: -68.7259674, 169.2932587, -69.0598907, 170.1712341, -238.8971710, 238.3531494
4: -187.5333405, 139.8242035, -188.5170746, 140.5437622, -328.0770874, 328.3412476

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5293691, upper bound: 187.6263038
time: 0.73 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297745, upper bound: 187.6297745
time: 0.60 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -93.5899734, 92.8750916, -214.5149994, 185.7349548, -279.3249207, 307.3900757
1: -73.1107025, 87.0040359, -168.5262756, 175.6992188, -248.8099060, 255.5303040
2: -105.9443512, 97.7353897, -243.3150024, 193.7258453, -299.6701965, 341.0503235
3: -47.2811356, 109.5639648, -95.5249634, 240.7821655, -288.0632935, 204.3985138
4: -118.2672424, 97.3865814, -270.8131104, 193.7270813, -311.9942627, 368.1997070

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282974, upper bound: 187.6292093
time: 0.54 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6305481, upper bound: 187.6361837
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -148.4843750, 134.0473938, -214.7231140, 185.9732056, -334.4575806, 348.7704773
1: -116.2964935, 126.5755463, -168.6910553, 175.9269562, -292.2234192, 295.2665710
2: -168.2982788, 140.5842438, -243.5518188, 193.9766388, -362.2749023, 384.1360474
3: -68.7259674, 169.2932587, -95.6573029, 241.0182190, -309.7441406, 264.4364014
4: -187.5333405, 139.8242035, -271.0773621, 193.9729767, -381.5063171, 410.9015198

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282974, upper bound: 187.6269352
time: 0.69 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6305481, upper bound: 187.6293981
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -150.6682129, 137.8992462, -149.2839355, 134.6893616, -285.3575134, 287.1831360
1: -117.9103165, 130.2588501, -116.9905624, 127.1221771, -245.0324860, 247.2494202
2: -170.7127686, 144.5482178, -169.2326202, 141.1778259, -311.8905640, 313.7808228
3: -70.9683914, 171.8507385, -69.0598907, 170.1712341, -241.1395874, 240.9106293
4: -190.3519287, 143.9389343, -188.5170746, 140.5437622, -330.8956909, 332.4559937

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269352, upper bound: 187.6286071
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -150.6682129, 137.8992462, -214.7231140, 185.9732056, -336.6414185, 352.6223450
1: -117.9103165, 130.2588501, -168.6910553, 175.9269562, -293.8372498, 298.9498596
2: -170.7127686, 144.5482178, -243.5518188, 193.9766388, -364.6893921, 388.1000061
3: -70.9683914, 171.8507385, -95.6573029, 241.0182190, -311.9865723, 266.9418335
4: -190.3519287, 143.9389343, -271.0773621, 193.9729767, -384.3248901, 415.0162964

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
time: 0.61 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309568
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -149.2839355, 134.6893616, -349.4123840, 335.2571411
1: -168.6910553, 175.9269562, -116.9905624, 127.1221771, -295.8131714, 292.9174805
2: -243.5518188, 193.9766388, -169.2326202, 141.1778259, -384.7296143, 363.2092590
3: -95.6573029, 241.0182190, -69.0598907, 170.1712341, -265.3165283, 310.0780640
4: -271.0773621, 193.9729767, -188.5170746, 140.5437622, -411.6211243, 382.4900208

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292093, upper bound: 187.6279995
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6361837, upper bound: 187.6303796
time: 0.80 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -214.7231140, 185.9732056, -400.6963196, 400.6963196
1: -168.6910553, 175.9269562, -168.6910553, 175.9269562, -344.6179504, 344.6179504
2: -243.5518188, 193.9766388, -243.5518188, 193.9766388, -437.5284424, 437.5284424
3: -95.6573029, 241.0182190, -95.6573029, 241.0182190, -336.2658691, 336.2658691
4: -271.0773621, 193.9729767, -271.0773621, 193.9729767, -465.0502930, 465.0503235

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5365508, upper bound: 187.6272877
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6361838, upper bound: 187.6303795
time: 0.78 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.44 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6279461, upper bound: 187.8012999
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6279461, upper bound: 187.8012999
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6285558, upper bound: 187.7987681
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6309205, upper bound: 187.8012999
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6274614, upper bound: 187.5290925
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6309568, upper bound: 187.6293981
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6286071, upper bound: 187.6269352
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6309569, upper bound: 187.6293981
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6370516, upper bound: 187.6303796
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6370516, upper bound: 187.6303796
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.5293691, upper bound: 187.6263038
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6297745, upper bound: 187.6298520
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.5293691, upper bound: 187.6263038
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6297745, upper bound: 187.6297745
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6282974, upper bound: 187.6292093
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6305481, upper bound: 187.6361837
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6282974, upper bound: 187.6269352
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6305481, upper bound: 187.6293981
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6269352, upper bound: 187.6286071
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
IS_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
IS_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309568
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6292093, upper bound: 187.6279995
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6361837, upper bound: 187.6303796
IS_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.5365508, upper bound: 187.6272877
IS_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -187.6361838, upper bound: 187.6303795

## BFS IS instance: IS_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -79.7371521, 81.4387817, -122.4197693, 134.5162659
1: -32.1188278, 51.3675804, -62.2732544, 76.2692413, -108.3880386, 113.6408234
2: -47.0364113, 57.8066292, -90.3249054, 85.6733170, -132.7097168, 148.1315308
3: -27.2729225, 55.0080185, -41.1421471, 94.7014923, -121.9744034, 96.1501617
4: -53.0424118, 57.1621284, -100.9268417, 85.4481277, -138.4905243, 158.0889587

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8176645, upper bound: 187.8176645
time: 0.84 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8176645, upper bound: 187.8199886
time: 0.70 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -93.3917847, 92.7331238, -182.1810760, 183.2736816
1: -69.8412399, 84.1715393, -72.9543228, 86.8698502, -156.7110748, 157.1258087
2: -101.2739258, 94.6007919, -105.7210693, 97.5862808, -198.8601990, 200.3218536
3: -45.8811989, 105.0105896, -47.2158279, 109.3487320, -155.2299347, 152.2264099
4: -113.0568390, 94.2403870, -118.0184097, 97.2374649, -210.2943115, 212.2587891

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_A2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6517903, upper bound: 187.4442482
time: 0.64 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8168358, upper bound: 187.8168357
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -79.7371521, 81.4387817, -90.0296860, 95.7459259, -175.4830780, 171.4684753
1: -62.2732544, 76.2692413, -70.5223999, 90.7102203, -152.9834747, 146.7916107
2: -90.3249054, 85.6733170, -102.4902802, 100.3906708, -190.7155762, 188.1636047
3: -41.1421471, 94.7014923, -48.7527466, 108.2179565, -149.3600769, 143.4542236
4: -100.9268417, 85.4481277, -114.9058533, 99.2772675, -200.2040863, 200.3539734

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258351, upper bound: 187.7987681
time: 0.79 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258351, upper bound: 187.7987681
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -93.3917847, 92.7331238, -146.0465698, 134.7337646, -228.1255493, 238.7796936
1: -72.9543228, 86.8698502, -114.2232513, 127.2852173, -200.2394867, 201.0930786
2: -105.7210693, 97.5862808, -165.4884644, 141.3374176, -247.0584564, 263.0747375
3: -47.2158279, 109.3487320, -69.3965378, 166.8359833, -214.0518188, 178.7452698
4: -118.0184097, 97.2374649, -184.5755615, 140.5543518, -258.5727539, 281.8129883

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6198679, upper bound: 187.6964804
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299725, upper bound: 187.4896723
time: 0.67 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290361, upper bound: 187.7972518
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -79.7371521, 81.4387817, -171.4684753, 175.4830780
1: -70.5223999, 90.7102203, -62.2732544, 76.2692413, -146.7916260, 152.9834747
2: -102.4902802, 100.3906708, -90.3249054, 85.6733170, -188.1636047, 190.7155762
3: -48.7527466, 108.2179565, -41.1421471, 94.7014923, -143.4542236, 149.3600769
4: -114.9058533, 99.2772675, -100.9268417, 85.4481277, -200.3539734, 200.2041016

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6258351
time: 0.59 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6285558
time: 0.70 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -93.3917847, 92.7331238, -238.7796936, 228.1255493
1: -114.2232513, 127.2852173, -72.9543228, 86.8698502, -201.0930939, 200.2395020
2: -165.4884644, 141.3374176, -105.7210693, 97.5862808, -263.0747375, 247.0584869
3: -69.3965378, 166.8359833, -47.2158279, 109.3487320, -178.7452698, 214.0518188
4: -184.5755615, 140.5543518, -118.0184097, 97.2374649, -281.8130188, 258.5727539

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6964804, upper bound: 187.6198679
time: 0.74 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
time: 0.59 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -135.9420776, 125.6306152, -215.6603088, 231.6880035
1: -70.5223999, 90.7102203, -106.3213120, 118.7759705, -189.2983551, 197.0315247
2: -102.4902802, 100.3906708, -154.0769043, 131.6185913, -234.1088715, 254.4675598
3: -48.7527466, 108.2179565, -64.5234070, 156.0122223, -204.7649689, 172.7413330
4: -114.9058533, 99.2772675, -171.9312897, 131.0277863, -245.9336090, 271.2085571

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258351, upper bound: 187.6258351
time: 0.86 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258351, upper bound: 187.6282582
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -149.7934418, 137.2201385, -283.2666626, 284.5272217
1: -114.2232513, 127.2852173, -117.1645584, 129.6675110, -243.8907471, 244.4497528
2: -165.4884644, 141.3374176, -169.6975098, 143.9083099, -309.3967590, 311.0349121
3: -69.3965378, 166.8359833, -70.6193466, 170.8952637, -240.2918091, 237.4553223
4: -184.5755615, 140.5543518, -189.2721252, 143.1866150, -327.7620544, 329.8264771

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350329
time: 0.63 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289659, upper bound: 187.6289659
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -79.7371521, 81.4387817, -170.0465088, 172.5475159
1: -69.5185242, 87.8645935, -62.2732544, 76.2692413, -145.7877502, 150.1378326
2: -100.9003143, 97.3328018, -90.3249054, 85.6733170, -186.5736389, 187.6577148
3: -47.0367928, 106.5991135, -41.1421471, 94.7014923, -141.7382812, 147.7412415
4: -113.0341187, 96.2283783, -100.9268417, 85.4481277, -198.4822388, 197.1551819

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982111, upper bound: 187.5270964
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982111, upper bound: 187.5293691
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -93.3917847, 92.7331238, -237.5423889, 225.0508270
1: -113.4549103, 124.2506790, -72.9543228, 86.8698502, -200.3247681, 197.2050018
2: -164.1941833, 138.0722961, -105.7210693, 97.5862808, -261.7804565, 243.7933655
3: -67.5839310, 165.3264618, -47.2158279, 109.3487320, -176.9326630, 212.5422974
4: -182.9160309, 137.3219604, -118.0184097, 97.2374649, -280.1535034, 255.3403625

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4443526, upper bound: 187.5786626
time: 0.61 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982785, upper bound: 187.6297534
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -134.2789459, 122.3004990, -90.0296860, 95.7459259, -230.0248718, 212.3301544
1: -105.1659775, 115.5536118, -70.5223999, 90.7102203, -195.8761902, 186.0759735
2: -152.2682495, 128.1363678, -102.4902802, 100.3906708, -252.6588745, 230.6266479
3: -62.5633125, 154.0523529, -48.7527466, 108.2179565, -170.7812653, 202.8050842
4: -169.7571564, 127.5054779, -114.9058533, 99.2772675, -269.0344238, 242.4113312

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5272318
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6269352
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -149.0803223, 134.5512085, -146.0465698, 134.7337646, -283.8140869, 280.5977783
1: -116.8296204, 126.9912109, -114.2232513, 127.2852173, -244.1148376, 241.2144165
2: -169.0030518, 141.0368652, -165.4884644, 141.3374176, -310.3404541, 306.5253296
3: -68.9912720, 169.9507294, -69.3965378, 166.8359833, -235.8272400, 239.3472595
4: -188.2618713, 140.3968811, -184.5755615, 140.5543518, -328.8162231, 324.9724121

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350550, upper bound: 187.6285091
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6276910
time: 0.67 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.7027283, 137.3081360, -79.7371521, 81.4387817, -225.1415100, 216.1154633
1: -112.7039871, 130.4257812, -62.2732544, 76.2692413, -188.9732208, 190.9268799
2: -163.4894104, 143.3984528, -90.3249054, 85.6733170, -249.1626740, 231.7618561
3: -69.9211426, 166.6706390, -41.1421471, 94.7014923, -162.7039795, 207.8127747
4: -182.6679993, 142.1687927, -100.9268417, 85.4481277, -268.1161194, 242.1531219

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6255764
time: 2.02 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6282974
time: 2.27 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -209.4572144, 182.0884857, -93.3917847, 92.7331238, -302.1903076, 275.4802246
1: -164.5061646, 172.2222595, -72.9543228, 86.8698502, -251.3759613, 245.1765137
2: -237.6264496, 189.9697571, -105.7210693, 97.5862808, -335.2127380, 295.6908264
3: -93.7043915, 235.3221741, -47.2158279, 109.3487320, -202.3717346, 282.5379639
4: -264.4715881, 189.8687897, -118.0184097, 97.2374649, -361.7090454, 307.8872070

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4509566, upper bound: 187.5855679
time: 0.70 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982106, upper bound: 187.6305415
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -135.9420776, 125.6306152, -269.5880737, 272.9370117
1: -112.9016113, 130.6958466, -106.3213120, 118.7759705, -231.6775665, 235.4334259
2: -163.7802124, 143.6897125, -154.0769043, 131.6185913, -295.3988037, 296.0818481
3: -70.0736237, 166.9585724, -64.5234070, 156.0122223, -224.2823639, 231.4819641
4: -182.9930725, 142.4557343, -171.9312897, 131.0277863, -314.0208740, 313.7619934

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255915
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6279995
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -149.7934418, 137.2201385, -347.5008545, 332.8009644
1: -165.1631775, 173.1105804, -117.1645584, 129.6675110, -294.8306274, 290.2751465
2: -238.5625763, 190.9392853, -169.6975098, 143.9083099, -382.4708252, 360.6367798
3: -94.2064362, 236.2539673, -70.6193466, 170.8952637, -264.4667053, 306.8733215
4: -265.5162659, 190.8201752, -189.2721252, 143.1866150, -408.7028503, 380.0922852

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6294468
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6335489, upper bound: 187.6286043
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -79.7371521, 81.4387817, -88.6077271, 92.8103714, -172.5475006, 170.0465088
1: -62.2732544, 76.2692413, -69.5185242, 87.8645935, -150.1378326, 145.7877655
2: -90.3249054, 85.6733170, -100.9003143, 97.3328018, -187.6577148, 186.5736237
3: -41.1421471, 94.7014923, -47.0367928, 106.5991135, -147.7412262, 141.7382812
4: -100.9268417, 85.4481277, -113.0341187, 96.2283783, -197.1551666, 198.4822388

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7982111
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7982111
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -93.3917847, 92.7331238, -144.8092651, 131.6590424, -225.0508270, 237.5423889
1: -72.9543228, 86.8698502, -113.4549103, 124.2506790, -197.2050018, 200.3247681
2: -105.7210693, 97.5862808, -164.1941833, 138.0722961, -243.7933350, 261.7804565
3: -47.2158279, 109.3487320, -67.5839310, 165.3264618, -212.5422974, 176.9326630
4: -118.0184097, 97.2374649, -182.9160309, 137.3219604, -255.3403625, 280.1535034

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5786626, upper bound: 187.4443526
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6297534, upper bound: 187.7982785
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -134.2789459, 122.3004990, -88.6077271, 92.8103714, -227.0893097, 210.9082031
1: -105.1659775, 115.5536118, -69.5185242, 87.8645935, -193.0305634, 185.0721130
2: -152.2682495, 128.1363678, -100.9003143, 97.3328018, -249.6010437, 229.0366821
3: -62.5633125, 154.0523529, -47.0367928, 106.5991135, -169.1624298, 201.0891418
4: -169.7571564, 127.5054779, -113.0341187, 96.2283783, -265.9855347, 240.5395813

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5263046, upper bound: 187.5263046
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5263046, upper bound: 187.6263038
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -148.2934875, 133.9195099, -144.8092651, 131.6590424, -279.9525146, 278.7287598
1: -116.1464539, 126.4533005, -113.4549103, 124.2506790, -240.3971252, 239.9082031
2: -168.0833893, 140.4526978, -164.1941833, 138.0722961, -306.1557007, 304.6468506
3: -68.6626892, 169.0866852, -67.5839310, 165.3264618, -233.9891510, 236.6706238
4: -187.2937012, 139.6888123, -182.9160309, 137.3219604, -324.6156311, 322.6048584

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4195023, upper bound: 187.5787226
time: 0.77 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298325, upper bound: 187.6297534
time: 0.72 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -79.7371521, 81.4387817, -143.7027283, 137.3081360, -216.1154633, 225.1415100
1: -62.2732544, 76.2692413, -112.7039871, 130.4257812, -190.9268799, 188.9732056
2: -90.3249054, 85.6733170, -163.4894104, 143.3984528, -231.7618713, 249.1627045
3: -41.1421471, 94.7014923, -69.9211426, 166.6706390, -207.8127747, 162.7039795
4: -100.9268417, 85.4481277, -182.6679993, 142.1687927, -242.1531219, 268.1160889

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
time: 0.58 seconds

## Relational analysis of IS_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -93.3917847, 92.7331238, -209.4572144, 182.0884857, -275.4802246, 302.1903076
1: -72.9543228, 86.8698502, -164.5061646, 172.2222595, -245.1765442, 251.3759460
2: -105.7210693, 97.5862808, -237.6264496, 189.9697571, -295.6908264, 335.2127075
3: -47.2158279, 109.3487320, -93.7043915, 235.3221741, -282.5379639, 202.3717346
4: -118.0184097, 97.2374649, -264.4715881, 189.8687897, -307.8872070, 361.7090454

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5855680, upper bound: 187.4509566
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6305415, upper bound: 187.7982106
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -134.2789459, 122.3004990, -143.9574738, 137.5898285, -271.3230896, 266.2579041
1: -105.1659775, 115.5536118, -112.9016113, 130.6958466, -234.3113098, 228.4552002
2: -152.2682495, 128.1363678, -163.7802124, 143.6897125, -294.3291626, 291.9165649
3: -62.5633125, 154.0523529, -70.0736237, 166.9585724, -229.5218811, 222.3678131
4: -169.7571564, 127.5054779, -182.9930725, 142.4557343, -311.6460571, 310.4985352

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250977, upper bound: 187.5272318
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250977, upper bound: 187.6269352
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -148.2934875, 133.9195099, -210.2807465, 183.0075378, -331.3009949, 344.2002563
1: -116.1464539, 126.4533005, -165.1631775, 173.1105804, -289.2570190, 291.6164246
2: -168.0833893, 140.4526978, -238.5625763, 190.9392853, -359.0226746, 379.0152588
3: -68.6626892, 169.0866852, -94.2064362, 236.2539673, -304.9166565, 262.7128296
4: -187.2937012, 139.6888123, -265.5162659, 190.8201752, -378.1138916, 405.2050781

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4181084, upper bound: 187.4076338
time: 0.62 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6305780, upper bound: 187.6293981
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -134.2789459, 122.3004990, -212.3301544, 230.0248718
1: -70.5223999, 90.7102203, -105.1659775, 115.5536118, -186.0759888, 195.8761749
2: -102.4902802, 100.3906708, -152.2682495, 128.1363678, -230.6266479, 252.6588898
3: -48.7527466, 108.2179565, -62.5633125, 154.0523529, -202.8050690, 170.7812653
4: -114.9058533, 99.2772675, -169.7571564, 127.5054779, -242.4113312, 269.0344238

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272318, upper bound: 187.6253563
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5270964, upper bound: 187.6286071
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -146.5121613, 135.0994873, -149.0803223, 134.5512085, -281.0633545, 284.1798096
1: -114.6307831, 127.5979538, -116.8296204, 126.9912109, -241.6219788, 244.4275818
2: -166.0353241, 141.6752014, -169.0030518, 141.0368652, -307.0721741, 310.6782532
3: -69.5864105, 167.3457031, -68.9912720, 169.9507294, -239.5371246, 236.3369751
4: -185.1463776, 140.9630737, -188.2618713, 140.3968811, -325.5432129, 329.2249451

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285091, upper bound: 187.4350550
time: 0.80 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276910, upper bound: 187.6290606
time: 0.74 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -135.9420776, 125.6306152, -143.9574738, 137.5898285, -272.9370422, 269.5880432
1: -106.3213120, 118.7759705, -112.9016113, 130.6958466, -235.4334412, 231.6775818
2: -154.0769043, 131.6185913, -163.7802124, 143.6897125, -296.0818481, 295.3988037
3: -64.5234070, 156.0122223, -70.0736237, 166.9585724, -231.4819489, 224.2823792
4: -171.9312897, 131.0277863, -182.9930725, 142.4557343, -313.7620239, 314.0208740

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6253563
time: 0.71 seconds

## Relational analysis of IS_B2_A2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6274614
time: 0.68 seconds

## BFS IS instance: IS_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -150.4766083, 137.7697296, -210.2807465, 183.0075378, -333.4841309, 348.0504150
1: -117.7590103, 130.1358948, -165.1631775, 173.1105804, -290.8695374, 295.2990417
2: -170.4966888, 144.4160309, -238.5625763, 190.9392853, -361.4359741, 382.9786072
3: -70.9042969, 171.6432495, -94.2064362, 236.2539673, -307.1581726, 265.2167664
4: -190.1115570, 143.8014832, -265.5162659, 190.8201752, -380.9317322, 409.3177185

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292600, upper bound: 187.4350550
time: 0.84 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284903, upper bound: 187.6290606
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -134.2789459, 122.3004990, -266.2579346, 271.3230286
1: -112.9016113, 130.6958466, -105.1659775, 115.5536118, -228.4552002, 234.3113251
2: -163.7802124, 143.6897125, -152.2682495, 128.1363678, -291.9165649, 294.3291931
3: -70.0736237, 166.9585724, -62.5633125, 154.0523529, -222.3678131, 229.5218811
4: -182.9930725, 142.4557343, -169.7571564, 127.5054779, -310.4985352, 311.6460266

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6250977
time: 0.70 seconds

## Relational analysis of IS_B2_A2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6282974
time: 0.73 seconds

## BFS IS instance: IS_B2_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -149.0803223, 134.5512085, -344.8319702, 332.0878601
1: -165.1631775, 173.1105804, -116.8296204, 126.9912109, -292.1543274, 289.9401855
2: -238.5625763, 190.9392853, -169.0030518, 141.0368652, -379.5993958, 359.9423218
3: -94.2064362, 236.2539673, -68.9912720, 169.9507294, -263.5791321, 305.2452087
4: -265.5162659, 190.8201752, -188.2618713, 140.3968811, -405.9131165, 379.0820312

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5874588, upper bound: 187.4825839
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6361837, upper bound: 187.6305415
time: 0.64 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -198.5921936, 172.9775543, -143.9574738, 137.5898285, -335.9071045, 316.9349976
1: -155.8728333, 163.7884216, -112.9016113, 130.6958466, -285.2734375, 276.6899414
2: -225.3496094, 180.3471985, -163.7802124, 143.6897125, -367.6812744, 344.1274109
3: -88.8018494, 223.7629547, -70.0736237, 166.9585724, -254.7816162, 292.1777344
4: -250.9470367, 180.2328949, -182.9930725, 142.4557343, -393.2190552, 363.2259521

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6252249
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6272877
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -214.5238037, 185.8393707, -210.2807465, 183.0075378, -397.5313416, 396.1201172
1: -168.5326691, 175.7993317, -165.1631775, 173.1105804, -341.6432190, 340.9624634
2: -243.3277893, 193.8392792, -238.5625763, 190.9392853, -434.2670898, 432.4018555
3: -95.5921478, 240.8043213, -94.2064362, 236.2539673, -331.2065125, 334.5350952
4: -270.8276978, 193.8303833, -265.5162659, 190.8201752, -461.6478882, 459.3466492

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4892789, upper bound: 187.5892709
time: 0.69 seconds

## Relational analysis of IS_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6303795
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.47 seconds
IS_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.8176645, upper bound: 187.8176645
IS_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.8176645, upper bound: 187.8199886
IS_B1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6517903, upper bound: 187.4442482
IS_B1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.8168358, upper bound: 187.8168357
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6258351, upper bound: 187.7987681
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6258351, upper bound: 187.7987681
IS_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6299725, upper bound: 187.4896723
IS_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6290361, upper bound: 187.7972518
IS_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6258351
IS_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6285558
IS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
IS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
IS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6258351, upper bound: 187.6258351
IS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6258351, upper bound: 187.6282582
IS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350329
IS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6289659, upper bound: 187.6289659
IS_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982111, upper bound: 187.5270964
IS_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982111, upper bound: 187.5293691
IS_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4443526, upper bound: 187.5786626
IS_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982785, upper bound: 187.6297534
IS_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5272318
IS_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6269352
IS_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4350550, upper bound: 187.6285091
IS_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6276910
IS_B1_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6255764
IS_B1_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6282974
IS_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4509566, upper bound: 187.5855679
IS_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.7982106, upper bound: 187.6305415
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255915
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6279995
IS_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6294468
IS_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6335489, upper bound: 187.6286043
IS_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7982111
IS_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7982111
IS_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5786626, upper bound: 187.4443526
IS_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6297534, upper bound: 187.7982785
IS_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5263046, upper bound: 187.5263046
IS_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5263046, upper bound: 187.6263038
IS_B2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4195023, upper bound: 187.5787226
IS_B2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6298325, upper bound: 187.6297534
IS_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
IS_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
IS_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5855680, upper bound: 187.4509566
IS_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6305415, upper bound: 187.7982106
IS_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6250977, upper bound: 187.5272318
IS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6250977, upper bound: 187.6269352
IS_B2_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4181084, upper bound: 187.4076338
IS_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6305780, upper bound: 187.6293981
IS_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5272318, upper bound: 187.6253563
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5270964, upper bound: 187.6286071
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6285091, upper bound: 187.4350550
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6276910, upper bound: 187.6290606
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6253563
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6274614
IS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6292600, upper bound: 187.4350550
IS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6284903, upper bound: 187.6290606
IS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6250977
IS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6282974
IS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.5874588, upper bound: 187.4825839
IS_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6361837, upper bound: 187.6305415
IS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6252249
IS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6272877
IS_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.4892789, upper bound: 187.5892709
IS_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.47
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6303795

## BFS IS instance: IS_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -40.9809875, 54.7791138, -95.7601013, 95.7600937
1: -32.1188278, 51.3675804, -32.1188278, 51.3675804, -83.4863968, 83.4863968
2: -47.0364113, 57.8066292, -47.0364113, 57.8066292, -104.8430405, 104.8430405
3: -27.2729225, 55.0080185, -27.2729225, 55.0080185, -82.2809448, 82.2809448
4: -53.0424118, 57.1621284, -53.0424118, 57.1621284, -110.2045288, 110.2045135

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8124576, upper bound: 187.3877039
time: 0.65 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8126227, upper bound: 187.8121031
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -89.4479752, 89.8819046, -130.8628845, 144.2270508
1: -32.1188278, 51.3675804, -69.8412399, 84.1715393, -116.2903595, 121.2088165
2: -47.0364113, 57.8066292, -101.2739258, 94.6007919, -141.6372070, 159.0805511
3: -27.2729225, 55.0080185, -45.8811989, 105.0105896, -132.2835083, 100.8892136
4: -53.0424118, 57.1621284, -113.0568390, 94.2403870, -147.2828064, 170.2189484

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8162422, upper bound: 187.7619780
time: 0.58 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8181337, upper bound: 187.8199886
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -74.7018280, 80.6288452, -90.4210892, 90.9067383, -165.6085663, 171.0499268
1: -58.1614151, 75.5373001, -70.6095963, 85.1660919, -143.3274841, 146.1468964
2: -84.4807281, 85.4331818, -102.3586121, 95.7739563, -180.2546844, 187.7917938
3: -41.7390480, 88.9322357, -46.3760376, 106.1724014, -147.9114532, 135.3082733
4: -94.5108185, 84.7035065, -114.2919922, 95.3324356, -189.8432465, 198.9954834

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6513821, upper bound: 187.4440515
time: 0.64 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6507981, upper bound: 187.4420780
time: 0.62 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -88.8516617, 89.4957199, -93.3917847, 92.7331238, -181.5847778, 182.8875122
1: -69.3641663, 83.8208542, -72.9543228, 86.8698502, -156.2339935, 156.7751312
2: -100.5895386, 94.2378540, -105.7210693, 97.5862808, -198.1758118, 199.9588928
3: -45.7091484, 104.3674927, -47.2158279, 109.3487320, -155.0578613, 151.5833130
4: -112.3173523, 93.8545685, -118.0184097, 97.2374649, -209.5548096, 211.8729858

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8000586, upper bound: 187.7248378
time: 0.68 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7238876, upper bound: 187.7238874
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -90.0296860, 95.7459259, -136.7269135, 144.8087921
1: -32.1188278, 51.3675804, -70.5223999, 90.7102203, -122.8290405, 121.8899765
2: -47.0364113, 57.8066292, -102.4902802, 100.3906708, -147.4270782, 160.2969055
3: -27.2729225, 55.0080185, -48.7527466, 108.2179565, -135.4908600, 103.7258301
4: -53.0424118, 57.1621284, -114.9058533, 99.2772675, -152.3196716, 172.0679779

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241148, upper bound: 187.3879327
time: 0.62 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239264, upper bound: 187.7953593
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -90.0296860, 95.7459259, -185.1938782, 179.9115906
1: -69.8412399, 84.1715393, -70.5223999, 90.7102203, -160.5514374, 154.6938934
2: -101.2739258, 94.6007919, -102.4902802, 100.3906708, -201.6645660, 197.0910645
3: -45.8811989, 105.0105896, -48.7527466, 108.2179565, -154.0991516, 153.6334076
4: -113.0568390, 94.2403870, -114.9058533, 99.2772675, -212.3341064, 209.1462402

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4195676, upper bound: 187.7964483
time: 0.58 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258260, upper bound: 187.7987580
time: 0.76 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -82.5392914, 86.3880310, -145.9277954, 134.6750488, -217.2143402, 232.3158112
1: -64.6071396, 80.9783936, -114.1324387, 127.2294235, -191.8365631, 195.1108398
2: -93.6184311, 91.1768112, -165.3568573, 141.2772369, -234.8956604, 256.5336609
3: -44.0479164, 98.4312515, -69.3667374, 166.7200623, -210.7679596, 167.7979889
4: -104.7108383, 90.5531921, -184.4309692, 140.4914246, -245.2022400, 274.9841614

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6193492, upper bound: 187.4821913
time: 0.63 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6123496, upper bound: 187.4827854
time: 0.71 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299725, upper bound: 187.4870612
time: 0.80 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -90.0182343, 90.5549622, -146.0465698, 134.7337646, -224.7519989, 236.6015320
1: -70.2976913, 84.8072433, -114.2232513, 127.2852173, -197.5829010, 199.0304871
2: -101.8951874, 95.3282242, -165.4884644, 141.3374176, -243.2325745, 260.8166809
3: -46.1052017, 105.8526535, -69.3965378, 166.8359833, -212.9411926, 175.2491913
4: -113.8105698, 94.9274139, -184.5755615, 140.5543518, -254.3648682, 279.5029907

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189342, upper bound: 187.6929544
time: 0.72 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248659, upper bound: 187.7582411
time: 0.71 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290239, upper bound: 187.7901784
time: 0.58 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -40.9809875, 54.7791138, -144.8087921, 136.7269135
1: -70.5223999, 90.7102203, -32.1188278, 51.3675804, -121.8899765, 122.8290405
2: -102.4902802, 100.3906708, -47.0364113, 57.8066292, -160.2969055, 147.4270782
3: -48.7527466, 108.2179565, -27.2729225, 55.0080185, -103.7258301, 135.4908752
4: -114.9058533, 99.2772675, -53.0424118, 57.1621284, -172.0679779, 152.3196564

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3877502, upper bound: 187.6241148
time: 0.64 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6239264
time: 0.78 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -89.4479752, 89.8819046, -179.9115906, 185.1938782
1: -70.5223999, 90.7102203, -69.8412399, 84.1715393, -154.6938782, 160.5514221
2: -102.4902802, 100.3906708, -101.2739258, 94.6007919, -197.0910645, 201.6645813
3: -48.7527466, 108.2179565, -45.8811989, 105.0105896, -153.6334076, 154.0991516
4: -114.9058533, 99.2772675, -113.0568390, 94.2403870, -209.1462402, 212.3341064

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7964483, upper bound: 187.4195676
time: 0.57 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7987580, upper bound: 187.6285551
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -145.9277954, 134.6750488, -82.5392914, 86.3880310, -232.3157959, 217.2143402
1: -114.1324387, 127.2294235, -64.6071396, 80.9783936, -195.1108398, 191.8365479
2: -165.3568573, 141.2772369, -93.6184311, 91.1768112, -256.5336609, 234.8956604
3: -69.3667374, 166.7200623, -44.0479164, 98.4312515, -167.7979889, 210.7679749
4: -184.4309692, 140.4914246, -104.7108383, 90.5531921, -274.9841614, 245.2022247

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4821913, upper bound: 187.6193492
time: 0.69 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4827854, upper bound: 187.6123496
time: 0.80 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4870612, upper bound: 187.6299725
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -90.0182343, 90.5549622, -236.6015167, 224.7519989
1: -114.2232513, 127.2852173, -70.2976913, 84.8072433, -199.0304871, 197.5828857
2: -165.4884644, 141.3374176, -101.8951874, 95.3282242, -260.8166809, 243.2325745
3: -69.3965378, 166.8359833, -46.1052017, 105.8526535, -175.2491913, 212.9411926
4: -184.5755615, 140.5543518, -113.8105698, 94.9274139, -279.5029602, 254.3648682

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6929544, upper bound: 187.6189342
time: 0.79 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7582411, upper bound: 187.6248659
time: 0.72 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7901784, upper bound: 187.6290239
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -90.0296860, 95.7459259, -185.7756042, 185.7756042
1: -70.5223999, 90.7102203, -70.5223999, 90.7102203, -161.2325897, 161.2326050
2: -102.4902802, 100.3906708, -102.4902802, 100.3906708, -202.8809509, 202.8809509
3: -48.7527466, 108.2179565, -48.7527466, 108.2179565, -156.9706573, 156.9706726
4: -114.9058533, 99.2772675, -114.9058533, 99.2772675, -214.1831207, 214.1831207

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6183427, upper bound: 187.6072655
time: 0.59 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259777, upper bound: 187.6258351
time: 0.92 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -146.0465698, 134.7337646, -224.7634583, 241.7924957
1: -70.5223999, 90.7102203, -114.2232513, 127.2852173, -197.8076019, 204.9334412
2: -102.4902802, 100.3906708, -165.4884644, 141.3374176, -243.8276825, 265.8791504
3: -48.7527466, 108.2179565, -69.3965378, 166.8359833, -215.5791626, 177.4169769
4: -114.9058533, 99.2772675, -184.5755615, 140.5543518, -255.4602051, 283.8528442

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6117348, upper bound: 187.4141545
time: 0.61 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259690, upper bound: 187.6258260
time: 0.63 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -135.3423920, 128.6092987, -149.6694489, 137.1584930, -272.5008850, 278.2787170
1: -105.9854965, 121.5867310, -117.0696259, 129.6089478, -235.5944214, 238.6563110
2: -153.5312347, 135.1462860, -169.5599670, 143.8451996, -297.3764343, 304.7062378
3: -66.2919998, 156.0335083, -70.5882416, 170.7739410, -237.0659485, 226.6217499
4: -171.4211121, 134.0570679, -189.1209259, 143.1206360, -314.5417175, 323.1779785

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6123835, upper bound: 187.4314815
time: 0.72 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350266
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -142.5714111, 132.4927673, -149.7934418, 137.2201385, -279.7915039, 282.2861938
1: -111.4859314, 125.1495209, -117.1645584, 129.6675110, -241.1534424, 242.3140411
2: -161.5406494, 139.0050201, -169.6975098, 143.9083099, -305.4488831, 308.7025146
3: -68.2609253, 163.1841278, -70.6193466, 170.8952637, -239.1561737, 233.8034668
4: -180.2251587, 138.1768951, -189.2721252, 143.1866150, -323.4117126, 327.4489746

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
time: 0.69 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -40.9809875, 54.7791138, -143.3868256, 133.7913513
1: -69.5185242, 87.8645935, -32.1188278, 51.3675804, -120.8861008, 119.9834061
2: -100.9003143, 97.3328018, -47.0364113, 57.8066292, -158.7069397, 144.3692017
3: -47.0367928, 106.5991135, -27.2729225, 55.0080185, -102.0447693, 133.8720245
4: -113.0341187, 96.2283783, -53.0424118, 57.1621284, -170.1962280, 149.2707672

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3872464, upper bound: 187.5055142
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7947934, upper bound: 187.5044697
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -89.4479752, 89.8819046, -178.4896240, 182.2583160
1: -69.5185242, 87.8645935, -69.8412399, 84.1715393, -153.6900330, 157.7058105
2: -100.9003143, 97.3328018, -101.2739258, 94.6007919, -195.5010986, 198.6067047
3: -47.0367928, 106.5991135, -45.8811989, 105.0105896, -152.0473785, 152.4803009
4: -113.0341187, 96.2283783, -113.0568390, 94.2403870, -207.2745056, 209.2851868

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7577321, upper bound: 187.5230579
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7911118, upper bound: 187.5293691
time: 0.68 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -141.1969452, 129.4559174, -78.1125488, 83.2486115, -224.4455414, 207.5684052
1: -110.5956345, 122.1934662, -60.8538132, 78.0529633, -188.6485901, 183.0472717
2: -160.0959167, 135.8744354, -88.3261261, 88.1764526, -248.2723389, 224.2005463
3: -66.5068054, 161.4639130, -42.8978882, 92.7433777, -159.2501831, 204.3617859
4: -178.3852386, 134.9893646, -98.8100128, 87.4425964, -265.8278198, 233.7993774

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.2427821, upper bound: 187.5313408
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4424378, upper bound: 187.5369604
time: 0.61 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -92.8833923, 92.3974380, -237.2066956, 224.5424347
1: -113.4549103, 124.2506790, -72.5469284, 86.5662308, -200.0211487, 196.7976074
2: -164.1941833, 138.0722961, -105.1364594, 97.2733383, -261.4675293, 243.2087555
3: -67.5839310, 165.3264618, -47.0671234, 108.7951965, -176.3791199, 212.3935852
4: -182.9160309, 137.3219604, -117.3885803, 96.9032516, -279.8192444, 254.7104797

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7185044, upper bound: 187.5436577
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6907133, upper bound: 187.5416841
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -90.0296860, 95.7459259, -184.3536530, 182.8400574
1: -69.5185242, 87.8645935, -70.5223999, 90.7102203, -160.2287445, 158.3869629
2: -100.9003143, 97.3328018, -102.4902802, 100.3906708, -201.2909851, 199.8230896
3: -47.0367928, 106.5991135, -48.7527466, 108.2179565, -155.2547455, 155.3518372
4: -113.0341187, 96.2283783, -114.9058533, 99.2772675, -212.3113861, 211.1342163

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6067470, upper bound: 187.5187196
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5272318
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -144.1452942, 131.1593170, -90.0296860, 95.7459259, -239.8912201, 221.1889954
1: -112.8837967, 123.8181610, -70.5223999, 90.7102203, -203.5940094, 194.3405609
2: -163.4188080, 137.6029663, -102.4902802, 100.3906708, -263.8094788, 240.0932465
3: -67.3246002, 164.6031952, -48.7527466, 108.2179565, -175.5425415, 213.3559265
4: -182.0937500, 136.7665100, -114.9058533, 99.2772675, -281.3710022, 251.6723480

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3715287, upper bound: 187.6259690
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234969, upper bound: 187.6258461
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -148.9440002, 134.4782104, -135.3423920, 128.6092987, -277.5532532, 269.8206177
1: -116.7244568, 126.9229736, -105.9854965, 121.5867310, -238.3111877, 232.9084320
2: -168.8512268, 140.9629364, -153.5312347, 135.1462860, -303.9974976, 294.4941406
3: -68.9539795, 169.8156586, -66.2919998, 156.0335083, -224.9874878, 236.1076660
4: -188.0950775, 140.3184204, -171.4211121, 134.0570679, -322.1521606, 311.7394714

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4322450, upper bound: 187.5404742
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4312241, upper bound: 187.5612370
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350506, upper bound: 187.6285091
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -149.0803223, 134.5512085, -142.5714111, 132.4927673, -281.5730896, 277.1226196
1: -116.8296204, 126.9912109, -111.4859314, 125.1495209, -241.9791412, 238.4771423
2: -169.0030518, 141.0368652, -161.5406494, 139.0050201, -308.0080566, 302.5774841
3: -68.9912720, 169.9507294, -68.2609253, 163.1841278, -232.1753998, 238.2116547
4: -188.2618713, 140.3968811, -180.2251587, 138.1768951, -326.4387817, 320.6219788

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290607, upper bound: 187.4672258
time: 0.77 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6276910
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -142.9301300, 136.5297241, -40.9809875, 54.7791138, -197.7092438, 176.2460175
1: -112.1072311, 129.6569214, -32.1188278, 51.3675804, -163.4748077, 159.8806000
2: -162.6166534, 142.5903015, -47.0364113, 57.8066292, -220.4232483, 187.4566345
3: -69.5167236, 165.8107758, -27.2729225, 55.0080185, -122.6167221, 193.0836792
4: -181.6892090, 141.3737183, -53.0424118, 57.1621284, -238.8513336, 193.1997070

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3903753, upper bound: 187.6238557
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7948078, upper bound: 187.6236672
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.6309509, 137.2387695, -89.4479752, 89.8819046, -233.5128479, 225.6481171
1: -112.6476974, 130.3599548, -69.8412399, 84.1715393, -196.8192139, 198.3559113
2: -163.4073792, 143.3272705, -101.2739258, 94.6007919, -258.0081787, 242.3435669
3: -69.8844528, 166.5899963, -45.8811989, 105.0105896, -172.8058014, 212.4711914
4: -182.5763702, 142.0978088, -113.0568390, 94.2403870, -276.8167114, 253.9522858

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7941578, upper bound: 187.4088854
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6282967
time: 0.60 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -205.4258423, 179.6349640, -78.1125488, 83.2486115, -288.6744385, 257.7474976
1: -161.3577423, 169.9179230, -60.8538132, 78.0529633, -239.4107056, 230.7717285
2: -233.0604553, 187.5063477, -88.3261261, 88.1764526, -321.2369080, 275.8324585
3: -92.4965973, 231.0563660, -42.8978882, 92.7433777, -184.3302002, 273.9541931
4: -259.4066772, 187.2655945, -98.8100128, 87.4425964, -346.8491821, 286.0756226

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3485907, upper bound: 187.5855679
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4442422, upper bound: 187.5853066
time: 0.71 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -209.3486176, 181.9826050, -92.8833923, 92.3974380, -301.7460632, 274.8659363
1: -164.4173126, 172.1151276, -72.5469284, 86.5662308, -250.9835510, 244.6620331
2: -237.5040588, 189.8574982, -105.1364594, 97.2733383, -334.7773743, 294.9939575
3: -93.6510162, 235.2007599, -47.0671234, 108.7951965, -201.7828674, 282.2678223
4: -264.3345032, 189.7580109, -117.3885803, 96.9032516, -361.2377319, 307.1465759

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4911821, upper bound: 187.6295918
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7942499, upper bound: 187.6286455
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -90.0296860, 95.7459259, -239.7033997, 226.6428986
1: -112.9016113, 130.6958466, -70.5223999, 90.7102203, -203.6118011, 199.4726105
2: -163.7802124, 143.6897125, -102.4902802, 100.3906708, -264.1708984, 244.2701569
3: -70.0736237, 166.9585724, -48.7527466, 108.2179565, -176.3800354, 215.7113037
4: -182.9930725, 142.4557343, -114.9058533, 99.2772675, -282.2703247, 256.5001831

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6097132, upper bound: 187.6166270
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255915
time: 0.70 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -146.0465698, 134.7337646, -278.6912231, 282.8958740
1: -112.9016113, 130.6958466, -114.2232513, 127.2852173, -240.1868134, 243.3010559
2: -163.7802124, 143.6897125, -165.4884644, 141.3374176, -305.1176147, 307.2244568
3: -70.0736237, 166.9585724, -69.3965378, 166.8359833, -234.8969727, 236.3424835
4: -182.9930725, 142.4557343, -184.5755615, 140.5543518, -323.5474243, 326.2516174

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6120622, upper bound: 187.4010512
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255824
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -210.1392517, 182.9332123, -139.0402222, 131.1075745, -341.2468262, 321.9734192
1: -165.0560150, 173.0409851, -108.8877182, 123.9752121, -289.0312195, 281.9286499
2: -238.4053040, 190.8639221, -157.6835480, 137.7197266, -376.1250305, 348.5474854
3: -94.1684799, 236.1148987, -67.5197983, 160.0581207, -253.4903564, 303.6347046
4: -265.3433228, 190.7401733, -176.0632782, 136.6995392, -402.0428467, 366.8034058

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4398745, upper bound: 187.4965127
time: 0.62 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6286043
time: 0.79 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -146.3229065, 134.9822693, -345.2630005, 329.3304443
1: -165.1631775, 173.1105804, -114.4310303, 127.5312195, -292.6943359, 287.5415955
2: -238.5625763, 190.9392853, -165.7576294, 141.5741119, -380.1366577, 356.6968994
3: -94.2064362, 236.2539673, -69.4877243, 167.2463074, -260.8176575, 305.7416687
4: -265.5162659, 190.8201752, -184.9295807, 140.8143768, -406.3306274, 375.7497559

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6335490, upper bound: 187.4965122
time: 0.64 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6335490, upper bound: 187.6286043
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -88.6077271, 92.8103714, -133.7913513, 143.3868256
1: -32.1188278, 51.3675804, -69.5185242, 87.8645935, -119.9834061, 120.8861008
2: -47.0364113, 57.8066292, -100.9003143, 97.3328018, -144.3691864, 158.7069397
3: -27.2729225, 55.0080185, -47.0367928, 106.5991135, -133.8720245, 102.0447998
4: -53.0424118, 57.1621284, -113.0341187, 96.2283783, -149.2707672, 170.1962280

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5055142, upper bound: 187.3872464
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5044697, upper bound: 187.7947934
time: 0.82 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -88.6077271, 92.8103714, -182.2582855, 178.4896240
1: -69.8412399, 84.1715393, -69.5185242, 87.8645935, -157.7057953, 153.6900330
2: -101.2739258, 94.6007919, -100.9003143, 97.3328018, -198.6067200, 195.5010986
3: -45.8811989, 105.0105896, -47.0367928, 106.5991135, -152.4803009, 152.0473785
4: -113.0568390, 94.2403870, -113.0341187, 96.2283783, -209.2852020, 207.2745056

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5172433, upper bound: 187.7577321
time: 0.59 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7911118
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -78.1125488, 83.2486115, -141.1969452, 129.4559174, -207.5684052, 224.4455414
1: -60.8538132, 78.0529633, -110.5956345, 122.1934662, -183.0472717, 188.6485901
2: -88.3261261, 88.1764526, -160.0959167, 135.8744354, -224.2005615, 248.2723694
3: -42.8978882, 92.7433777, -66.5068054, 161.4639130, -204.3617859, 159.2501831
4: -98.8100128, 87.4425964, -178.3852386, 134.9893646, -233.7993469, 265.8277893

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5313408, upper bound: 187.2427821
time: 0.67 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5369604, upper bound: 187.4424378
time: 0.73 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -92.8833923, 92.3974380, -144.8092651, 131.6590424, -224.5424347, 237.2066956
1: -72.5469284, 86.5662308, -113.4549103, 124.2506790, -196.7976074, 200.0211487
2: -105.1364594, 97.2733383, -164.1941833, 138.0722961, -243.2087555, 261.4675293
3: -47.0671234, 108.7951965, -67.5839310, 165.3264618, -212.3935852, 176.3791199
4: -117.3885803, 96.9032516, -182.9160309, 137.3219604, -254.7105103, 279.8192749

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5436577, upper bound: 187.7185044
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5416841, upper bound: 187.6907133
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -144.1452942, 131.1593170, -88.6077271, 92.8103714, -236.9556580, 219.7670441
1: -112.8837967, 123.8181610, -69.5185242, 87.8645935, -200.7483521, 193.3366852
2: -163.4188080, 137.6029663, -100.9003143, 97.3328018, -260.7515869, 238.5032806
3: -67.3246002, 164.6031952, -47.0367928, 106.5991135, -173.9237061, 211.6399689
4: -182.0937500, 136.7665100, -113.0341187, 96.2283783, -278.3220215, 249.8006287

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3635642, upper bound: 187.6253392
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5040416, upper bound: 187.6252001
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -144.7948914, 131.8384247, -127.6571198, 121.2996750, -266.0945740, 259.4955444
1: -113.3896179, 124.4891510, -99.7887268, 114.5272980, -227.9168854, 224.2778778
2: -164.1222992, 138.3610077, -144.5712585, 127.7422256, -291.8645325, 282.9322205
3: -67.6604538, 165.3612823, -62.6627769, 146.6948853, -214.3553467, 228.0240631
4: -182.9035950, 137.4911194, -161.3106079, 126.4780350, -309.3816223, 298.8016968

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_B2_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3286681, upper bound: 187.5770821
time: 0.60 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_B1_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4167922, upper bound: 187.5769901
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -148.2934875, 133.9195099, -144.0959778, 131.2398987, -279.5333557, 278.0155029
1: -116.1464539, 126.4533005, -112.8738327, 123.8645172, -240.0109406, 239.3271179
2: -168.0833893, 140.4526978, -163.3707428, 137.6708374, -305.7542114, 303.8233948
3: -68.6626892, 169.0866852, -67.3830719, 164.5631714, -233.2258606, 236.4697418
4: -187.2937012, 139.6888123, -182.0305176, 136.8935242, -324.1872253, 321.7193298

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4648163, upper bound: 187.6288050
time: 0.74 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279392, upper bound: 187.6278758
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.59 seconds
IS_B1_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.8124576, upper bound: 187.3877039
IS_B1_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.8126227, upper bound: 187.8121031
IS_B1_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.8162422, upper bound: 187.7619780
IS_B1_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.8181337, upper bound: 187.8199886
IS_B1_A1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6513821, upper bound: 187.4440515
IS_B1_A1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6507981, upper bound: 187.4420780
IS_B1_A1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.8000586, upper bound: 187.7248378
IS_B1_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7238876, upper bound: 187.7238874
IS_B1_A1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6241148, upper bound: 187.3879327
IS_B1_A1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6239264, upper bound: 187.7953593
IS_B1_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4195676, upper bound: 187.7964483
IS_B1_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6258260, upper bound: 187.7987580
IS_B1_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6123496, upper bound: 187.4827854
IS_B1_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6299725, upper bound: 187.4870612
IS_B1_A1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6248659, upper bound: 187.7582411
IS_B1_A1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6290239, upper bound: 187.7901784
IS_B1_A1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3877502, upper bound: 187.6241148
IS_B1_A1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6239264
IS_B1_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7964483, upper bound: 187.4195676
IS_B1_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7987580, upper bound: 187.6285551
IS_B1_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4827854, upper bound: 187.6123496
IS_B1_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4870612, upper bound: 187.6299725
IS_B1_A1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7582411, upper bound: 187.6248659
IS_B1_A1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7901784, upper bound: 187.6290239
IS_B1_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6183427, upper bound: 187.6072655
IS_B1_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6259777, upper bound: 187.6258351
IS_B1_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6117348, upper bound: 187.4141545
IS_B1_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6259690, upper bound: 187.6258260
IS_B1_A1_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6123835, upper bound: 187.4314815
IS_B1_A1_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350266
IS_B1_A1_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
IS_B1_A1_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
IS_B1_A2_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3872464, upper bound: 187.5055142
IS_B1_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7947934, upper bound: 187.5044697
IS_B1_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7577321, upper bound: 187.5230579
IS_B1_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7911118, upper bound: 187.5293691
IS_B1_A2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.2427821, upper bound: 187.5313408
IS_B1_A2_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4424378, upper bound: 187.5369604
IS_B1_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7185044, upper bound: 187.5436577
IS_B1_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6907133, upper bound: 187.5416841
IS_B1_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6067470, upper bound: 187.5187196
IS_B1_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5272318
IS_B1_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3715287, upper bound: 187.6259690
IS_B1_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6234969, upper bound: 187.6258461
IS_B1_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4312241, upper bound: 187.5612370
IS_B1_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4350506, upper bound: 187.6285091
IS_B1_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6290607, upper bound: 187.4672258
IS_B1_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6276910
IS_B1_A2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3903753, upper bound: 187.6238557
IS_B1_A2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7948078, upper bound: 187.6236672
IS_B1_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7941578, upper bound: 187.4088854
IS_B1_A2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7982443, upper bound: 187.6282967
IS_B1_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3485907, upper bound: 187.5855679
IS_B1_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4442422, upper bound: 187.5853066
IS_B1_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4911821, upper bound: 187.6295918
IS_B1_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.7942499, upper bound: 187.6286455
IS_B1_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6097132, upper bound: 187.6166270
IS_B1_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255915
IS_B1_A2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6120622, upper bound: 187.4010512
IS_B1_A2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6282939, upper bound: 187.6255824
IS_B1_A2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4398745, upper bound: 187.4965127
IS_B1_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6286043
IS_B1_A2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6335490, upper bound: 187.4965122
IS_B1_A2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6335490, upper bound: 187.6286043
IS_B2_A1_B1_A1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5055142, upper bound: 187.3872464
IS_B2_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5044697, upper bound: 187.7947934
IS_B2_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5172433, upper bound: 187.7577321
IS_B2_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5270964, upper bound: 187.7911118
IS_B2_A1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5313408, upper bound: 187.2427821
IS_B2_A1_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5369604, upper bound: 187.4424378
IS_B2_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5436577, upper bound: 187.7185044
IS_B2_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5416841, upper bound: 187.6907133
IS_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3635642, upper bound: 187.6253392
IS_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.5040416, upper bound: 187.6252001
IS_B2_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.3286681, upper bound: 187.5770821
IS_B2_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4167922, upper bound: 187.5769901
IS_B2_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.4648163, upper bound: 187.6288050
IS_B2_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.59
Output dim: 3, lower bound: -187.6279392, upper bound: 187.6278758
IS_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
IS_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6255764, upper bound: 187.7982443
IS_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5855680, upper bound: 187.4509566
IS_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6305415, upper bound: 187.7982106
IS_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6250977, upper bound: 187.5272318
IS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6250977, upper bound: 187.6269352
IS_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6305780, upper bound: 187.6293981
IS_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5272318, upper bound: 187.6253563
IS_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5270964, upper bound: 187.6286071
IS_B2_A2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6285091, upper bound: 187.4350550
IS_B2_A2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6276910, upper bound: 187.6290606
IS_B2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6253563
IS_B2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6257190, upper bound: 187.6274614
IS_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6292600, upper bound: 187.4350550
IS_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6284903, upper bound: 187.6290606
IS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6250977
IS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6282974
IS_B2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.5874588, upper bound: 187.4825839
IS_B2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6361837, upper bound: 187.6305415
IS_B2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6252249
IS_B2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6280352, upper bound: 187.6272877
IS_B2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.4892789, upper bound: 187.5892709
IS_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6303795
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7833230, upper bound: 187.6676341
time: 0.79 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6687624, upper bound: 187.6687624
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.60 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 3, lower bound: -187.7833230, upper bound: 187.6676341
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 3, lower bound: -187.6687624, upper bound: 187.6687624

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -146.9069672, 125.0689240, -119.9573212, 108.7468872, -255.6538391, 245.0262451
1: -115.1647415, 116.8796234, -93.8448410, 101.7164536, -216.8811951, 210.7244568
2: -166.5767059, 129.9570160, -135.7941437, 113.6401825, -280.2168884, 265.7510986
3: -62.5208206, 166.3341980, -54.4526558, 137.4407501, -199.9615631, 220.7868195
4: -185.2062378, 131.6817169, -151.2667084, 114.1601410, -299.3663330, 282.9483643

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
time: 0.62 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
time: 0.63 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -147.2906494, 125.2701874, -182.1602936, 153.6605072, -300.9511414, 307.4304810
1: -115.4979095, 117.0591202, -143.0398865, 144.6755829, -260.1734009, 260.0989990
2: -167.0516052, 130.1336060, -206.5966339, 159.9870605, -327.0386658, 336.7301941
3: -62.6109695, 166.8166809, -78.3434219, 204.8266907, -267.4376526, 245.1600952
4: -185.7081909, 131.9127808, -229.7526093, 160.7959442, -346.5041504, 361.6653442

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271248, upper bound: 187.6287476
time: 0.72 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294621, upper bound: 187.6294621
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.28 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6271248, upper bound: 187.6287476
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6294621, upper bound: 187.6294621

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7825379, upper bound: 187.6622823
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7825379, upper bound: 187.6673384
time: 0.57 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -180.2966766, 152.7804108, -119.9573212, 108.7468872, -289.0435181, 272.7377319
1: -141.5775146, 143.8618774, -93.8448410, 101.7164536, -243.2939301, 237.7067261
2: -204.5590515, 159.1097717, -135.7941437, 113.6401825, -318.1992188, 294.9038391
3: -77.8814087, 203.1510773, -54.4526558, 137.4407501, -215.3221588, 257.6037292
4: -227.4683685, 159.8488617, -151.2667084, 114.1601410, -341.6284180, 311.1155396

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7791787, upper bound: 187.6508823
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7826767, upper bound: 187.6673384
time: 0.87 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -117.3231201, 107.1182022, -180.3580475, 152.6167297, -269.9398499, 287.4762268
1: -91.8556595, 100.1594772, -141.6127472, 143.7078400, -235.5635071, 241.7722168
2: -132.8888245, 111.9686050, -204.5517120, 158.9521179, -291.8409119, 316.5203247
3: -53.6608925, 134.8885803, -77.8298187, 202.9295044, -256.5903625, 212.7183990
4: -148.0039673, 112.4096527, -227.4966125, 159.6778717, -307.6817932, 339.9062195

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264239, upper bound: 187.6264239
time: 0.74 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264239, upper bound: 187.6264239
time: 0.77 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -181.9570312, 155.6326904, -181.0826416, 153.0663910, -335.0234070, 336.7153320
1: -142.8092957, 146.6803131, -142.1895752, 144.1121368, -286.9214478, 288.8698730
2: -206.3678589, 162.1219635, -205.3767395, 159.3802795, -365.7481079, 367.4985962
3: -79.5327072, 204.9821930, -78.0478210, 203.7300568, -283.2627563, 283.0299988
4: -229.6176300, 162.7950745, -228.4169769, 160.1762390, -389.7938232, 391.2120361

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6251496, upper bound: 187.6263104
time: 0.58 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6290606
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.13 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.7825379, upper bound: 187.6622823
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.7825379, upper bound: 187.6673384
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.7791787, upper bound: 187.6508823
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.7826767, upper bound: 187.6673384
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.6264239, upper bound: 187.6264239
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.6264239, upper bound: 187.6264239
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.6251496, upper bound: 187.6263104
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6290606

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -96.7168961, 90.5865326, -62.5989532, 68.6813965, -165.3982849, 153.1854858
1: -75.6750565, 84.9021606, -49.0822411, 64.2613220, -139.9363708, 133.9843903
2: -109.6172333, 94.5198593, -71.3749008, 71.9796066, -181.5967865, 165.8947601
3: -44.7764587, 112.7566147, -33.6786270, 77.7695847, -122.5460434, 146.4352417
4: -122.1720047, 95.0657730, -79.9356384, 71.8462219, -194.0182190, 175.0014038

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247612, upper bound: 187.7290234
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -117.9716415, 107.3318253, -115.7513199, 105.7508011, -223.7224426, 223.0831299
1: -92.2776642, 100.3613434, -90.5305862, 98.8567810, -191.1344452, 190.8919373
2: -133.5452423, 112.1558685, -131.0461578, 110.5165710, -244.0617981, 243.2020264
3: -53.7486954, 135.2761688, -52.9895172, 132.8755493, -186.6242371, 188.2656860
4: -148.7582245, 112.6418839, -145.9682770, 110.9683304, -259.7265625, 258.6101685

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272868
time: 0.77 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272885, upper bound: 187.6272885
time: 0.64 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -115.9623871, 108.1399078, -96.7168961, 90.5865326, -206.5489197, 204.8568115
1: -91.0345230, 102.2205048, -75.6750565, 84.9021606, -175.9366302, 177.8955536
2: -131.9047089, 112.9005203, -109.6172333, 94.5198593, -226.4245300, 222.5177002
3: -54.4819260, 135.4607697, -44.7764587, 112.7566147, -167.2385406, 180.2372284
4: -147.2206879, 112.4841080, -122.1720047, 95.0657730, -242.2864532, 234.6561127

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7288709, upper bound: 187.6245032
time: 0.60 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255261, upper bound: 187.6243781
time: 0.77 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -176.0116882, 149.7702179, -117.9716415, 107.3318253, -283.3435059, 267.7418518
1: -138.1857758, 140.9952545, -92.2776642, 100.3613434, -238.5471191, 233.2729187
2: -199.7132263, 156.0321960, -133.5452423, 112.1558685, -311.8690796, 289.5774536
3: -76.4058685, 198.4794769, -53.7486954, 135.2761688, -211.6820374, 252.2281647
4: -222.0680847, 156.6505585, -148.7582245, 112.6418839, -334.7099609, 305.4087830

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
time: 0.63 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269199
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -117.3231201, 107.1182022, -149.2839355, 134.6893616, -252.0124359, 256.4021301
1: -91.8556595, 100.1594772, -116.9905624, 127.1221771, -218.9778137, 217.1500244
2: -132.8888245, 111.9686050, -169.2326202, 141.1778259, -274.0665894, 281.2012329
3: -53.6608925, 134.8885803, -69.0598907, 170.1712341, -223.8321228, 203.9484558
4: -148.0039673, 112.4096527, -188.5170746, 140.5437622, -288.5476990, 300.9267273

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272371
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262339, upper bound: 187.6262340
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -117.3231201, 107.1182022, -214.4000854, 185.6035156, -302.9266357, 321.5182800
1: -91.8556595, 100.1594772, -168.4353027, 175.5736237, -267.4292297, 268.5947876
2: -132.8888245, 111.9686050, -243.1842499, 193.5875549, -326.4762878, 355.1528625
3: -53.6608925, 134.8885803, -95.4519653, 240.6518707, -294.3127441, 229.8116760
4: -148.0039673, 112.4096527, -270.6673279, 193.5914612, -341.5953979, 383.0769653

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
time: 0.66 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6262339
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -116.2179794, 110.3096542, -154.5010071, 132.2512207, -248.4692078, 264.8106689
1: -91.1533356, 104.4107285, -121.2619781, 124.8456345, -215.9989624, 225.6726990
2: -132.1910248, 115.2093506, -175.3841553, 137.6223145, -269.8133545, 290.5935059
3: -55.7961807, 136.0564728, -66.7525558, 175.5458069, -231.3419800, 202.8090210
4: -147.6615906, 114.7121353, -195.1746216, 138.0774231, -285.7390137, 309.8866882

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244727, upper bound: 187.6263104
time: 0.68 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244727, upper bound: 187.6244066
time: 0.72 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -177.8385162, 152.7772522, -179.0215607, 151.6331177, -329.4716187, 331.7987976
1: -139.5541077, 143.9602661, -140.5600433, 142.7447815, -282.2988892, 284.5202942
2: -201.7189484, 159.2003784, -203.0456085, 157.9212799, -359.6401978, 362.2459717
3: -78.1259003, 200.5173645, -77.3501740, 201.4948730, -279.6207886, 277.8674927
4: -224.4423523, 159.7646484, -225.8204041, 158.6561432, -383.0985107, 385.5850525

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6290606
time: 0.72 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6269199
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.28 seconds
IS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6247612, upper bound: 187.7290234
IS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
IS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272868
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6272885, upper bound: 187.6272885
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.7288709, upper bound: 187.6245032
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6255261, upper bound: 187.6243781
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269199
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272371
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6262339, upper bound: 187.6262340
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6262339
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6244727, upper bound: 187.6263104
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6244727, upper bound: 187.6244066
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6290606
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6269199

## BFS IS instance: IS_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -70.8088837, 74.5321655, -61.3601875, 67.9426346, -138.7515259, 135.8923035
1: -55.3257141, 69.9133606, -48.1152992, 63.5708046, -118.8965073, 118.0286560
2: -80.3077316, 78.2066879, -69.9924850, 71.2292633, -151.5369873, 148.1991730
3: -37.1244087, 85.3472137, -33.3106194, 76.4981613, -113.6225662, 118.6578369
4: -89.7878342, 78.0283203, -78.4040070, 71.0635452, -160.8513641, 156.4323273

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248027
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
time: 0.61 seconds

## BFS IS instance: IS_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -125.1854477, 117.4320831, -61.5466385, 68.0616608, -193.2471008, 178.9786987
1: -97.9450531, 111.1974106, -48.2574577, 63.6742630, -161.6193237, 159.4548492
2: -141.9799957, 122.8201141, -70.2038116, 71.3310165, -213.3110046, 193.0239258
3: -59.8747673, 144.6909485, -33.3431396, 76.7382965, -136.6130676, 178.0340881
4: -158.4964294, 122.1660309, -78.6394806, 71.1897736, -229.6862030, 200.8054962

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248027
time: 0.74 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
time: 0.67 seconds

## BFS IS instance: IS_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -116.4860992, 106.4420319, -89.4479752, 89.8819046, -206.3679962, 195.8899536
1: -91.1109161, 99.5331268, -69.8412399, 84.1715393, -175.2824402, 169.3743591
2: -131.8602600, 111.2617874, -101.2739258, 94.6007919, -226.4610596, 212.5357056
3: -53.3435211, 133.7024841, -45.8811989, 105.0105896, -158.3541107, 179.5836792
4: -146.8954315, 111.6957550, -113.0568390, 94.2403870, -241.1358185, 224.7525482

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -116.4130402, 106.4331207, -146.0465698, 134.7337646, -251.1468048, 252.4796753
1: -91.0520401, 99.4962082, -114.2232513, 127.2852173, -218.3372345, 213.7194519
2: -131.7877502, 111.2180710, -165.4884644, 141.3374176, -273.1251831, 276.7065430
3: -53.2919350, 133.6934509, -69.3965378, 166.8359833, -220.1279144, 203.0899963
4: -146.8158569, 111.7104645, -184.5755615, 140.5543518, -287.3702087, 296.2859497

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272885
time: 0.78 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272885
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -114.4460907, 107.2869339, -70.8088837, 74.5321655, -188.9781799, 178.0958252
1: -89.8446274, 101.4188843, -55.3257141, 69.9133606, -159.7579803, 156.7445984
2: -130.1825104, 112.0322952, -80.3077316, 78.2066879, -208.3891907, 192.3400269
3: -54.0610733, 133.8537292, -37.1244087, 85.3472137, -139.4082947, 170.9781189
4: -145.3224640, 111.5741348, -89.7878342, 78.0283203, -223.3507690, 201.3619690

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240133, upper bound: 187.5268282
time: 0.61 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240133, upper bound: 187.6243781
time: 0.62 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -115.2381210, 107.7372513, -125.2193146, 117.4710617, -232.7091827, 232.9565735
1: -90.4602051, 101.8276367, -97.9710541, 111.2349319, -201.6951294, 199.7986908
2: -131.0829163, 112.4798965, -142.0182037, 122.8610458, -253.9439697, 254.4981079
3: -54.2834740, 134.7241516, -59.8991852, 144.7281342, -199.0116119, 194.6233215
4: -146.3206940, 112.0655975, -158.5394745, 122.2061996, -268.5268860, 270.6050720

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240133, upper bound: 187.5268282
time: 0.60 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6240133, upper bound: 187.6243781
time: 0.60 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -116.4860992, 106.4420319, -251.2512970, 248.1451416
1: -113.4549103, 124.2506790, -91.1109161, 99.5331268, -212.9880371, 215.3616028
2: -164.1941833, 138.0722961, -131.8602600, 111.2617874, -275.4559631, 269.9325562
3: -67.5839310, 165.3264618, -53.3435211, 133.7024841, -201.2864075, 218.6699829
4: -182.9160309, 137.3219604, -146.8954315, 111.6957550, -294.6117859, 284.2173157

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
time: 0.66 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -205.9465485, 180.1755219, -116.4130402, 106.4331207, -312.3796692, 296.5885315
1: -161.7383118, 170.4427795, -91.0520401, 99.4962082, -261.2345276, 261.4948120
2: -233.7561646, 188.0472260, -131.7877502, 111.2180710, -344.9742432, 319.8349609
3: -92.7118301, 232.0392609, -53.2919350, 133.6934509, -225.7503052, 285.3311768
4: -260.1350403, 187.8265533, -146.8158569, 111.7104645, -371.8454895, 334.6423950

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269179
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269199
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -60.7287216, 67.5609207, -123.6794357, 114.4391174, -175.1678162, 191.2403259
1: -47.6900101, 63.1847534, -96.9159088, 108.2451935, -155.9351959, 160.1006622
2: -69.3438416, 70.8050919, -140.3470459, 119.6642990, -189.0081329, 211.1521301
3: -33.0708771, 76.1239090, -58.0654373, 142.9347076, -176.0055542, 134.1893463
4: -77.6507797, 70.6399307, -156.5294342, 119.0380554, -196.6888428, 227.1693726

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272371
time: 0.60 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272075
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -113.0714035, 104.0583725, -147.2054901, 133.2802734, -246.3516846, 251.2638550
1: -88.5095215, 97.2490158, -115.3470154, 125.7866287, -214.2961426, 212.5960388
2: -128.0974579, 108.7795715, -166.8895874, 139.7389832, -267.8364258, 275.6691589
3: -52.1716537, 130.2787628, -68.3726730, 167.9203033, -220.0919189, 198.6514282
4: -142.6618195, 109.1446457, -185.9134521, 139.0450897, -281.7069092, 295.0581055

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262026, upper bound: 187.6262340
time: 0.65 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262026, upper bound: 187.6262027
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -94.2517395, 89.0168686, -143.3848114, 136.9977264, -230.6543427, 232.4016724
1: -73.8350754, 83.3316040, -112.4561691, 130.1258698, -202.4648895, 195.7877808
2: -106.8972321, 92.8226242, -163.1278534, 143.0783386, -248.4565125, 255.9504700
3: -43.9149857, 110.3977356, -69.7585449, 166.3150330, -210.2300110, 178.4607697
4: -119.1268005, 93.3897629, -182.2634430, 141.8511353, -260.5270691, 275.6531982

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245032, upper bound: 187.6243218
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245032, upper bound: 187.6236270
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -115.2825546, 105.6503296, -208.8561096, 181.4887848, -296.7713318, 314.5064392
1: -90.2471313, 98.7571869, -164.0234985, 171.6398926, -261.8870239, 262.7806702
2: -130.5821075, 110.4345856, -236.9441528, 189.3404694, -319.9225769, 347.3787231
3: -52.9391861, 132.6692352, -93.3867798, 234.6490936, -287.5882874, 225.4492798
4: -145.4324646, 110.8372040, -263.7104492, 189.2452698, -334.6777344, 374.5476685

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6283582
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6258430
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -154.5010071, 132.2512207, -222.2808990, 250.2469330
1: -70.5223999, 90.7102203, -121.2619781, 124.8456345, -195.3680115, 211.9721985
2: -102.4902802, 100.3906708, -175.3841553, 137.6223145, -240.1125946, 275.7748413
3: -48.7527466, 108.2179565, -66.7525558, 175.5458069, -224.2985229, 174.9705048
4: -114.9058533, 99.2772675, -195.1746216, 138.0774231, -252.9832764, 294.4519043

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6245132
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6245411
time: 0.76 seconds

## BFS IS instance: IS_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -143.9209900, 137.5663147, -154.5010071, 132.2512207, -276.1722107, 291.6904602
1: -112.8730698, 130.6749878, -121.2619781, 124.8456345, -237.7187042, 250.5402374
2: -163.7390137, 143.6699066, -175.3841553, 137.6223145, -301.3613281, 317.7254333
3: -70.0665512, 166.9166107, -66.7525558, 175.5458069, -244.0095673, 233.6691437
4: -182.9464722, 142.4316864, -195.1746216, 138.0774231, -321.0238953, 337.3763428

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6244066
time: 0.67 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6244066
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -146.5121613, 135.0994873, -179.0215607, 151.6331177, -298.1452332, 314.1210327
1: -114.6307831, 127.5979538, -140.5600433, 142.7447815, -257.3755493, 268.1579895
2: -166.0353241, 141.6752014, -203.0456085, 157.9212799, -323.9565735, 344.7207336
3: -69.5864105, 167.3457031, -77.3501740, 201.4948730, -271.0812683, 244.6958618
4: -185.1463776, 140.9630737, -225.8204041, 158.6561432, -343.8024902, 366.7834778

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269162
time: 0.65 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269629
time: 0.67 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -179.0215607, 151.6331177, -361.9138794, 362.0290222
1: -165.1631775, 173.1105804, -140.5600433, 142.7447815, -307.9078979, 313.6706238
2: -238.5625763, 190.9392853, -203.0456085, 157.9212799, -396.4838562, 393.9848938
3: -94.2064362, 236.2539673, -77.3501740, 201.4948730, -295.1039734, 313.6041260
4: -265.5162659, 190.8201752, -225.8204041, 158.6561432, -424.1724243, 416.6405640

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269162
time: 0.66 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269199
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.36 seconds
IS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248027
IS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
IS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248027
IS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6246367, upper bound: 187.6248497
IS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
IS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
IS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272885
IS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272885
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6240133, upper bound: 187.5268282
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6240133, upper bound: 187.6243781
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6240133, upper bound: 187.5268282
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6240133, upper bound: 187.6243781
IS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
IS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6272853, upper bound: 187.6258431
IS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269179
IS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6294224, upper bound: 187.6269199
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272371
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6228919, upper bound: 187.5272075
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6262026, upper bound: 187.6262340
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6262026, upper bound: 187.6262027
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6245032, upper bound: 187.6243218
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6245032, upper bound: 187.6236270
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6283582
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6258430
IS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6245132
IS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6245411
IS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6244066
IS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6244066
IS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269162
IS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269629
IS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269162
IS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6269199

## BFS IS instance: IS_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -70.8088837, 74.5321655, -40.9809875, 54.7791138, -125.5879898, 115.5131531
1: -55.3257141, 69.9133606, -32.1188278, 51.3675804, -106.6932907, 102.0321732
2: -80.3077316, 78.2066879, -47.0364113, 57.8066292, -138.1143494, 125.2431030
3: -37.1244087, 85.3472137, -27.2729225, 55.0080185, -92.1324310, 112.6201324
4: -89.7878342, 78.0283203, -53.0424118, 57.1621284, -146.9499512, 131.0706940

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
time: 0.58 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -70.8088837, 74.5321655, -88.7503510, 95.0967102, -165.9055939, 163.2824707
1: -55.3257141, 69.9133606, -69.4899979, 90.1278381, -145.4535522, 139.4033356
2: -80.3077316, 78.2066879, -100.9360809, 99.7855835, -180.0933075, 179.1427612
3: -37.1244087, 85.3472137, -48.5418625, 106.7330246, -143.8574371, 133.8890686
4: -89.7878342, 78.0283203, -113.2336960, 98.6521301, -188.4399719, 191.2619629

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -124.8847351, 117.0864487, -40.9809875, 54.7791138, -179.6638489, 158.0674438
1: -97.7140808, 110.8647995, -32.1188278, 51.3675804, -149.0816650, 142.9835968
2: -141.6406708, 122.4571228, -47.0364113, 57.8066292, -199.4472961, 169.4935303
3: -59.6583481, 144.3610077, -27.2729225, 55.0080185, -114.6663513, 171.6339264
4: -158.1141205, 121.8099518, -53.0424118, 57.1621284, -215.2762451, 174.8523407

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3727450, upper bound: 187.5305762
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6230269, upper bound: 187.6239043
time: 0.70 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -125.2193146, 117.4710617, -88.7503510, 95.0967102, -220.3160095, 206.2214050
1: -97.9710541, 111.2349319, -69.4899979, 90.1278381, -188.0988770, 180.7248993
2: -142.0182037, 122.8610458, -100.9360809, 99.7855835, -241.8037872, 223.7971191
3: -59.8991852, 144.7281342, -48.5418625, 106.7330246, -166.6321869, 193.2699585
4: -158.5394745, 122.2061996, -113.2336960, 98.6521301, -257.1915894, 235.4398346

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6226914
time: 0.71 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6248027
time: 0.63 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -91.5989838, 91.4505310, -89.4479752, 89.8819046, -181.4808655, 180.8984680
1: -71.5373001, 85.6562042, -69.8412399, 84.1715393, -155.7088318, 155.4974213
2: -103.6967926, 96.2342453, -101.2739258, 94.6007919, -198.2975769, 197.5081329
3: -46.6185226, 107.3869629, -45.8811989, 105.0105896, -151.6291199, 153.2681580
4: -115.7632828, 95.8914185, -113.0568390, 94.2403870, -210.0036621, 208.9482574

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7062345, upper bound: 187.6191941
time: 0.79 seconds

## Relational analysis of IS_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272856
time: 0.60 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -148.1500549, 136.1284180, -89.4479752, 89.8819046, -238.0319366, 225.5763855
1: -115.8733826, 128.6213989, -69.8412399, 84.1715393, -200.0449066, 198.4626312
2: -167.8495941, 142.7842712, -101.2739258, 94.6007919, -262.4503784, 244.0581818
3: -70.0773010, 169.1162109, -45.8811989, 105.0105896, -175.0878906, 214.9973755
4: -187.2107849, 142.0318451, -113.0568390, 94.2403870, -281.4511719, 255.0886841

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6613281, upper bound: 187.6067986
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7304526, upper bound: 187.6268213
time: 0.76 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -91.5989838, 91.4505310, -145.8793182, 134.5294037, -226.1283875, 237.3298492
1: -71.5373001, 85.6562042, -114.0937805, 127.0895996, -198.6268921, 199.7499542
2: -103.6967926, 96.2342453, -165.2985535, 141.1205444, -244.8173218, 261.5328064
3: -46.6185226, 107.3869629, -69.2801971, 166.6479492, -213.2664795, 176.6671600
4: -115.7632828, 95.8914185, -184.3617096, 140.3413849, -256.1046143, 280.2531128

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6191941, upper bound: 187.6094675
time: 0.65 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272856, upper bound: 187.6272873
time: 0.71 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -148.1500549, 136.1284180, -146.0465698, 134.7337646, -282.8838196, 282.1749878
1: -115.8733826, 128.6213989, -114.2232513, 127.2852173, -243.1585999, 242.8446350
2: -167.8495941, 142.7842712, -165.4884644, 141.3374176, -309.1870117, 308.2727356
3: -70.0773010, 169.1162109, -69.3965378, 166.8359833, -236.9132843, 238.5127563
4: -187.2107849, 142.0318451, -184.5755615, 140.5543518, -327.7651367, 326.6072998

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
time: 0.70 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265245, upper bound: 187.6265245
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -70.8088837, 74.5321655, -163.1398468, 163.6192627
1: -69.5185242, 87.8645935, -55.3257141, 69.9133606, -139.4318542, 143.1902924
2: -100.9003143, 97.3328018, -80.3077316, 78.2066879, -179.1069946, 177.6405182
3: -47.0367928, 106.5991135, -37.1244087, 85.3472137, -132.3839874, 143.7235260
4: -113.0341187, 96.2283783, -89.7878342, 78.0283203, -191.0624084, 186.0161896

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7259387, upper bound: 187.5255726
time: 0.66 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7259387, upper bound: 187.5272075
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -142.0965118, 135.9161377, -70.8088837, 74.5321655, -216.6286316, 205.9223633
1: -111.4415436, 129.0892944, -55.3257141, 69.9133606, -181.3548737, 182.8164673
2: -161.6604462, 141.9721985, -80.3077316, 78.2066879, -239.8671265, 220.5394745
3: -69.2105255, 164.8815460, -37.1244087, 85.3472137, -152.7509460, 202.0059509
4: -180.6218567, 140.7363281, -89.7878342, 78.0283203, -258.6501770, 229.8215637

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7259387, upper bound: 187.6223350
time: 0.67 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7259387, upper bound: 187.6245032
time: 0.75 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -125.2193146, 117.4710617, -206.0787964, 218.0296783
1: -69.5185242, 87.8645935, -97.9710541, 111.2349319, -180.7534180, 185.8356476
2: -100.9003143, 97.3328018, -142.0182037, 122.8610458, -223.7613525, 239.3509979
3: -47.0367928, 106.5991135, -59.8991852, 144.7281342, -191.7649231, 166.4982910
4: -113.0341187, 96.2283783, -158.5394745, 122.2061996, -235.2403107, 254.7678528

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6215760, upper bound: 187.5256570
time: 0.72 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6215760, upper bound: 187.5256570
time: 0.63 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -143.9209900, 137.5663147, -125.2193146, 117.4710617, -261.3920593, 262.1355591
1: -112.8730698, 130.6749878, -97.9710541, 111.2349319, -224.1080017, 227.0660553
2: -163.7390137, 143.6699066, -142.0182037, 122.8610458, -286.6000671, 284.0611267
3: -70.0665512, 166.9166107, -59.8991852, 144.7281342, -213.0203400, 226.8157654
4: -182.9464722, 142.4316864, -158.5394745, 122.2061996, -305.1526794, 300.4073792

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6215760, upper bound: 187.6223438
time: 0.61 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6215760, upper bound: 187.6243781
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -91.5989838, 91.4505310, -236.2597961, 223.2580261
1: -113.4549103, 124.2506790, -71.5373001, 85.6562042, -199.1111145, 195.7879791
2: -164.1941833, 138.0722961, -103.6967926, 96.2342453, -260.4284363, 241.7690582
3: -67.5839310, 165.3264618, -46.6185226, 107.3869629, -174.9708862, 211.9449768
4: -182.9160309, 137.3219604, -115.7632828, 95.8914185, -278.8074341, 253.0852051

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6093294, upper bound: 187.6151980
time: 0.65 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272846, upper bound: 187.6258431
time: 0.80 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -148.1500549, 136.1284180, -280.9376831, 279.8090820
1: -113.4549103, 124.2506790, -115.8733826, 128.6213989, -242.0763092, 240.1240540
2: -164.1941833, 138.0722961, -167.8495941, 142.7842712, -306.9784241, 305.9218750
3: -67.5839310, 165.3264618, -70.0773010, 169.1162109, -236.7001343, 235.4037628
4: -182.9160309, 137.3219604, -187.2107849, 142.0318451, -324.9478149, 324.5326538

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4597249
time: 0.73 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265237, upper bound: 187.6250970
time: 0.59 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -204.5228577, 178.8265076, -91.5989838, 91.4505310, -295.9733887, 270.4254761
1: -160.5974731, 169.1442566, -71.5373001, 85.6562042, -246.2536316, 240.6815491
2: -232.1378479, 186.6405792, -103.6967926, 96.2342453, -328.3721008, 290.3373108
3: -92.0014496, 230.4472809, -46.6185226, 107.3869629, -198.7447357, 277.0657654
4: -258.3312073, 186.4234772, -115.7632828, 95.8914185, -354.2226257, 302.1867676

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4394220, upper bound: 187.5899942
time: 0.68 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289531, upper bound: 187.6261656
time: 0.59 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -207.3404541, 181.6126862, -148.1500549, 136.1284180, -343.4688721, 329.7626648
1: -162.8531342, 171.8320618, -115.8733826, 128.6213989, -291.4744873, 287.7054443
2: -235.3392334, 189.5546875, -167.8495941, 142.7842712, -378.1235046, 357.4042053
3: -93.4821930, 233.6065369, -70.0773010, 169.1162109, -261.8392334, 303.6837463
4: -261.9020691, 189.3196564, -187.2107849, 142.0318451, -403.9338989, 376.5303955

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5937594, upper bound: 187.4952701
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289531, upper bound: 187.6261665
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -40.8391457, 54.6433601, -123.6794357, 114.4391174, -155.2782593, 178.3227997
1: -32.0079727, 51.2383080, -96.9159088, 108.2451935, -140.2531586, 148.1542206
2: -46.8787918, 57.6636047, -140.3470459, 119.6642990, -166.5430908, 198.0106506
3: -27.1843395, 54.8626633, -58.0654373, 142.9347076, -170.1190491, 112.9281006
4: -52.8647118, 57.0142860, -156.5294342, 119.0380554, -171.9027557, 213.5437164

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5358576, upper bound: 187.3647378
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6220018, upper bound: 187.5051509
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -86.7296219, 91.9936829, -123.6794357, 114.4391174, -201.1687317, 215.6730957
1: -68.0428391, 87.1070557, -96.9159088, 108.2451935, -176.2880249, 184.0229645
2: -98.7414093, 96.5650635, -140.3470459, 119.6642990, -218.4057007, 236.9120941
3: -46.7383194, 104.7160034, -58.0654373, 142.9347076, -189.6730347, 162.7814178
4: -110.6546783, 95.4268112, -156.5294342, 119.0380554, -229.6927338, 251.9562378

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5253691, upper bound: 187.5253691
time: 0.64 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5253691, upper bound: 187.5253691
time: 0.61 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -147.2054901, 133.2802734, -222.7282104, 237.0874023
1: -69.8412399, 84.1715393, -115.3470154, 125.7866287, -195.6278381, 199.5185547
2: -101.2739258, 94.6007919, -166.8895874, 139.7389832, -241.0128784, 261.4903870
3: -45.8811989, 105.0105896, -68.3726730, 167.9203033, -213.8014832, 173.3832703
4: -113.0568390, 94.2403870, -185.9134521, 139.0450897, -252.1019287, 280.1538391

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_A1_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4670488
time: 0.68 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_A2

### Relational analysis result of IS_B2_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254605, upper bound: 187.6254945
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -144.2667847, 131.2299805, -147.2054901, 133.2802734, -277.5470581, 278.4354858
1: -112.9824219, 123.8839874, -115.3470154, 125.7866287, -238.7690430, 239.2310028
2: -163.5588074, 137.6753693, -166.8895874, 139.7389832, -303.2977905, 304.5649414
3: -67.3619995, 164.7313690, -68.3726730, 167.9203033, -235.2823029, 233.1040344
4: -182.2491913, 136.8414307, -185.9134521, 139.0450897, -321.2942810, 322.7548828

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1_A2_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4671710
time: 0.63 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254605, upper bound: 187.6254605
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -70.8088837, 74.5321655, -142.0965118, 135.9161377, -205.9223785, 216.6286316
1: -55.3257141, 69.9133606, -111.4415436, 129.0892944, -182.8164673, 181.3548737
2: -80.3077316, 78.2066879, -161.6604462, 141.9721985, -220.5394897, 239.8671265
3: -37.1244087, 85.3472137, -69.2105255, 164.8815460, -202.0059204, 152.7509308
4: -89.7878342, 78.0283203, -180.6218567, 140.7363281, -229.8215637, 258.6501770

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6213253, upper bound: 187.5281009
time: 0.69 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6213253, upper bound: 187.6243218
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -123.2587738, 114.1208725, -143.9574738, 137.5898285, -260.2254028, 258.0783386
1: -96.5901871, 107.9249039, -112.9016113, 130.6958466, -225.7218628, 220.8264923
2: -139.8773193, 119.3313065, -163.7802124, 143.6897125, -281.9678345, 283.1115112
3: -57.8815842, 142.4792786, -70.0736237, 166.9585724, -224.8401489, 210.8069611
4: -156.0059509, 118.7006912, -182.9930725, 142.4557343, -297.9247437, 301.6937256

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6213253, upper bound: 187.5258906
time: 0.72 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6213253, upper bound: 187.6236270
time: 0.65 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -91.5989838, 91.4505310, -207.4080505, 180.1850586, -271.7840576, 298.8585815
1: -71.5373001, 85.6562042, -162.8629150, 170.3901978, -241.9274902, 248.5190887
2: -103.6967926, 96.2342453, -235.2981110, 187.9857788, -291.6825562, 331.5323486
3: -46.6185226, 107.3869629, -92.7059631, 233.0419922, -279.6605225, 199.4611511
4: -115.7632828, 95.8914185, -261.8776550, 187.8891296, -303.6523743, 357.7690735

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5833507, upper bound: 187.4709791
time: 0.70 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6278920
time: 0.69 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -146.5341949, 132.7420807, -210.2807465, 183.0075378, -329.5417480, 343.0228271
1: -114.7632294, 125.3279495, -165.1631775, 173.1105804, -287.8738098, 290.4911194
2: -166.1043243, 139.2408447, -238.5625763, 190.9392853, -357.0436096, 377.8033447
3: -68.0938110, 167.1837006, -94.2064362, 236.2539673, -304.3477783, 260.7029114
4: -185.0879517, 138.4415894, -265.5162659, 190.8201752, -375.9081421, 403.9578552

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5833507, upper bound: 187.4672622
time: 0.84 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -123.6794357, 114.4391174, -204.4688110, 219.4253540
1: -70.5223999, 90.7102203, -96.9159088, 108.2451935, -178.7675629, 187.6261292
2: -102.4902802, 100.3906708, -140.3470459, 119.6642990, -222.1545715, 240.7377167
3: -48.7527466, 108.2179565, -58.0654373, 142.9347076, -191.6874390, 166.2833557
4: -114.9058533, 99.2772675, -156.5294342, 119.0380554, -233.9439087, 255.8067017

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B2_A2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6153720, upper bound: 187.6061277
time: 0.65 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6247712
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -186.3793030, 163.9709625, -254.0006409, 282.1252441
1: -70.5223999, 90.7102203, -146.2506104, 155.4834290, -225.7991638, 236.9608154
2: -102.4902802, 100.3906708, -211.5796509, 170.7777863, -273.2680054, 311.9703369
3: -48.7527466, 108.2179565, -83.7342224, 210.9392853, -259.6920166, 190.9368286
4: -114.9058533, 99.2772675, -235.7194214, 170.4737244, -285.3795471, 334.9967041

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5350520, upper bound: 187.3730466
time: 0.74 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6227843, upper bound: 187.6231698
time: 0.62 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -143.9209900, 137.5663147, -123.6794357, 114.4391174, -258.3600464, 260.6297302
1: -112.8730698, 130.6749878, -96.9159088, 108.2451935, -221.1182556, 226.0312195
2: -163.7390137, 143.6699066, -140.3470459, 119.6642990, -283.4033203, 282.4245605
3: -70.0665512, 166.9166107, -58.0654373, 142.9347076, -211.2608032, 224.9820251
4: -182.9464722, 142.4316864, -156.5294342, 119.0380554, -301.9844971, 298.4325256

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6213253
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A1_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6213253
time: 0.65 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -143.9209900, 137.5663147, -186.3793030, 163.9709625, -307.8919373, 323.6213684
1: -112.8730698, 130.6749878, -146.2506104, 155.4834290, -268.3498840, 275.5286560
2: -163.7390137, 143.6699066, -211.5796509, 170.7777863, -334.5167847, 353.8685913
3: -70.0665512, 166.9166107, -83.7342224, 210.9392853, -279.3730164, 249.8205261
4: -182.9464722, 142.4316864, -235.7194214, 170.4737244, -353.4201965, 377.8632202

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A2_A1_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6223351
time: 0.68 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6244066
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -146.5121613, 135.0994873, -147.2054901, 133.2802734, -279.7924194, 282.3049927
1: -114.6307831, 127.5979538, -115.3470154, 125.7866287, -240.4174194, 242.9449768
2: -166.0353241, 141.6752014, -166.8895874, 139.7389832, -305.7742920, 308.5647888
3: -69.5864105, 167.3457031, -68.3726730, 167.9203033, -237.5066681, 235.7183838
4: -185.1463776, 140.9630737, -185.9134521, 139.0450897, -324.1914673, 326.8765259

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250970, upper bound: 187.6265236
time: 0.66 seconds

## BFS IS instance: IS_B2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -146.5121613, 135.0994873, -212.6791992, 184.6053772, -331.1174927, 347.7786865
1: -114.6307831, 127.5979538, -167.0668182, 174.6245880, -289.2553711, 294.6647644
2: -166.0353241, 141.6752014, -241.2538605, 192.5746002, -358.6099243, 382.9290466
3: -69.5864105, 167.3457031, -94.9916992, 238.8250122, -308.4114380, 261.5038452
4: -185.1463776, 140.9630737, -268.5172729, 192.5163574, -377.6627197, 409.4803467

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
time: 0.63 seconds

## Relational analysis of IS_B2_A2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250970, upper bound: 187.6265236
time: 0.77 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -147.2054901, 133.2802734, -343.5610352, 330.2130127
1: -165.1631775, 173.1105804, -115.3470154, 125.7866287, -290.9497375, 288.4575806
2: -238.5625763, 190.9392853, -166.8895874, 139.7389832, -378.3015747, 357.8288574
3: -94.2064362, 236.2539673, -68.3726730, 167.9203033, -261.4416504, 304.6265869
4: -265.5162659, 190.8201752, -185.9134521, 139.0450897, -404.5613403, 376.7336121

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4710047, upper bound: 187.5935476
time: 0.64 seconds

## Relational analysis of IS_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278920, upper bound: 187.6261644
time: 0.70 seconds

## BFS IS instance: IS_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -212.6791992, 184.6053772, -394.8861084, 395.6867371
1: -165.1631775, 173.1105804, -167.0668182, 174.6245880, -339.7877808, 340.1773987
2: -238.5625763, 190.9392853, -241.2538605, 192.5746002, -431.1371765, 432.1931458
3: -94.2064362, 236.2539673, -94.9916992, 238.8250122, -332.4508972, 330.5827637
4: -265.5162659, 190.8201752, -268.5172729, 192.5163574, -458.0326233, 459.3374634

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6003024, upper bound: 187.4961552
time: 0.69 seconds

## Relational analysis of IS_B2_A2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278921, upper bound: 187.6261665
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.48 seconds
IS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
IS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
IS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
IS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
IS_B1_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.3727450, upper bound: 187.5305762
IS_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6230269, upper bound: 187.6239043
IS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6226914
IS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6248027
IS_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7062345, upper bound: 187.6191941
IS_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272856
IS_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6613281, upper bound: 187.6067986
IS_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7304526, upper bound: 187.6268213
IS_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6191941, upper bound: 187.6094675
IS_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6272856, upper bound: 187.6272873
IS_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
IS_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6265245, upper bound: 187.6265245
IS_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7259387, upper bound: 187.5255726
IS_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7259387, upper bound: 187.5272075
IS_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7259387, upper bound: 187.6223350
IS_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.7259387, upper bound: 187.6245032
IS_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6215760, upper bound: 187.5256570
IS_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6215760, upper bound: 187.5256570
IS_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6215760, upper bound: 187.6223438
IS_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6215760, upper bound: 187.6243781
IS_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6093294, upper bound: 187.6151980
IS_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6272846, upper bound: 187.6258431
IS_B1_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4597249
IS_B1_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6265237, upper bound: 187.6250970
IS_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.4394220, upper bound: 187.5899942
IS_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6289531, upper bound: 187.6261656
IS_B1_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5937594, upper bound: 187.4952701
IS_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6289531, upper bound: 187.6261665
IS_B2_A1_B1_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5358576, upper bound: 187.3647378
IS_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6220018, upper bound: 187.5051509
IS_B2_A1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5253691, upper bound: 187.5253691
IS_B2_A1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5253691, upper bound: 187.5253691
IS_B2_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4670488
IS_B2_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6254605, upper bound: 187.6254945
IS_B2_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4671710
IS_B2_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6254605, upper bound: 187.6254605
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6213253, upper bound: 187.5281009
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6213253, upper bound: 187.6243218
IS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6213253, upper bound: 187.5258906
IS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6213253, upper bound: 187.6236270
IS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5833507, upper bound: 187.4709791
IS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6278920
IS_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5833507, upper bound: 187.4672622
IS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
IS_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6153720, upper bound: 187.6061277
IS_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6247712
IS_B2_A2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5350520, upper bound: 187.3730466
IS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6227843, upper bound: 187.6231698
IS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6213253
IS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6213253
IS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6223351
IS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5281009, upper bound: 187.6244066
IS_B2_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
IS_B2_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6250970, upper bound: 187.6265236
IS_B2_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
IS_B2_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6250970, upper bound: 187.6265236
IS_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.4710047, upper bound: 187.5935476
IS_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6278920, upper bound: 187.6261644
IS_B2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6003024, upper bound: 187.4961552
IS_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.48
Output dim: 3, lower bound: -187.6278921, upper bound: 187.6261665

## BFS IS instance: IS_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -40.9809875, 54.7791138, -95.7601013, 95.7600937
1: -32.1188278, 51.3675804, -32.1188278, 51.3675804, -83.4863968, 83.4863968
2: -47.0364113, 57.8066292, -47.0364113, 57.8066292, -104.8430405, 104.8430405
3: -27.2729225, 55.0080185, -27.2729225, 55.0080185, -82.2809448, 82.2809448
4: -53.0424118, 57.1621284, -53.0424118, 57.1621284, -110.2045288, 110.2045135

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3876313, upper bound: 187.6532486
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8003495, upper bound: 187.8016882
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -83.9958038, 86.3003998, -40.9809875, 54.7791138, -138.7749176, 127.2813873
1: -65.5746994, 80.7194061, -32.1188278, 51.3675804, -116.9422760, 112.8382263
2: -95.1113968, 90.8879929, -47.0364113, 57.8066292, -152.9180298, 137.9244080
3: -43.9111938, 99.3519669, -27.2729225, 55.0080185, -98.9192047, 126.6248703
4: -106.2133560, 90.4611893, -53.0424118, 57.1621284, -163.3754730, 143.5036011

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3876313, upper bound: 187.6532486
time: 0.63 seconds

## Relational analysis of IS_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8003495, upper bound: 187.8016882
time: 0.88 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -88.5702744, 94.8887787, -135.8697662, 143.3493500
1: -32.1188278, 51.3675804, -69.3504181, 89.9299164, -122.0487213, 120.7179871
2: -47.0364113, 57.8066292, -100.7325668, 99.5717163, -146.6081085, 158.5391998
3: -27.2729225, 55.0080185, -48.4277000, 106.5347595, -133.8076782, 103.4250565
4: -53.0424118, 57.1621284, -113.0051575, 98.4368286, -151.4792328, 170.1672821

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5295746, upper bound: 187.3878685
time: 0.85 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6210369, upper bound: 187.7222831
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -83.9084015, 86.2436676, -88.7503510, 95.0967102, -179.0051117, 174.9939880
1: -65.5053635, 80.6649551, -69.4899979, 90.1278381, -155.6332092, 150.1549225
2: -95.0113373, 90.8286591, -100.9360809, 99.7855835, -194.7969208, 191.7647400
3: -43.8787804, 99.2585144, -48.5418625, 106.7330246, -150.6118011, 147.5324707
4: -106.1036606, 90.4007339, -113.2336960, 98.6521301, -204.7557983, 203.6343994

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6042536, upper bound: 187.7203125
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6225936, upper bound: 187.7290234
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -124.1356125, 116.5861969, -39.0077286, 53.3303146, -177.4659271, 155.5939026
1: -97.1253967, 110.3894806, -30.5722923, 50.0091782, -147.1345673, 140.9617767
2: -140.7915955, 121.9384384, -44.8006439, 56.3047638, -197.0963593, 166.7390747
3: -59.4011345, 143.5706635, -26.5797043, 52.9133911, -112.3145294, 170.1503601
4: -157.1769104, 121.2831802, -50.5919037, 55.6129456, -212.7898560, 171.8750916

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7019171, upper bound: 187.6164144
time: 0.61 seconds

## Relational analysis of IS_B1_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7241665, upper bound: 187.6239043
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -90.0296860, 95.7459259, -88.7503510, 95.0967102, -185.1263885, 184.4962769
1: -70.5223999, 90.7102203, -69.4899979, 90.1278381, -160.6502228, 160.2001801
2: -102.4902802, 100.3906708, -100.9360809, 99.7855835, -202.2758636, 201.3267517
3: -48.7527466, 108.2179565, -48.5418625, 106.7330246, -155.4857330, 156.7597809
4: -114.9058533, 99.2772675, -113.2336960, 98.6521301, -213.5579834, 212.5109406

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6043074, upper bound: 187.6140935
time: 0.69 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6226914
time: 0.72 seconds

## BFS IS instance: IS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -141.8084412, 131.2259827, -88.7503510, 95.0967102, -236.9051514, 219.9763184
1: -110.9115448, 123.7805481, -69.4899979, 90.1278381, -201.0393829, 193.2705231
2: -160.7136688, 137.5880280, -100.9360809, 99.7855835, -260.4992676, 238.5241089
3: -67.5044250, 162.1680298, -48.5418625, 106.7330246, -173.7824554, 210.5228424
4: -179.2406006, 136.7677307, -113.2336960, 98.6521301, -277.8927002, 250.0013428

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6043074, upper bound: 187.6172788
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6226224, upper bound: 187.6248027
time: 0.74 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -90.1970367, 90.3968811, -86.4171143, 87.4578857, -177.6549225, 176.8139801
1: -70.4170380, 84.6815796, -67.3825378, 81.9784775, -152.3954926, 152.0641022
2: -102.0909729, 95.1755905, -97.7557373, 92.2331390, -194.3240814, 192.9313354
3: -46.0638351, 105.8283768, -44.5546227, 101.4642258, -147.5280457, 150.3829651
4: -113.9848404, 94.7895279, -109.1534348, 91.6931381, -205.6779633, 203.9429626

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5077430, upper bound: 187.6279921
time: 0.58 seconds

## Relational analysis of IS_B1_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7541058, upper bound: 187.7797690
time: 0.65 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.5989838, 91.4505310, -88.4245300, 89.1895752, -180.7885437, 179.8750458
1: -71.5373001, 85.6562042, -69.0220642, 83.5351639, -155.0724335, 154.6782684
2: -103.6967926, 96.2342453, -100.1094131, 93.9118881, -197.6086731, 196.3436127
3: -46.6185226, 107.3869629, -45.5779648, 103.8955078, -150.5140076, 152.9649353
4: -115.7632828, 95.8914185, -111.7675171, 93.5200806, -209.2833557, 207.6589355

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4280861, upper bound: 187.5545643
time: 0.76 seconds

## Relational analysis of IS_B1_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7960062, upper bound: 187.7960065
time: 0.77 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -136.7086029, 129.1925507, -67.0772324, 75.0328751, -211.7414856, 196.2697754
1: -106.8968735, 122.0820007, -52.3760071, 70.2153320, -177.1122131, 174.4580078
2: -154.9251862, 135.8133240, -76.1112671, 79.5657120, -234.4908752, 211.9245605
3: -66.7758255, 156.7858734, -38.7699432, 81.0885849, -147.8643799, 195.5558167
4: -172.8619843, 134.7124939, -85.1397247, 78.6775513, -251.5395203, 219.8522034

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4883516, upper bound: 187.5789068
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6584957, upper bound: 187.6059889
time: 0.68 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -147.2351532, 135.5146332, -87.1010132, 88.2613983, -235.4965515, 222.6156464
1: -115.1447372, 128.0402374, -67.9715729, 82.6533890, -197.7981262, 196.0117798
2: -166.7989197, 142.1491699, -98.5855026, 92.9570694, -259.7559814, 240.7346649
3: -69.7660065, 168.1156921, -45.1060333, 102.4525528, -172.2185364, 213.2217255
4: -186.0553131, 141.3800049, -110.1000519, 92.5308685, -278.5861816, 251.4800568

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7057718, upper bound: 187.6187067
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_B1_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7304526, upper bound: 187.6268183
time: 0.64 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -88.7478867, 89.1764908, -144.2770081, 133.2842255, -222.0321045, 233.4534912
1: -69.2218246, 83.6018448, -112.8129883, 125.9415207, -195.1633453, 196.4148254
2: -100.3838272, 94.0109100, -163.4558258, 139.8710022, -240.2548218, 257.4667358
3: -45.3364944, 104.0492020, -68.6477203, 164.8642273, -210.2007141, 172.6969147
4: -112.0880203, 93.5044250, -182.3238525, 139.0298920, -251.1179199, 275.8282776

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_B1_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6066982, upper bound: 187.7043868
time: 0.62 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_B1_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6066982, upper bound: 187.7062345
time: 0.75 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -90.5690079, 90.7532349, -145.8483124, 134.4916229, -225.0606232, 236.6015472
1: -70.7125702, 85.0148315, -114.0698090, 127.0534286, -197.7659912, 199.0846405
2: -102.5243378, 95.5646133, -165.2633667, 141.0804596, -243.6047974, 260.8279724
3: -46.3124428, 106.2642441, -69.2586975, 166.6131592, -212.9255981, 175.5229492
4: -114.4644547, 95.1655579, -184.3220978, 140.3020172, -254.7664642, 279.4876709

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_B1_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5832655, upper bound: 187.4857609
time: 0.66 seconds

## Relational analysis of IS_B1_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_B1_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265245, upper bound: 187.7241665
time: 0.73 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -145.3360901, 134.7312317, -135.3423920, 128.6092987, -273.9453125, 270.0736084
1: -113.7183151, 127.2939301, -105.9854965, 121.5867310, -235.3050385, 233.2793884
2: -164.7272797, 141.3537750, -153.5312347, 135.1462860, -299.8735352, 294.8849792
3: -69.3720016, 166.3632355, -66.2919998, 156.0335083, -225.4055176, 232.6552429
4: -183.7806396, 140.5365906, -171.4211121, 134.0570679, -317.8377075, 311.9576721

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4346536, upper bound: 187.4346536
time: 0.75 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
time: 0.69 seconds

## BFS IS instance: IS_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -147.3744202, 135.6159363, -142.5714111, 132.4927673, -279.8671875, 278.1873474
1: -115.2622528, 128.1344147, -111.4859314, 125.1495209, -240.4117432, 239.6203461
2: -166.9682312, 142.2507629, -161.5406494, 139.0050201, -305.9732361, 303.7914124
3: -69.8214188, 168.2931366, -68.2609253, 163.1841278, -233.0055237, 236.5540619
4: -186.2393188, 141.4892578, -180.2251587, 138.1768951, -324.4161987, 321.7143250

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_B1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6090455, upper bound: 187.6191628
time: 0.67 seconds

## Relational analysis of IS_B1_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265252, upper bound: 187.6265245
time: 0.64 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -40.9809875, 54.7791138, -143.3868256, 133.7913513
1: -69.5185242, 87.8645935, -32.1188278, 51.3675804, -120.8861008, 119.9834061
2: -100.9003143, 97.3328018, -47.0364113, 57.8066292, -158.7069397, 144.3692017
3: -47.0367928, 106.5991135, -27.2729225, 55.0080185, -102.0447693, 133.8720245
4: -113.0341187, 96.2283783, -53.0424118, 57.1621284, -170.1962280, 149.2707672

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3796222, upper bound: 187.5041244
time: 0.69 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7189907, upper bound: 187.5038593
time: 0.65 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -84.2954559, 86.4964600, -175.1041870, 177.1058044
1: -69.5185242, 87.8645935, -65.8132782, 80.9064255, -150.4249420, 153.6778412
2: -100.9003143, 97.3328018, -95.4560165, 91.0930481, -191.9933472, 192.7887878
3: -47.0367928, 106.5991135, -44.0249329, 99.6720581, -146.7088318, 150.6240387
4: -113.0341187, 96.2283783, -106.5901413, 90.6715698, -203.7056885, 202.8185120

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6164293, upper bound: 187.3647271
time: 0.79 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7189913, upper bound: 187.5051288
time: 0.78 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -140.7002563, 134.6485596, -40.9809875, 54.7791138, -195.4793701, 174.5962830
1: -110.3509521, 127.8535767, -32.1188278, 51.3675804, -161.7185364, 158.2808685
2: -160.0756989, 140.6630554, -47.0364113, 57.8066292, -217.8823242, 185.7634583
3: -68.5588608, 163.3278503, -27.2729225, 55.0080185, -121.7680130, 190.6007690
4: -178.8477173, 139.4344788, -53.0424118, 57.1621284, -236.0098419, 191.5192871

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1136.60 seconds
