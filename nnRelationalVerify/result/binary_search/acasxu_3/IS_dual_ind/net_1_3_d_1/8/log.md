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
execution time: IAR + LP analysis = 1.83 + 1.75 = 3.58 seconds
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
Binary search time: 60.66 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1135.77 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.8827013
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 3, lower bound: -187.6721943, upper bound: 187.8827013
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -149.6440735, 126.7424088, -246.6997375, 258.3909607
1: -93.8448410, 101.7164536, -117.3338928, 118.4335785, -212.2784119, 219.0503540
2: -135.7941437, 113.6401825, -169.7016296, 131.6250763, -267.4192200, 283.3417969
3: -54.4526558, 137.4407501, -63.3496017, 169.2627869, -223.7154388, 200.7903442
4: -151.2667084, 114.1601410, -188.6523895, 133.4867859, -284.7534485, 302.8125305

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6721943
time: 0.62 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6774721
time: 0.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -149.5648346, 126.6903992, -308.8507080, 303.2253113
1: -143.0398865, 144.6755829, -117.2719879, 118.3852081, -261.4251099, 261.9475708
2: -206.5966339, 159.9870605, -169.6125488, 131.5723724, -338.1690063, 329.5996094
3: -78.3434219, 204.8266907, -63.3235054, 169.1803894, -247.5238037, 268.1501770
4: -229.7526093, 160.7959442, -188.5533752, 133.4308624, -363.1834106, 349.3493042

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6721943
time: 0.91 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6721943
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 3, lower bound: -187.6721943, upper bound: 187.6774721
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6721943
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.48
Output dim: 3, lower bound: -187.6774721, upper bound: 187.6774721

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8769586
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8769586
time: 0.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -182.1602936, 153.6605072, -273.6178284, 290.9071655
1: -93.8448410, 101.7164536, -143.0398865, 144.6755829, -238.5203857, 244.7563477
2: -135.7941437, 113.6401825, -206.5966339, 159.9870605, -295.7811584, 320.2367859
3: -54.4526558, 137.4407501, -78.3434219, 204.8266907, -259.2793579, 215.7841492
4: -151.2667084, 114.1601410, -229.7526093, 160.7959442, -312.0625916, 343.9127197

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8823298
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8823298
time: 0.63 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -119.9573212, 108.7468872, -290.9071655, 273.6178284
1: -143.0398865, 144.6755829, -93.8448410, 101.7164536, -244.7563477, 238.5203857
2: -206.5966339, 159.9870605, -135.7941437, 113.6401825, -320.2368164, 295.7811584
3: -78.3434219, 204.8266907, -54.4526558, 137.4407501, -215.7841339, 259.2793579
4: -229.7526093, 160.7959442, -151.2667084, 114.1601410, -343.9127502, 312.0626221

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
time: 0.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -182.1602936, 153.6605072, -335.8208008, 335.8208008
1: -143.0398865, 144.6755829, -143.0398865, 144.6755829, -287.7154541, 287.7154541
2: -206.5966339, 159.9870605, -206.5966339, 159.9870605, -366.5836792, 366.5836792
3: -78.3434219, 204.8266907, -78.3434219, 204.8266907, -283.1701050, 283.1701050
4: -229.7526093, 160.7959442, -229.7526093, 160.7959442, -390.5485229, 390.5485535

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8769586
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8769586
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6673717, upper bound: 187.8823298
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6717293, upper bound: 187.8823298
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6311779, upper bound: 187.6309287
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6396430, upper bound: 187.6316936

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -116.4893188, 105.8845367, -168.4834747, 185.1707153
1: -49.0822411, 64.2613220, -91.1257248, 99.0349350, -148.1171722, 155.3870544
2: -71.3749008, 71.9796066, -131.8765869, 110.6282196, -182.0031128, 203.8561401
3: -33.6786270, 77.7695847, -52.9740067, 133.7148743, -167.3934937, 130.7435913
4: -79.9356384, 71.8462219, -146.9134216, 111.1821060, -191.1177368, 218.7596436

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8219008, upper bound: 187.6291626
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291867
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -119.9573212, 108.7468872, -224.4981689, 225.7081299
1: -90.5305862, 98.8567810, -93.8448410, 101.7164536, -192.2470398, 192.7015991
2: -131.0461578, 110.5165710, -135.7941437, 113.6401825, -244.6863403, 246.3106995
3: -52.9895172, 132.8755493, -54.4526558, 137.4407501, -190.4302673, 187.3281860
4: -145.9682770, 110.9683304, -151.2667084, 114.1601410, -260.1284180, 262.2350464

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8726010
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8769586
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -178.3146515, 150.4789886, -213.0779419, 246.9960480
1: -49.0822411, 64.2613220, -139.9986572, 141.7141266, -190.7963715, 204.2599792
2: -71.3749008, 71.9796066, -202.2458649, 156.6603241, -228.0352173, 274.2254639
3: -33.6786270, 77.7695847, -76.6783066, 200.7028809, -234.3815002, 154.4478760
4: -79.9356384, 71.8462219, -224.9373169, 157.4495392, -237.3851776, 296.7835388

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6291800
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286725, upper bound: 187.6375376
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -182.1602936, 153.6605072, -269.4118347, 287.9111023
1: -90.5305862, 98.8567810, -143.0398865, 144.6755829, -235.2061462, 241.8966522
2: -131.0461578, 110.5165710, -206.5966339, 159.9870605, -291.0332031, 317.1132202
3: -52.9895172, 132.8755493, -78.3434219, 204.8266907, -257.8162231, 211.2189484
4: -145.9682770, 110.9683304, -229.7526093, 160.7959442, -306.7642212, 340.7209473

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6312910
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -119.9573212, 108.7468872, -258.0307922, 254.6466827
1: -116.9905624, 127.1221771, -93.8448410, 101.7164536, -218.7070160, 220.9670105
2: -169.2326202, 141.1778259, -135.7941437, 113.6401825, -282.8728027, 276.9718933
3: -69.0598907, 170.1712341, -54.4526558, 137.4407501, -206.5006256, 224.6238861
4: -188.5170746, 140.5437622, -151.2667084, 114.1601410, -302.6772156, 291.8104248

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291800, upper bound: 187.6279076
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301558
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -119.8824997, 108.7030869, -323.4262085, 305.8557129
1: -168.6910553, 175.9269562, -93.7861786, 101.6748047, -270.3657837, 269.7130737
2: -243.5518188, 193.9766388, -135.7100830, 113.5954590, -357.1472778, 329.6866760
3: -95.6573029, 241.0182190, -54.4305954, 137.3644867, -232.4452362, 295.4487610
4: -271.0773621, 193.9729767, -151.1734619, 114.1141815, -385.1915283, 345.1464233

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375376, upper bound: 187.6286725
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -182.1602936, 153.6605072, -302.9444580, 316.8496704
1: -116.9905624, 127.1221771, -143.0398865, 144.6755829, -261.6661072, 270.1620178
2: -169.2326202, 141.1778259, -206.5966339, 159.9870605, -329.2196655, 347.7744141
3: -69.0598907, 170.1712341, -78.3434219, 204.8266907, -273.8865967, 248.5146027
4: -188.5170746, 140.5437622, -229.7526093, 160.7959442, -349.3130188, 370.2963867

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6309287
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -182.1177826, 153.6374054, -368.3604736, 368.0910034
1: -168.6910553, 175.9269562, -143.0064392, 144.6536407, -313.3446655, 318.9333191
2: -243.5518188, 193.9766388, -206.5487366, 159.9633484, -403.5151672, 400.5253601
3: -95.6573029, 241.0182190, -78.3319092, 204.7839813, -300.0346985, 319.3501282
4: -271.0773621, 193.9729767, -229.7001953, 160.7719421, -431.8493042, 423.6731262

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6316936
time: 0.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.8219008, upper bound: 187.6291626
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6290496, upper bound: 187.6291867
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8726010
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.8769586, upper bound: 187.8769586
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6279076, upper bound: 187.6291800
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6286725, upper bound: 187.6375376
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6301558, upper bound: 187.6312910
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6309206, upper bound: 187.6396486
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6291800, upper bound: 187.6279076
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6312910, upper bound: 187.6301558
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6375376, upper bound: 187.6286725
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6396486, upper bound: 187.6309206
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6304130
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6304130, upper bound: 187.6309287
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6311779
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -187.6388782, upper bound: 187.6316936

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -90.1271210, 90.0096359, -152.6085663, 158.8085175
1: -49.0822411, 64.2613220, -70.3955536, 84.3041153, -133.3863525, 134.6568604
2: -71.3749008, 71.9796066, -102.0280762, 94.7010880, -166.0759888, 174.0076447
3: -33.6786270, 77.7695847, -45.7114983, 105.8294601, -139.5080872, 123.4810638
4: -79.9356384, 71.8462219, -113.9134064, 94.3664169, -174.3020477, 185.7596130

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -62.5538635, 68.6532745, -146.5204315, 134.4402618, -196.9941254, 215.1737061
1: -49.0467415, 64.2349854, -114.5982819, 127.0411911, -176.0879364, 178.8332672
2: -71.3245544, 71.9503098, -165.9999237, 140.9418030, -212.2663574, 237.9502258
3: -33.6632614, 77.7246017, -69.1322937, 167.3540039, -201.0172729, 146.8568573
4: -79.8798065, 71.8161316, -185.1622620, 140.2452240, -220.1250153, 256.9783325

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -62.5989532, 68.6813965, -184.4327087, 168.3497620
1: -90.5305862, 98.8567810, -49.0822411, 64.2613220, -154.7919006, 147.9390259
2: -131.0461578, 110.5165710, -71.3749008, 71.9796066, -203.0257568, 181.8914795
3: -52.9895172, 132.8755493, -33.6786270, 77.7695847, -130.7590942, 166.5541534
4: -145.9682770, 110.9683304, -79.9356384, 71.8462219, -217.8144989, 190.9039612

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -115.7513199, 105.7508011, -221.5021210, 221.5021210
1: -90.5305862, 98.8567810, -90.5305862, 98.8567810, -189.3873596, 189.3873444
2: -131.0461578, 110.5165710, -131.0461578, 110.5165710, -241.5627136, 241.5627136
3: -52.9895172, 132.8755493, -52.9895172, 132.8755493, -185.8650665, 185.8650665
4: -145.9682770, 110.9683304, -145.9682770, 110.9683304, -256.9366150, 256.9366150

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6297159
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -145.5118103, 131.5657959, -194.1647034, 214.1932068
1: -49.0822411, 64.2613220, -114.0084686, 124.1955643, -173.2778015, 178.2697601
2: -71.3749008, 71.9796066, -164.9636383, 137.8805237, -209.2554321, 236.9432068
3: -33.6786270, 77.7695847, -67.3987885, 166.0986328, -199.7772522, 145.1683655
4: -79.9356384, 71.8462219, -183.7941284, 137.2277069, -217.1633453, 255.6403503

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -62.5538635, 68.6532745, -210.5842438, 182.6192322, -245.1730804, 279.2374878
1: -49.0467415, 64.2349854, -165.3393250, 172.7866974, -221.8334351, 229.5743103
2: -71.3245544, 71.9503098, -238.8750916, 190.4543152, -261.7788696, 310.8254089
3: -33.6632614, 77.7246017, -93.8774033, 236.5684204, -270.2316895, 170.7260895
4: -79.8798065, 71.8161316, -265.9034119, 190.4143066, -270.2940674, 337.7195435

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
time: 0.56 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -149.2839355, 134.6893616, -250.4406281, 255.0347137
1: -90.5305862, 98.8567810, -116.9905624, 127.1221771, -217.6527710, 215.8473053
2: -131.0461578, 110.5165710, -169.2326202, 141.1778259, -272.2239380, 279.7492065
3: -52.9895172, 132.8755493, -69.0598907, 170.1712341, -223.1607513, 201.9354248
4: -145.9682770, 110.9683304, -188.5170746, 140.5437622, -286.5120239, 299.4854126

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -115.6754456, 105.7059631, -214.7231140, 185.9732056, -301.6486511, 320.4290771
1: -90.4710693, 98.8141022, -168.6910553, 175.9269562, -266.3979797, 267.5051270
2: -130.9607849, 110.4703064, -243.5518188, 193.9766388, -324.9374390, 354.0221252
3: -52.9669495, 132.7978973, -95.6573029, 241.0182190, -293.9851685, 227.6103516
4: -145.8735962, 110.9213715, -271.0773621, 193.9729767, -339.8465271, 381.9987183

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -145.5118103, 131.5657959, -62.5989532, 68.6813965, -214.1932068, 194.1647034
1: -114.0084686, 124.1955643, -49.0822411, 64.2613220, -178.2697754, 173.2778015
2: -164.9636383, 137.8805237, -71.3749008, 71.9796066, -236.9431915, 209.2554321
3: -67.3987885, 166.0986328, -33.6786270, 77.7695847, -145.1683655, 199.7772522
4: -183.7941284, 137.2277069, -79.9356384, 71.8462219, -255.6403503, 217.1633453

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658922
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -115.7513199, 105.7508011, -255.0347290, 250.4406433
1: -116.9905624, 127.1221771, -90.5305862, 98.8567810, -215.8473206, 217.6527710
2: -169.2326202, 141.1778259, -131.0461578, 110.5165710, -279.7492065, 272.2239380
3: -69.0598907, 170.1712341, -52.9895172, 132.8755493, -201.9354248, 223.1607513
4: -188.5170746, 140.5437622, -145.9682770, 110.9683304, -299.4854126, 286.5119934

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -210.5842438, 182.6192322, -62.5538635, 68.6532745, -279.2375183, 245.1730957
1: -165.3393250, 172.7866974, -49.0467415, 64.2349854, -229.5743103, 221.8334351
2: -238.8750916, 190.4543152, -71.3245544, 71.9503098, -310.8254089, 261.7788696
3: -93.8774033, 236.5684204, -33.6632614, 77.7246017, -170.7261047, 270.2316895
4: -265.9034119, 190.4143066, -79.8798065, 71.8161316, -337.7195435, 270.2940674

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6266799
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6286725
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -115.6754456, 105.7059631, -320.4290771, 301.6486511
1: -168.6910553, 175.9269562, -90.4710693, 98.8141022, -267.5051575, 266.3980103
2: -243.5518188, 193.9766388, -130.9607849, 110.4703064, -354.0221252, 324.9373779
3: -95.6573029, 241.0182190, -52.9669495, 132.7978973, -227.6103668, 293.9851685
4: -271.0773621, 193.9729767, -145.8735962, 110.9213715, -381.9986877, 339.8465271

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6309206
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -149.2839355, 134.6893616, -283.9732361, 283.9732056
1: -116.9905624, 127.1221771, -116.9905624, 127.1221771, -244.1127319, 244.1127167
2: -169.2326202, 141.1778259, -169.2326202, 141.1778259, -310.4104614, 310.4104614
3: -69.0598907, 170.1712341, -69.0598907, 170.1712341, -239.2311096, 239.2311096
4: -188.5170746, 140.5437622, -188.5170746, 140.5437622, -329.0608215, 329.0608215

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -214.7231140, 185.9732056, -335.2571411, 349.4124451
1: -116.9905624, 127.1221771, -168.6910553, 175.9269562, -292.9174500, 295.8131714
2: -169.2326202, 141.1778259, -243.5518188, 193.9766388, -363.2092590, 384.7296143
3: -69.0598907, 170.1712341, -95.6573029, 241.0182190, -310.0781250, 265.3165283
4: -188.5170746, 140.5437622, -271.0773621, 193.9729767, -382.4900208, 411.6211243

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295757
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -149.2839355, 134.6893616, -349.4123840, 335.2571411
1: -168.6910553, 175.9269562, -116.9905624, 127.1221771, -295.8131714, 292.9174805
2: -243.5518188, 193.9766388, -169.2326202, 141.1778259, -384.7296143, 363.2092590
3: -95.6573029, 241.0182190, -69.0598907, 170.1712341, -265.3165283, 310.0780640
4: -271.0773621, 193.9729767, -188.5170746, 140.5437622, -411.6211243, 382.4900208

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -214.7231140, 185.9732056, -400.6963196, 400.6963196
1: -168.6910553, 175.9269562, -168.6910553, 175.9269562, -344.6179504, 344.6179504
2: -243.5518188, 193.9766388, -243.5518188, 193.9766388, -437.5284424, 437.5284424
3: -95.6573029, 241.0182190, -95.6573029, 241.0182190, -336.2658691, 336.2658691
4: -271.0773621, 193.9729767, -271.0773621, 193.9729767, -465.0502930, 465.0503235

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289272
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309206
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291626
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6288563, upper bound: 187.6291867
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6291626, upper bound: 187.8219008
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6297159
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6277144, upper bound: 187.6291800
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6284793, upper bound: 187.6375376
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6301317, upper bound: 187.6312910
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6308965, upper bound: 187.6396486
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658922
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6280304, upper bound: 187.5295757
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6280304, upper bound: 187.6301558
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6266799
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6286725
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6289280
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6323670, upper bound: 187.6309206
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295517
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301317
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6268885, upper bound: 187.5295757
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6301490, upper bound: 187.6301558
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289039
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6308965
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6312251, upper bound: 187.6289272
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.40
Output dim: 3, lower bound: -187.6385066, upper bound: 187.6309206

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -90.1271210, 90.0096359, -130.9906311, 144.9062347
1: -32.1188278, 51.3675804, -70.3955536, 84.3041153, -116.4229126, 121.7631302
2: -47.0364113, 57.8066292, -102.0280762, 94.7010880, -141.7375031, 159.8347015
3: -27.2729225, 55.0080185, -45.7114983, 105.8294601, -133.1023712, 100.7195053
4: -53.0424118, 57.1621284, -113.9134064, 94.3664169, -147.4088287, 171.0755005

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4879346, upper bound: 187.6274424
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149799, upper bound: 187.6263038
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -90.1271210, 90.0096359, -178.7599792, 185.2238007
1: -69.4899979, 90.1278381, -70.3955536, 84.3041153, -153.7940979, 160.5233765
2: -100.9360809, 99.7855835, -102.0280762, 94.7010880, -195.6371765, 201.8136597
3: -48.5418625, 106.7330246, -45.7114983, 105.8294601, -154.3713074, 152.4445190
4: -113.2336960, 98.6521301, -113.9134064, 94.3664169, -207.6000977, 212.5655060

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4879346, upper bound: 187.6274424
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149799, upper bound: 187.6263038
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -146.5204315, 134.4402618, -175.4212494, 201.2995453
1: -32.1188278, 51.3675804, -114.5982819, 127.0411911, -159.1600037, 165.9658661
2: -47.0364113, 57.8066292, -165.9999237, 140.9418030, -187.9781952, 223.8065491
3: -27.2729225, 55.0080185, -69.1322937, 167.3540039, -194.6269226, 124.1403046
4: -53.0424118, 57.1621284, -185.1622620, 140.2452240, -193.2876129, 242.3243561

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4329129, upper bound: 187.6274665
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.6263033
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -146.5204315, 134.4402618, -223.1906128, 241.6171265
1: -69.4899979, 90.1278381, -114.5982819, 127.0411911, -196.5311584, 204.7261200
2: -100.9360809, 99.7855835, -165.9999237, 140.9418030, -241.8778839, 265.7854919
3: -48.5418625, 106.7330246, -69.1322937, 167.3540039, -215.8958282, 175.8652802
4: -113.2336960, 98.6521301, -185.1622620, 140.2452240, -253.4788971, 283.8143311

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4329129, upper bound: 187.6274424
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.6263033
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -62.5989532, 68.6813965, -158.1293488, 152.4808350
1: -69.8412399, 84.1715393, -49.0822411, 64.2613220, -134.1025696, 133.2537537
2: -101.2739258, 94.6007919, -71.3749008, 71.9796066, -173.2534943, 165.9756927
3: -45.8811989, 105.0105896, -33.6786270, 77.7695847, -123.6507874, 138.6892090
4: -113.0568390, 94.2403870, -79.9356384, 71.8462219, -184.9030609, 174.1760254

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6288563
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6290496
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -62.5538635, 68.6532745, -214.6998444, 197.2876129
1: -114.2232513, 127.2852173, -49.0467415, 64.2349854, -178.4582367, 176.3319550
2: -165.4884644, 141.3374176, -71.3245544, 71.9503098, -237.4387817, 212.6619720
3: -69.3965378, 166.8359833, -33.6632614, 77.7246017, -147.1211243, 200.4992371
4: -184.5755615, 140.5543518, -79.8798065, 71.8161316, -256.3916321, 220.4341431

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6288563
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -115.7513199, 105.7508011, -195.1987762, 205.6332092
1: -69.8412399, 84.1715393, -90.5305862, 98.8567810, -168.6979675, 174.7021179
2: -101.2739258, 94.6007919, -131.0461578, 110.5165710, -211.7904816, 225.6469421
3: -45.8811989, 105.0105896, -52.9895172, 132.8755493, -178.7567444, 158.0001068
4: -113.0568390, 94.2403870, -145.9682770, 110.9683304, -224.0251770, 240.2086639

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6295361
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6297159
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -115.6754456, 105.7059631, -251.7525177, 250.4091797
1: -114.2232513, 127.2852173, -90.4710693, 98.8141022, -213.0373535, 217.7562866
2: -165.4884644, 141.3374176, -130.9607849, 110.4703064, -275.9587708, 272.2982178
3: -69.3965378, 166.8359833, -52.9669495, 132.7978973, -202.1944275, 219.8029022
4: -184.5755615, 140.5543518, -145.8735962, 110.9213715, -295.4969177, 286.4279480

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6295361
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6297159
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -145.5118103, 131.5657959, -172.5467682, 200.2909241
1: -32.1188278, 51.3675804, -114.0084686, 124.1955643, -156.3143768, 165.3760223
2: -47.0364113, 57.8066292, -164.9636383, 137.8805237, -184.9169006, 222.7702637
3: -27.2729225, 55.0080185, -67.3987885, 166.0986328, -193.3715515, 122.4068069
4: -53.0424118, 57.1621284, -183.7941284, 137.2277069, -190.2701111, 240.9562378

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4657145, upper bound: 187.6274597
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257306, upper bound: 187.6263125
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -145.5118103, 131.5657959, -220.3161163, 240.6085205
1: -69.4899979, 90.1278381, -114.0084686, 124.1955643, -193.6855164, 204.1362762
2: -100.9360809, 99.7855835, -164.9636383, 137.8805237, -238.8166046, 264.7492065
3: -48.5418625, 106.7330246, -67.3987885, 166.0986328, -214.6404419, 174.1318054
4: -113.2336960, 98.6521301, -183.7941284, 137.2277069, -250.4613953, 282.4462280

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4657145, upper bound: 187.6274597
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257306, upper bound: 187.6263125
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -210.2420349, 182.2295837, -223.2105713, 265.0211182
1: -32.1188278, 51.3675804, -165.0722046, 172.4158630, -204.5346985, 216.4397888
2: -47.0364113, 57.8066292, -238.4860687, 190.0432739, -237.0796356, 296.2926941
3: -27.2729225, 55.0080185, -93.6605988, 236.1824341, -263.4553528, 147.6490173
4: -53.0424118, 57.1621284, -265.4706421, 190.0118713, -243.0542603, 322.6327515

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302560
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302494
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -210.6589203, 182.7044373, -271.4547729, 305.7556152
1: -69.4899979, 90.1278381, -165.3976288, 172.8678131, -242.3578186, 255.5254669
2: -100.9360809, 99.7855835, -238.9599762, 190.5442200, -291.4802856, 338.7455444
3: -48.5418625, 106.7330246, -93.9248505, 236.6527405, -285.1946106, 199.7076874
4: -113.2336960, 98.6521301, -265.9977722, 190.5023956, -303.7360229, 364.6499023

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -149.2839355, 134.6893616, -224.1372833, 239.1658325
1: -69.8412399, 84.1715393, -116.9905624, 127.1221771, -196.9633942, 201.1620789
2: -101.2739258, 94.6007919, -169.2326202, 141.1778259, -242.4517365, 263.8334045
3: -45.8811989, 105.0105896, -69.0598907, 170.1712341, -216.0524139, 174.0704803
4: -113.0568390, 94.2403870, -188.5170746, 140.5437622, -253.6005859, 282.7574463

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -149.2839355, 134.6893616, -280.7358704, 284.0176392
1: -114.2232513, 127.2852173, -116.9905624, 127.1221771, -241.3454132, 244.2757874
2: -165.4884644, 141.3374176, -169.2326202, 141.1778259, -306.6662598, 310.5700378
3: -69.3965378, 166.8359833, -69.0598907, 170.1712341, -239.5677795, 235.8958740
4: -184.5755615, 140.5543518, -188.5170746, 140.5437622, -325.1193237, 329.0714111

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -214.7231140, 185.9732056, -275.4211731, 304.6050110
1: -69.8412399, 84.1715393, -168.6910553, 175.9269562, -245.7681885, 252.8625793
2: -101.2739258, 94.6007919, -243.5518188, 193.9766388, -295.2505493, 338.1526184
3: -45.8811989, 105.0105896, -95.6573029, 241.0182190, -286.8993835, 199.7381897
4: -113.0568390, 94.2403870, -271.0773621, 193.9729767, -307.0298157, 365.3177490

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -214.7231140, 185.9732056, -332.0197754, 349.4568176
1: -114.2232513, 127.2852173, -168.6910553, 175.9269562, -290.1501770, 295.9762573
2: -165.4884644, 141.3374176, -243.5518188, 193.9766388, -359.4650879, 384.8892212
3: -69.3965378, 166.8359833, -95.6573029, 241.0182190, -310.4147644, 261.6839600
4: -184.5755615, 140.5543518, -271.0773621, 193.9729767, -378.5485229, 411.6316833

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -134.6475983, 125.2181320, -62.5989532, 68.6813965, -203.3289948, 187.8170624
1: -105.6099548, 118.3325500, -49.0822411, 64.2613220, -169.8712616, 167.4147949
2: -152.8045502, 131.4833069, -71.3749008, 71.9796066, -224.7841339, 202.8582153
3: -64.1565170, 155.1156464, -33.6786270, 77.7695847, -141.9261017, 188.7942810
4: -170.4454498, 130.4599762, -79.9356384, 71.8462219, -242.2916718, 210.3956146

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4657145
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658922
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -141.3795471, 128.9066467, -62.5989532, 68.6813965, -210.0609436, 191.5055695
1: -110.7312317, 121.6841736, -49.0822411, 64.2613220, -174.9925537, 170.7664185
2: -160.2674408, 135.1351776, -71.3749008, 71.9796066, -232.2469940, 206.5100708
3: -66.0420303, 161.7633057, -33.6786270, 77.7695847, -143.8116150, 195.4419250
4: -178.6372070, 134.3881073, -79.9356384, 71.8462219, -250.4834290, 214.3237457

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6257306
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -115.7513199, 105.7508011, -194.3585205, 208.5616608
1: -69.5185242, 87.8645935, -90.5305862, 98.8567810, -168.3752747, 178.3951721
2: -100.9003143, 97.3328018, -131.0461578, 110.5165710, -211.4168854, 228.3789520
3: -47.0367928, 106.5991135, -52.9895172, 132.8755493, -179.9123383, 159.5886230
4: -113.0341187, 96.2283783, -145.9682770, 110.9683304, -224.0024414, 242.1966553

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295517
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295757
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -115.7513199, 105.7508011, -250.5600586, 247.4103546
1: -113.4549103, 124.2506790, -90.5305862, 98.8567810, -212.3116913, 214.7812653
2: -164.1941833, 138.0722961, -131.0461578, 110.5165710, -274.7107544, 269.1184387
3: -67.5839310, 165.3264618, -52.9895172, 132.8755493, -200.4594727, 218.3159790
4: -182.9160309, 137.3219604, -145.9682770, 110.9683304, -293.8843689, 283.2901917

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6285483
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6287846
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.9379730, 137.5682220, -62.5538635, 68.6532745, -212.5912476, 199.0756073
1: -112.8864594, 130.6750946, -49.0467415, 64.2349854, -177.1214447, 177.9100037
2: -163.7579346, 143.6673584, -71.3245544, 71.9503098, -235.7082520, 213.0226135
3: -70.0619278, 166.9364624, -33.6632614, 77.7246017, -145.9428406, 200.5997314
4: -182.9681396, 142.4337006, -79.8798065, 71.8161316, -254.7842102, 221.3197784

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6264867
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6265020
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -210.1815033, 182.8923798, -62.5538635, 68.6532745, -278.8347473, 245.4462433
1: -165.0845642, 173.0005798, -49.0467415, 64.2349854, -229.3195343, 222.0319824
2: -238.4494476, 190.8179474, -71.3245544, 71.9503098, -310.3996887, 262.1425171
3: -94.1423187, 236.1411896, -33.6632614, 77.7246017, -170.7870026, 269.8044434
4: -265.3901367, 190.7013702, -79.8798065, 71.8161316, -337.2062683, 270.5811157

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6284793
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6285038
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -115.6754456, 105.7059631, -249.6634369, 252.3068695
1: -112.9016113, 130.6958466, -90.4710693, 98.8141022, -211.7156982, 219.3267212
2: -163.7802124, 143.6897125, -130.9607849, 110.4703064, -274.2505188, 272.4970703
3: -70.0736237, 166.9585724, -52.9669495, 132.7978973, -200.8233948, 219.9254913
4: -182.9930725, 142.4557343, -145.8735962, 110.9213715, -293.9144287, 287.2185059

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289039
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289272
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -115.6754456, 105.7059631, -315.9866943, 298.6829834
1: -165.1631775, 173.1105804, -90.4710693, 98.8141022, -263.9772339, 263.5816345
2: -238.5625763, 190.9392853, -130.9607849, 110.4703064, -349.0328979, 321.9000549
3: -94.2064362, 236.2539673, -52.9669495, 132.7978973, -226.1066132, 289.2209167
4: -265.5162659, 190.8201752, -145.8735962, 110.9213715, -376.4376221, 336.6937561

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6291677
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6292015
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -145.5118103, 131.5657959, -220.1735077, 238.3221741
1: -69.5185242, 87.8645935, -114.0084686, 124.1955643, -193.7140656, 201.8730011
2: -100.9003143, 97.3328018, -164.9636383, 137.8805237, -238.7808380, 262.2964172
3: -47.0367928, 106.5991135, -67.3987885, 166.0986328, -213.1354065, 173.9978943
4: -113.0341187, 96.2283783, -183.7941284, 137.2277069, -250.2618256, 280.0224915

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4651043, upper bound: 187.5079903
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252268, upper bound: 187.5068518
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -149.2839355, 134.6893616, -279.4986267, 280.9429626
1: -113.4549103, 124.2506790, -116.9905624, 127.1221771, -240.5770874, 241.2412415
2: -164.1941833, 138.0722961, -169.2326202, 141.1778259, -305.3720093, 307.3049316
3: -67.5839310, 165.3264618, -69.0598907, 170.1712341, -237.7551575, 234.3863525
4: -182.9160309, 137.3219604, -188.5170746, 140.5437622, -323.4597778, 325.8389587

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -210.6589203, 182.7044373, -271.3121033, 303.4692688
1: -69.5185242, 87.8645935, -165.3976288, 172.8678131, -242.3863373, 253.2621918
2: -100.9003143, 97.3328018, -238.9599762, 190.5442200, -291.4445190, 336.2927856
3: -47.0367928, 106.5991135, -93.9248505, 236.6527405, -283.6895447, 199.6252899
4: -113.0341187, 96.2283783, -265.9977722, 190.5023956, -303.5364990, 362.2261353

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5273276
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295757
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -214.7231140, 185.9732056, -330.7824707, 346.3821411
1: -113.4549103, 124.2506790, -168.6910553, 175.9269562, -289.3818665, 292.9416809
2: -164.1941833, 138.0722961, -243.5518188, 193.9766388, -358.1708374, 381.6241150
3: -67.5839310, 165.3264618, -95.6573029, 241.0182190, -308.6020813, 260.2376709
4: -182.9160309, 137.3219604, -271.0773621, 193.9729767, -376.8889771, 408.3992615

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6279076
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6301558
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -145.5118103, 131.5657959, -275.5232544, 282.6395874
1: -112.9016113, 130.6958466, -114.0084686, 124.1955643, -237.0971527, 243.2751312
2: -163.7802124, 143.6897125, -164.9636383, 137.8805237, -301.6607361, 307.0957642
3: -70.0736237, 166.9585724, -67.3987885, 166.0986328, -234.4406738, 234.3573456
4: -182.9930725, 142.4557343, -183.7941284, 137.2277069, -320.2207642, 325.8397827

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4682701, upper bound: 187.6271832
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283557, upper bound: 187.6260447
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -149.2839355, 134.6893616, -344.9700928, 332.2914429
1: -165.1631775, 173.1105804, -116.9905624, 127.1221771, -292.2853088, 290.1011353
2: -238.5625763, 190.9392853, -169.2326202, 141.1778259, -379.7403564, 360.1719055
3: -94.2064362, 236.2539673, -69.0598907, 170.1712341, -263.8127441, 305.3138428
4: -265.5162659, 190.8201752, -188.5170746, 140.5437622, -406.0600281, 379.3372498

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276534
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6308965
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -210.6589203, 182.7044373, -326.6618958, 348.1036072
1: -112.9016113, 130.6958466, -165.3976288, 172.8678131, -285.7693787, 294.9006653
2: -163.7802124, 143.6897125, -238.9599762, 190.5442200, -354.3244324, 381.3421936
3: -70.0736237, 166.9585724, -93.9248505, 236.6527405, -305.1015930, 260.1168823
4: -182.9930725, 142.4557343, -265.9977722, 190.5023956, -373.4954224, 408.4270020

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6289272
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -214.7231140, 185.9732056, -396.2539673, 397.7306213
1: -165.1631775, 173.1105804, -168.6910553, 175.9269562, -341.0900574, 341.8016052
2: -238.5625763, 190.9392853, -243.5518188, 193.9766388, -432.5392151, 434.4910889
3: -94.2064362, 236.2539673, -95.6573029, 241.0182190, -334.7620850, 331.2743225
4: -265.5162659, 190.8201752, -271.0773621, 193.9729767, -459.4892578, 461.8974915

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6309206
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.46 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4879346, upper bound: 187.6274424
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.8149799, upper bound: 187.6263038
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4879346, upper bound: 187.6274424
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.8149799, upper bound: 187.6263038
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4329129, upper bound: 187.6274665
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6268437, upper bound: 187.6263033
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4329129, upper bound: 187.6274424
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6268437, upper bound: 187.6263033
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6288563
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6291626, upper bound: 187.6290496
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6288563
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6291867, upper bound: 187.6290496
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6295361
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6312736, upper bound: 187.6297159
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6295361
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6312977, upper bound: 187.6297159
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4657145, upper bound: 187.6274597
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6257306, upper bound: 187.6263125
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4657145, upper bound: 187.6274597
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6257306, upper bound: 187.6263125
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302560
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6302494
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6259194
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5271344, upper bound: 187.6291800
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6280304
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295517, upper bound: 187.6287787
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4657145
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6274597, upper bound: 187.4658922
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6257306
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6263125, upper bound: 187.6259131
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295517
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6259194, upper bound: 187.5295757
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6285483
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6287846
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6264867
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6259194, upper bound: 187.6265020
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6284793
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6285038
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289039
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6289272
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6291677
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6302560, upper bound: 187.6292015
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4651043, upper bound: 187.5079903
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6252268, upper bound: 187.5068518
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6268885
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5295690, upper bound: 187.6301317
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5273276
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6256607, upper bound: 187.5295757
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6279076
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6289213, upper bound: 187.6301558
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.4682701, upper bound: 187.6271832
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6283557, upper bound: 187.6260447
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6276534
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.5379266, upper bound: 187.6308965
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6257908
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6299974, upper bound: 187.6289272
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6278615
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -187.6372789, upper bound: 187.6309206

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -79.3641357, 83.6889038, -124.6698914, 134.1432495
1: -32.1188278, 51.3675804, -62.1182899, 78.4378281, -110.5566406, 113.4858704
2: -47.0364113, 57.8066292, -90.0294266, 88.3096237, -135.3460083, 147.8360596
3: -27.2729225, 55.0080185, -42.5505676, 95.0083084, -122.2812042, 97.5585861
4: -53.0424118, 57.1621284, -100.7234650, 87.7017822, -140.7441711, 157.8855896

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4879346, upper bound: 187.3900516
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4879346, upper bound: 187.8172821
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -86.8396988, 87.8918228, -128.8728027, 141.6188049
1: -32.1188278, 51.3675804, -67.8093796, 82.2977982, -114.4166031, 119.1769562
2: -47.0364113, 57.8066292, -98.3046494, 92.5141525, -139.5505524, 156.1112823
3: -27.2729225, 55.0080185, -44.6555214, 102.4257660, -129.6986847, 99.6635437
4: -53.0424118, 57.1621284, -109.8163681, 92.1429977, -145.1854095, 166.9785004

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149807, upper bound: 187.3900516
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8149807, upper bound: 187.8172821
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -79.3641357, 83.6889038, -172.4392548, 174.4608459
1: -69.4899979, 90.1278381, -62.1182899, 78.4378281, -147.9278259, 152.2461090
2: -100.9360809, 99.7855835, -90.0294266, 88.3096237, -189.2456970, 189.8150024
3: -48.5418625, 106.7330246, -42.5505676, 95.0083084, -143.5004425, 149.2835846
4: -113.2336960, 98.6521301, -100.7234650, 87.7017822, -200.9354706, 199.3755951

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4881171, upper bound: 187.3743339
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4881171, upper bound: 187.6263038
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -86.8396988, 87.8918228, -176.6421509, 181.9363861
1: -69.4899979, 90.1278381, -67.8093796, 82.2977982, -151.7877808, 157.9371796
2: -100.9360809, 99.7855835, -98.3046494, 92.5141525, -193.4502258, 198.0902405
3: -48.5418625, 106.7330246, -44.6555214, 102.4257660, -150.9676056, 151.3885345
4: -113.2336960, 98.6521301, -109.8163681, 92.1429977, -205.3766632, 208.4685059

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151632, upper bound: 187.3743339
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8151632, upper bound: 187.6263038
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -135.7535553, 128.3042145, -169.2852020, 190.5326538
1: -32.1188278, 51.3675804, -106.3113556, 121.3298569, -153.4486694, 157.6789398
2: -47.0364113, 57.8066292, -153.9722137, 134.7279053, -181.7643127, 211.7788239
3: -27.2729225, 55.0080185, -66.0186615, 156.5064545, -183.7793732, 121.0266724
4: -53.0424118, 57.1621284, -171.9403839, 133.7289734, -186.7713623, 229.1025085

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4329129, upper bound: 187.3900510
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4329129, upper bound: 187.8172815
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -143.0343933, 132.1953430, -173.1763306, 197.8134918
1: -32.1188278, 51.3675804, -111.8522568, 124.8993530, -157.0181580, 163.2198181
2: -47.0364113, 57.8066292, -162.0431366, 138.5972443, -185.6336365, 219.8497620
3: -27.2729225, 55.0080185, -67.9975128, 163.6931000, -190.9660187, 123.0055313
4: -53.0424118, 57.1621284, -180.8023071, 137.8642426, -190.9066315, 237.9644318

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.3900510
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268437, upper bound: 187.8172815
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -135.7535553, 128.3042145, -217.0545654, 230.8502350
1: -69.4899979, 90.1278381, -106.3113556, 121.3298569, -190.8198242, 196.4391785
2: -100.9360809, 99.7855835, -153.9722137, 134.7279053, -235.6639709, 253.7577972
3: -48.5418625, 106.7330246, -66.0186615, 156.5064545, -205.0482941, 172.7516785
4: -113.2336960, 98.6521301, -171.9403839, 133.7289734, -246.9626465, 270.5924988

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4330954, upper bound: 187.3743333
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4330954, upper bound: 187.6263033
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -143.0343933, 132.1953430, -220.9456940, 238.1310730
1: -69.4899979, 90.1278381, -111.8522568, 124.8993530, -194.3892975, 201.9800720
2: -100.9360809, 99.7855835, -162.0431366, 138.5972443, -239.5333252, 261.8287354
3: -48.5418625, 106.7330246, -67.9975128, 163.6931000, -212.2349243, 174.7305298
4: -113.2336960, 98.6521301, -180.8023071, 137.8642426, -251.0979156, 279.4544373

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270262, upper bound: 187.3743333
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270262, upper bound: 187.6263033
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -40.9809875, 54.7791138, -144.2270508, 130.8628845
1: -69.8412399, 84.1715393, -32.1188278, 51.3675804, -121.2088165, 116.2903595
2: -101.2739258, 94.6007919, -47.0364113, 57.8066292, -159.0805359, 141.6372070
3: -45.8811989, 105.0105896, -27.2729225, 55.0080185, -100.8892212, 132.2835083
4: -113.0568390, 94.2403870, -53.0424118, 57.1621284, -170.2189636, 147.2828064

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4879346
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8149799
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -88.7503510, 95.0967102, -184.5446472, 178.6322479
1: -69.8412399, 84.1715393, -69.4899979, 90.1278381, -159.9690552, 153.6614990
2: -101.2739258, 94.6007919, -100.9360809, 99.7855835, -201.0594940, 195.5368652
3: -45.8811989, 105.0105896, -48.5418625, 106.7330246, -152.6142120, 153.4187927
4: -113.0568390, 94.2403870, -113.2336960, 98.6521301, -211.7089691, 207.4740753

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274424, upper bound: 187.4881171
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.8151624
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -40.9809875, 54.7791138, -200.8256836, 175.7147522
1: -114.2232513, 127.2852173, -32.1188278, 51.3675804, -165.5908356, 159.4040375
2: -165.4884644, 141.3374176, -47.0364113, 57.8066292, -223.2950897, 188.3737946
3: -69.3965378, 166.8359833, -27.2729225, 55.0080185, -124.4045563, 194.1089020
4: -184.5755615, 140.5543518, -53.0424118, 57.1621284, -241.7376556, 193.5967407

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -88.7503510, 95.0967102, -241.1432800, 223.4841156
1: -114.2232513, 127.2852173, -69.4899979, 90.1278381, -204.3510590, 196.7751617
2: -165.4884644, 141.3374176, -100.9360809, 99.7855835, -265.2740479, 242.2734985
3: -69.3965378, 166.8359833, -48.5418625, 106.7330246, -176.1295624, 215.3645477
4: -184.5755615, 140.5543518, -113.2336960, 98.6521301, -283.2276306, 253.7880554

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274664, upper bound: 187.4329129
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263032, upper bound: 187.6268437
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -89.4479752, 89.8819046, -179.3298340, 179.3298492
1: -69.8412399, 84.1715393, -69.8412399, 84.1715393, -154.0127411, 154.0127411
2: -101.2739258, 94.6007919, -101.2739258, 94.6007919, -195.8747253, 195.8747101
3: -45.8811989, 105.0105896, -45.8811989, 105.0105896, -150.8917847, 150.8917847
4: -113.0568390, 94.2403870, -113.0568390, 94.2403870, -207.2972260, 207.2972260

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900308
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8149336
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -146.0465698, 134.7337646, -224.1817169, 235.9284668
1: -69.8412399, 84.1715393, -114.2232513, 127.2852173, -197.1264496, 198.3947754
2: -101.2739258, 94.6007919, -165.4884644, 141.3374176, -242.6113129, 260.0892639
3: -45.8811989, 105.0105896, -69.3965378, 166.8359833, -212.7171631, 174.4071350
4: -113.0568390, 94.2403870, -184.5755615, 140.5543518, -253.6111755, 278.8159485

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6302836, upper bound: 187.4900804
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291451, upper bound: 187.8150484
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -89.4479752, 89.8819046, -235.9284668, 224.1817017
1: -114.2232513, 127.2852173, -69.8412399, 84.1715393, -198.3947906, 197.1264343
2: -165.4884644, 141.3374176, -101.2739258, 94.6007919, -260.0892639, 242.6113281
3: -69.3965378, 166.8359833, -45.8811989, 105.0105896, -174.4071350, 212.7171631
4: -184.5755615, 140.5543518, -113.0568390, 94.2403870, -278.8159485, 253.6111755

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -146.0465698, 134.7337646, -280.7803345, 280.7803040
1: -114.2232513, 127.2852173, -114.2232513, 127.2852173, -241.5084686, 241.5084534
2: -165.4884644, 141.3374176, -165.4884644, 141.3374176, -306.8258667, 306.8258667
3: -69.3965378, 166.8359833, -69.3965378, 166.8359833, -236.2325134, 236.2325134
4: -184.5755615, 140.5543518, -184.5755615, 140.5543518, -325.1298828, 325.1299133

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303077, upper bound: 187.4349487
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291445, upper bound: 187.6273117
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -134.6475983, 125.2181320, -166.1991119, 189.4267120
1: -32.1188278, 51.3675804, -105.6099548, 118.3325500, -150.4513550, 156.9775238
2: -47.0364113, 57.8066292, -152.8045502, 131.4833069, -178.5197144, 210.6111450
3: -27.2729225, 55.0080185, -64.1565170, 155.1156464, -182.3885651, 119.1645355
4: -53.0424118, 57.1621284, -170.4454498, 130.4599762, -183.5023804, 227.6075745

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4657145, upper bound: 187.3900602
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4657145, upper bound: 187.8172907
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -141.3795471, 128.9066467, -169.8876343, 196.1586609
1: -32.1188278, 51.3675804, -110.7312317, 121.6841736, -153.8030090, 162.0988159
2: -47.0364113, 57.8066292, -160.2674408, 135.1351776, -182.1715546, 218.0740356
3: -27.2729225, 55.0080185, -66.0420303, 161.7633057, -189.0362244, 121.0500488
4: -53.0424118, 57.1621284, -178.6372070, 134.3881073, -187.4305115, 235.7993164

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257307, upper bound: 187.3900602
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6257307, upper bound: 187.8172907
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -134.6475983, 125.2181320, -213.9684753, 229.7443085
1: -69.4899979, 90.1278381, -105.6099548, 118.3325500, -187.8224945, 195.7377472
2: -100.9360809, 99.7855835, -152.8045502, 131.4833069, -232.4193878, 252.5901337
3: -48.5418625, 106.7330246, -64.1565170, 155.1156464, -203.6575012, 170.8895416
4: -113.2336960, 98.6521301, -170.4454498, 130.4599762, -243.6936646, 269.0975647

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4657145, upper bound: 187.3743426
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4658922, upper bound: 187.6263125
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -141.3795471, 128.9066467, -217.6569977, 236.4762421
1: -69.4899979, 90.1278381, -110.7312317, 121.6841736, -191.1741486, 200.8590698
2: -100.9360809, 99.7855835, -160.2674408, 135.1351776, -236.0712433, 260.0530090
3: -48.5418625, 106.7330246, -66.0420303, 161.7633057, -210.3051453, 172.7750244
4: -113.2336960, 98.6521301, -178.6372070, 134.3881073, -247.6217957, 277.2893066

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259132, upper bound: 187.3743426
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259132, upper bound: 187.6263125
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -143.6822052, 137.2854614, -176.8308258, 198.4613190
1: -32.1188278, 51.3675804, -112.6880493, 130.4040375, -160.4764862, 164.0556335
2: -47.0364113, 57.8066292, -163.4659729, 143.3750000, -188.0644836, 221.2725983
3: -27.2729225, 55.0080185, -69.9088745, 166.6474915, -193.9204102, 122.9190521
4: -53.0424118, 57.1621284, -182.6418304, 142.1457367, -193.7832642, 239.8039551

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3897308
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8169605
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -209.8448181, 182.5023346, -223.4833221, 264.6239014
1: -32.1188278, 51.3675804, -164.8178711, 172.6278076, -204.5114899, 216.1854553
2: -47.0364113, 57.8066292, -238.0657501, 190.4068451, -237.1847839, 295.8723450
3: -27.2729225, 55.0080185, -93.9250412, 235.7587585, -263.0316772, 147.7104645
4: -53.0424118, 57.1621284, -264.9623718, 190.2988586, -243.3412323, 322.1244812

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6247507, upper bound: 187.3949111
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6237433, upper bound: 187.8221408
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -143.9574738, 137.5898285, -225.3655853, 239.0541840
1: -69.4899979, 90.1278381, -112.9016113, 130.6958466, -198.4430542, 203.0294189
2: -100.9360809, 99.7855835, -163.7802124, 143.6897125, -242.7204132, 263.5657959
3: -48.5418625, 106.7330246, -70.0736237, 166.9585724, -215.5003815, 174.8963318
4: -113.2336960, 98.6521301, -182.9930725, 142.4557343, -254.8316803, 281.6452026

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249471, upper bound: 187.3718545
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239258, upper bound: 187.6234985
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -210.2807465, 183.0075378, -271.7578735, 305.3774414
1: -69.4899979, 90.1278381, -165.1631775, 173.1105804, -242.6005707, 255.2909851
2: -100.9360809, 99.7855835, -238.5625763, 190.9392853, -291.8753662, 338.3481445
3: -48.5418625, 106.7330246, -94.2064362, 236.2539673, -284.7958069, 199.7813416
4: -113.2336960, 98.6521301, -265.5162659, 190.8201752, -304.0538635, 364.1683960

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6249471, upper bound: 187.3745005
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239258, upper bound: 187.6263123
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -88.6077271, 92.8103714, -182.2582855, 178.4896240
1: -69.8412399, 84.1715393, -69.5185242, 87.8645935, -157.7057953, 153.6900330
2: -101.2739258, 94.6007919, -100.9003143, 97.3328018, -198.6067200, 195.5010986
3: -45.8811989, 105.0105896, -47.0367928, 106.5991135, -152.4803009, 152.0473785
4: -113.0568390, 94.2403870, -113.0341187, 96.2283783, -209.2852020, 207.2745056

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4874308
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068517, upper bound: 187.8144761
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -144.8092651, 131.6590424, -221.1069946, 234.6911469
1: -69.8412399, 84.1715393, -113.4549103, 124.2506790, -194.0919037, 197.6264496
2: -101.2739258, 94.6007919, -164.1941833, 138.0722961, -239.3461914, 258.7949829
3: -45.8811989, 105.0105896, -67.5839310, 165.3264618, -211.2076569, 172.5945129
4: -113.0568390, 94.2403870, -182.9160309, 137.3219604, -250.3787994, 277.1564331

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5079903, upper bound: 187.4899905
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068518, upper bound: 187.8144761
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -88.6077271, 92.8103714, -238.8569336, 223.3414917
1: -114.2232513, 127.2852173, -69.5185242, 87.8645935, -202.0878448, 196.8037415
2: -165.4884644, 141.3374176, -100.9003143, 97.3328018, -262.8212585, 242.2377319
3: -69.3965378, 166.8359833, -47.0367928, 106.5991135, -175.9956512, 213.8727570
4: -184.5755615, 140.5543518, -113.0341187, 96.2283783, -280.8039246, 253.5884705

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6263398
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -144.8092651, 131.6590424, -277.7055969, 279.5430298
1: -114.2232513, 127.2852173, -113.4549103, 124.2506790, -238.4739380, 240.7401276
2: -165.4884644, 141.3374176, -164.1941833, 138.0722961, -303.5607605, 305.5316162
3: -69.3965378, 166.8359833, -67.5839310, 165.3264618, -234.7229919, 234.4199219
4: -184.5755615, 140.5543518, -182.9160309, 137.3219604, -321.8974609, 323.4703979

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080144, upper bound: 187.4324091
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068512, upper bound: 187.6269514
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -143.9574738, 137.5898285, -225.9231720, 233.8393860
1: -69.8412399, 84.1715393, -112.9016113, 130.6958466, -198.6280518, 197.0731201
2: -101.2739258, 94.6007919, -163.7802124, 143.6897125, -242.6262970, 258.3810120
3: -45.8811989, 105.0105896, -70.0736237, 166.9585724, -212.8397369, 172.9512329
4: -113.0568390, 94.2403870, -182.9930725, 142.4557343, -254.2246857, 277.2334595

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4899152
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -210.2339325, 182.9532471, -272.4011841, 300.1158142
1: -69.8412399, 84.1715393, -165.1260986, 173.0586700, -242.8998871, 249.2976227
2: -101.2739258, 94.6007919, -238.5092468, 190.8820038, -292.1559448, 333.1100464
3: -45.8811989, 105.0105896, -94.1761780, 236.2008057, -282.0820007, 198.2106934
4: -113.0568390, 94.2403870, -265.4568176, 190.7641144, -303.8209534, 359.6972046

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271832, upper bound: 187.4946240
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260447, upper bound: 187.8169605
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -143.9574738, 137.5898285, -282.8959045, 278.6911926
1: -114.2232513, 127.2852173, -112.9016113, 130.6958466, -243.3010864, 240.1868286
2: -165.4884644, 141.3374176, -163.7802124, 143.6897125, -307.2244263, 305.1176147
3: -69.3965378, 166.8359833, -70.0736237, 166.9585724, -236.3551025, 234.8969727
4: -184.5755615, 140.5543518, -182.9930725, 142.4557343, -326.2515869, 323.5474243

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4324091
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260054, upper bound: 187.6263398
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -210.2807465, 183.0075378, -329.0541077, 345.0145264
1: -114.2232513, 127.2852173, -165.1631775, 173.1105804, -287.3338318, 292.4483643
2: -165.4884644, 141.3374176, -238.5625763, 190.9392853, -356.4277344, 379.8999939
3: -69.3965378, 166.8359833, -94.2064362, 236.2539673, -305.6505127, 260.1801758
4: -184.5755615, 140.5543518, -265.5162659, 190.8201752, -375.3957520, 406.0706177

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272073, upper bound: 187.4324091
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260055, upper bound: 187.6269514
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -134.6475983, 125.2181320, -40.9809875, 54.7791138, -189.4267120, 166.1991119
1: -105.6099548, 118.3325500, -32.1188278, 51.3675804, -156.9775391, 150.4513550
2: -152.8045502, 131.4833069, -47.0364113, 57.8066292, -210.6111450, 178.5197144
3: -64.1565170, 155.1156464, -27.2729225, 55.0080185, -119.1645355, 182.3885651
4: -170.4454498, 130.4599762, -53.0424118, 57.1621284, -227.6075745, 183.5023804

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263182, upper bound: 187.4655715
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265921, upper bound: 187.4574827
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -134.6475983, 125.2181320, -88.7503510, 95.0967102, -229.7442932, 213.9684753
1: -105.6099548, 118.3325500, -69.4899979, 90.1278381, -195.7377472, 187.8225098
2: -152.8045502, 131.4833069, -100.9360809, 99.7855835, -252.5901337, 232.4193878
3: -64.1565170, 155.1156464, -48.5418625, 106.7330246, -170.8895416, 203.6575012
4: -170.4454498, 130.4599762, -113.2336960, 98.6521301, -269.0975647, 243.6936646

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263182, upper bound: 187.4657572
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265921, upper bound: 187.4576770
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -141.3795471, 128.9066467, -40.9809875, 54.7791138, -196.1586609, 169.8876343
1: -110.7312317, 121.6841736, -32.1188278, 51.3675804, -162.0988159, 153.8030090
2: -160.2674408, 135.1351776, -47.0364113, 57.8066292, -218.0740356, 182.1715393
3: -66.0420303, 161.7633057, -27.2729225, 55.0080185, -121.0500336, 189.0362244
4: -178.6372070, 134.3881073, -53.0424118, 57.1621284, -235.7993317, 187.4305115

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5045501
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.6257306
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -141.3795471, 128.9066467, -88.7503510, 95.0967102, -236.4762268, 217.6569977
1: -110.7312317, 121.6841736, -69.4899979, 90.1278381, -200.8590546, 191.1741486
2: -160.2674408, 135.1351776, -100.9360809, 99.7855835, -260.0529785, 236.0712585
3: -66.0420303, 161.7633057, -48.5418625, 106.7330246, -172.7750397, 210.3051453
4: -178.6372070, 134.3881073, -113.2336960, 98.6521301, -277.2893066, 247.6217957

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.5047325
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6234987, upper bound: 187.6259131
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -89.4479752, 89.8819046, -178.4896240, 182.2583160
1: -69.5185242, 87.8645935, -69.8412399, 84.1715393, -153.6900330, 157.7058105
2: -100.9003143, 97.3328018, -101.2739258, 94.6007919, -195.5010986, 198.6067047
3: -47.0367928, 106.5991135, -45.8811989, 105.0105896, -152.0473785, 152.4803009
4: -113.0341187, 96.2283783, -113.0568390, 94.2403870, -207.2745056, 209.2851868

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -146.0465698, 134.7337646, -223.3414917, 238.8569336
1: -69.5185242, 87.8645935, -114.2232513, 127.2852173, -196.8037262, 202.0878296
2: -100.9003143, 97.3328018, -165.4884644, 141.3374176, -242.2377319, 262.8212585
3: -47.0367928, 106.5991135, -69.3965378, 166.8359833, -213.8727417, 175.9956512
4: -113.0341187, 96.2283783, -184.5755615, 140.5543518, -253.5884705, 280.8038940

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6270404, upper bound: 187.3663691
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263399, upper bound: 187.5068508
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -89.4479752, 89.8819046, -234.6911621, 221.1069946
1: -113.4549103, 124.2506790, -69.8412399, 84.1715393, -197.6264496, 194.0919189
2: -164.1941833, 138.0722961, -101.2739258, 94.6007919, -258.7949829, 239.3461914
3: -67.5839310, 165.3264618, -45.8811989, 105.0105896, -172.5945129, 211.2076569
4: -182.9160309, 137.3219604, -113.0568390, 94.2403870, -277.1564331, 250.3787842

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6263070
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -146.0465698, 134.7337646, -279.5430298, 277.7056274
1: -113.4549103, 124.2506790, -114.2232513, 127.2852173, -240.7401276, 238.4739380
2: -164.1941833, 138.0722961, -165.4884644, 141.3374176, -305.5316162, 303.5607605
3: -67.5839310, 165.3264618, -69.3965378, 166.8359833, -234.4199219, 234.7229919
4: -182.9160309, 137.3219604, -184.5755615, 140.5543518, -323.4703979, 321.8974609

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303010, upper bound: 187.4680153
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6291537, upper bound: 187.6264775
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -143.6822052, 137.2854614, -40.9809875, 54.7791138, -198.4613190, 176.8308411
1: -112.6880493, 130.4040375, -32.1188278, 51.3675804, -164.0556335, 160.4764709
2: -163.4659729, 143.3750000, -47.0364113, 57.8066292, -221.2725983, 188.0644836
3: -69.9088745, 166.6474915, -27.2729225, 55.0080185, -122.9190445, 193.9204102
4: -182.6418304, 142.1457367, -53.0424118, 57.1621284, -239.8039551, 193.7832642

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6285358, upper bound: 187.3947660
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237432
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -88.7503510, 95.0967102, -239.0541840, 225.3655853
1: -112.9016113, 130.6958466, -69.4899979, 90.1278381, -203.0294342, 198.4430542
2: -163.7802124, 143.6897125, -100.9360809, 99.7855835, -263.5657654, 242.7203979
3: -70.0736237, 166.9585724, -48.5418625, 106.7330246, -174.8963318, 215.5003967
4: -182.9930725, 142.4557343, -113.2336960, 98.6521301, -281.6452026, 254.8317108

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6241992, upper bound: 187.3949478
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266275, upper bound: 187.6237582
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -209.8448181, 182.5023346, -40.9809875, 54.7791138, -264.6239319, 223.4833221
1: -164.8178711, 172.6278076, -32.1188278, 51.3675804, -216.1854553, 204.5114899
2: -238.0657501, 190.4068451, -47.0364113, 57.8066292, -295.8723450, 237.1847687
3: -93.9250412, 235.7587585, -27.2729225, 55.0080185, -147.7104645, 263.0316772
4: -264.9623718, 190.2988586, -53.0424118, 57.1621284, -322.1244812, 243.3412628

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4947344
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264760
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -88.7503510, 95.0967102, -305.3774414, 271.7578735
1: -165.1631775, 173.1105804, -69.4899979, 90.1278381, -255.2909698, 242.6005554
2: -238.5625763, 190.9392853, -100.9360809, 99.7855835, -338.3481445, 291.8753662
3: -94.2064362, 236.2539673, -48.5418625, 106.7330246, -199.7813416, 284.7958374
4: -265.5162659, 190.8201752, -113.2336960, 98.6521301, -364.1683960, 304.0538330

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6358173, upper bound: 187.4949169
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6311633, upper bound: 187.6264993
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -89.4479752, 89.8819046, -233.8393860, 225.9231720
1: -112.9016113, 130.6958466, -69.8412399, 84.1715393, -197.0731201, 198.6280212
2: -163.7802124, 143.6897125, -101.2739258, 94.6007919, -258.3810120, 242.6263123
3: -70.0736237, 166.9585724, -45.8811989, 105.0105896, -172.9512177, 212.8397522
4: -182.9930725, 142.4557343, -113.0568390, 94.2403870, -277.2334595, 254.2246857

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970662
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260439
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -146.0465698, 134.7337646, -278.6912231, 282.8958740
1: -112.9016113, 130.6958466, -114.2232513, 127.2852173, -240.1868134, 243.3010559
2: -163.7802124, 143.6897125, -165.4884644, 141.3374176, -305.1176147, 307.2244568
3: -70.0736237, 166.9585724, -69.3965378, 166.8359833, -234.8969727, 236.3550873
4: -182.9930725, 142.4557343, -184.5755615, 140.5543518, -323.5474243, 326.2516174

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6313771, upper bound: 187.3970662
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294688, upper bound: 187.6260440
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -210.2339325, 182.9532471, -89.4479752, 89.8819046, -300.1158142, 272.4011841
1: -165.1260986, 173.0586700, -69.8412399, 84.1715393, -249.2976227, 242.8999023
2: -238.5092468, 190.8820038, -101.2739258, 94.6007919, -333.1100464, 292.1559448
3: -94.1761780, 236.2008057, -45.8811989, 105.0105896, -198.2106934, 282.0820007
4: -265.4568176, 190.7641144, -113.0568390, 94.2403870, -359.6972046, 303.8209534

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969826
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6340045, upper bound: 187.6269505
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -146.0465698, 134.7337646, -345.0144653, 329.0541077
1: -165.1631775, 173.1105804, -114.2232513, 127.2852173, -292.4483643, 287.3338318
2: -238.5625763, 190.9392853, -165.4884644, 141.3374176, -379.8999939, 356.4277344
3: -94.2064362, 236.2539673, -69.3965378, 166.8359833, -260.1801758, 305.6505127
4: -265.5162659, 190.8201752, -184.5755615, 140.5543518, -406.0706177, 375.3957520

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6386586, upper bound: 187.4969978
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6340045, upper bound: 187.6269785
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -141.3795471, 128.9066467, -217.5143585, 234.1899109
1: -69.5185242, 87.8645935, -110.7312317, 121.6841736, -191.2026978, 198.5957947
2: -100.9003143, 97.3328018, -160.2674408, 135.1351776, -236.0354767, 257.6002197
3: -47.0367928, 106.5991135, -66.0420303, 161.7633057, -208.8000946, 172.6411438
4: -113.0341187, 96.2283783, -178.6372070, 134.3881073, -247.4222260, 274.8655701

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252269, upper bound: 187.3663697
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252269, upper bound: 187.5068518
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -88.6077271, 92.8103714, -237.6196289, 220.2667694
1: -113.4549103, 124.2506790, -69.5185242, 87.8645935, -201.3195038, 193.7691956
2: -164.1941833, 138.0722961, -100.9003143, 97.3328018, -261.5269775, 238.9726105
3: -67.5839310, 165.3264618, -47.0367928, 106.5991135, -174.1830444, 212.3632507
4: -182.9160309, 137.3219604, -113.0341187, 96.2283783, -279.1444092, 250.3560791

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080076, upper bound: 187.4652107
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6252268
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -144.8092651, 131.6590424, -276.4683228, 276.4683228
1: -113.4549103, 124.2506790, -113.4549103, 124.2506790, -237.7055969, 237.7055969
2: -164.1941833, 138.0722961, -164.1941833, 138.0722961, -302.2664795, 302.2664795
3: -67.5839310, 165.3264618, -67.5839310, 165.3264618, -232.9104004, 232.9104004
4: -182.9160309, 137.3219604, -182.9160309, 137.3219604, -320.2379456, 320.2379150

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5080077, upper bound: 187.4680246
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5068604, upper bound: 187.6259756
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -143.9574738, 137.5898285, -225.2692871, 236.7678528
1: -69.5185242, 87.8645935, -112.9016113, 130.6958466, -198.5025024, 200.7661743
2: -100.9003143, 97.3328018, -163.7802124, 143.6897125, -242.7417755, 261.1130066
3: -47.0367928, 106.5991135, -70.0736237, 166.9585724, -213.9953613, 174.8139343
4: -113.0341187, 96.2283783, -182.9930725, 142.4557343, -254.6831818, 279.2214355

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3641368
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5046185
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -210.2807465, 183.0075378, -271.6152344, 303.0911255
1: -69.5185242, 87.8645935, -165.1631775, 173.1105804, -242.6291046, 253.0277100
2: -100.9003143, 97.3328018, -238.5625763, 190.9392853, -291.8395996, 335.8953857
3: -47.0367928, 106.5991135, -94.2064362, 236.2539673, -283.2907715, 199.6989594
4: -113.0341187, 96.2283783, -265.5162659, 190.8201752, -303.8543091, 361.7446289

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239400, upper bound: 187.3663691
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6232395, upper bound: 187.5068508
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -143.9574738, 137.5898285, -281.7225647, 275.6165161
1: -113.4549103, 124.2506790, -112.9016113, 130.6958466, -242.5797119, 237.1522827
2: -164.1941833, 138.0722961, -163.7802124, 143.6897125, -306.0038452, 301.8525085
3: -67.5839310, 165.3264618, -70.0736237, 166.9585724, -234.5424957, 233.4506531
4: -182.9160309, 137.3219604, -182.9930725, 142.4557343, -324.6633911, 320.3149719

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4659287
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6257991
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -210.2807465, 183.0075378, -327.8168030, 341.9397888
1: -113.4549103, 124.2506790, -165.1631775, 173.1105804, -286.5654907, 289.4138489
2: -164.1941833, 138.0722961, -238.5625763, 190.9392853, -355.1334839, 376.6348877
3: -67.5839310, 165.3264618, -94.2064362, 236.2539673, -303.8377991, 258.7338257
4: -182.9160309, 137.3219604, -265.5162659, 190.8201752, -373.7362061, 402.8381958

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272006, upper bound: 187.4684190
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260533, upper bound: 187.6264775
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -134.6475983, 125.2181320, -269.1755981, 271.5823364
1: -112.9016113, 130.6958466, -105.6099548, 118.3325500, -231.2341461, 234.7382965
2: -163.7802124, 143.6897125, -152.8045502, 131.4833069, -295.2635193, 294.7548218
3: -70.0736237, 166.9585724, -64.1565170, 155.1156464, -223.3546143, 231.1150665
4: -182.9930725, 142.4557343, -170.4454498, 130.4599762, -313.4530640, 312.2597656

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4682701, upper bound: 187.3970754
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4682701, upper bound: 187.6260447
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -141.3795471, 128.9066467, -272.8641052, 278.4877014
1: -112.9016113, 130.6958466, -110.7312317, 121.6841736, -234.5857697, 239.9792786
2: -163.7802124, 143.6897125, -160.2674408, 135.1351776, -298.9154053, 302.3860168
3: -70.0736237, 166.9585724, -66.0420303, 161.7633057, -230.0995026, 233.0005951
4: -182.9930725, 142.4557343, -178.6372070, 134.3881073, -317.3811646, 320.6518555

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283558, upper bound: 187.3970758
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283558, upper bound: 187.6260447
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -88.6077271, 92.8103714, -303.0911255, 271.6152649
1: -165.1631775, 173.1105804, -69.5185242, 87.8645935, -253.0277405, 242.6291046
2: -238.5625763, 190.9392853, -100.9003143, 97.3328018, -335.8953552, 291.8395996
3: -94.2064362, 236.2539673, -47.0367928, 106.5991135, -199.6989594, 283.2907410
4: -265.5162659, 190.8201752, -113.0341187, 96.2283783, -361.7446289, 303.8543091

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5163652, upper bound: 187.4942305
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5117112, upper bound: 187.6259722
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -144.8092651, 131.6590424, -341.9397888, 327.8168030
1: -165.1631775, 173.1105804, -113.4549103, 124.2506790, -289.4138489, 286.5654907
2: -238.5625763, 190.9392853, -164.1941833, 138.0722961, -376.6348877, 355.1334839
3: -94.2064362, 236.2539673, -67.5839310, 165.3264618, -258.7338867, 303.8378296
4: -265.5162659, 190.8201752, -182.9160309, 137.3219604, -402.8381958, 373.7362061

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5163653, upper bound: 187.4969829
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5117113, upper bound: 187.6265915
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -143.9574738, 137.5898285, -280.9462280, 280.9462280
1: -112.9016113, 130.6958466, -112.9016113, 130.6958466, -242.0519104, 242.0519257
2: -163.7802124, 143.6897125, -163.7802124, 143.6897125, -305.8948975, 305.8948975
3: -70.0736237, 166.9585724, -70.0736237, 166.9585724, -235.3055420, 235.3055267
4: -182.9930725, 142.4557343, -182.9930725, 142.4557343, -324.9454956, 324.9454651

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282766, upper bound: 187.3961512
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263684, upper bound: 187.6233484
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -210.2807465, 183.0075378, -326.9650269, 347.5987854
1: -112.9016113, 130.6958466, -165.1631775, 173.1105804, -286.0121765, 294.5191040
2: -163.7802124, 143.6897125, -238.5625763, 190.9392853, -354.7194824, 380.6860962
3: -70.0736237, 166.9585724, -94.2064362, 236.2539673, -304.4873352, 260.1905212
4: -182.9930725, 142.4557343, -265.5162659, 190.8201752, -373.8132324, 407.7260132

Time for backsubstitution: 2.08 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=232.61239624023438
rel_dist={3: [-187.91820645300623, 187.91820645300623]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6713976, upper bound: 187.8581898
time: 0.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.44 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 3, lower bound: -187.6713976, upper bound: 187.8581898
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -149.6440735, 126.7424088, -246.6997375, 258.3909607
1: -93.8448410, 101.7164536, -117.3338928, 118.4335785, -212.2784119, 219.0503540
2: -135.7941437, 113.6401825, -169.7016296, 131.6250763, -267.4192200, 283.3417969
3: -54.4526558, 137.4407501, -63.3496017, 169.2627869, -223.7154388, 200.7903442
4: -151.2667084, 114.1601410, -188.6523895, 133.4867859, -284.7534485, 302.8125305

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6755892
time: 0.62 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -148.4383087, 125.9597473, -308.1200562, 302.0988159
1: -143.0398865, 144.6755829, -116.3920288, 117.7056580, -260.7455444, 261.0676270
2: -206.5966339, 159.9870605, -168.3444672, 130.8322754, -337.4288940, 328.3315430
3: -78.3434219, 204.8266907, -62.9577904, 168.0063171, -246.3497314, 267.7844543
4: -229.7526093, 160.7959442, -187.1445160, 132.6473389, -362.3999329, 347.9404602

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6713976
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.04 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6713976
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.6713976, upper bound: 187.6755892
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6713976
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.04
Output dim: 3, lower bound: -187.6755892, upper bound: 187.6755892

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6661122, upper bound: 187.8573601
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6709226, upper bound: 187.8573601
time: 0.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -182.1602936, 153.6605072, -273.6178284, 290.9071655
1: -93.8448410, 101.7164536, -143.0398865, 144.6755829, -238.5203857, 244.7563477
2: -135.7941437, 113.6401825, -206.5966339, 159.9870605, -295.7811584, 320.2367859
3: -54.4526558, 137.4407501, -78.3434219, 204.8266907, -259.2793579, 215.7841492
4: -151.2667084, 114.1601410, -229.7526093, 160.7959442, -312.0625916, 343.9127197

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6661122, upper bound: 187.8573601
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6709226, upper bound: 187.8573601
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -119.9573212, 108.7468872, -290.9071655, 273.6178284
1: -143.0398865, 144.6755829, -93.8448410, 101.7164536, -244.7563477, 238.5203857
2: -206.5966339, 159.9870605, -135.7941437, 113.6401825, -320.2368164, 295.7811584
3: -78.3434219, 204.8266907, -54.4526558, 137.4407501, -215.7841339, 259.2793579
4: -229.7526093, 160.7959442, -151.2667084, 114.1601410, -343.9127502, 312.0626221

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6299901
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839
time: 0.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -182.1602936, 153.6605072, -335.8208008, 335.8208008
1: -143.0398865, 144.6755829, -143.0398865, 144.6755829, -287.7154541, 287.7154541
2: -206.5966339, 159.9870605, -206.5966339, 159.9870605, -366.5836792, 366.5836792
3: -78.3434219, 204.8266907, -78.3434219, 204.8266907, -283.1701050, 283.1701050
4: -229.7526093, 160.7959442, -229.7526093, 160.7959442, -390.5485229, 390.5485535

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6299901
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839
time: 0.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6661122, upper bound: 187.8573601
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6709226, upper bound: 187.8573601
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6661122, upper bound: 187.8573601
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6709226, upper bound: 187.8573601
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6299901
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6307684, upper bound: 187.6299901
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 3, lower bound: -187.6375296, upper bound: 187.6307839

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -106.1626282, 97.4386749, -160.0376282, 174.8440094
1: -49.0822411, 64.2613220, -83.0551147, 91.3490372, -140.4312744, 147.3164368
2: -71.3749008, 71.9796066, -120.2491455, 101.6269455, -173.0018463, 192.2286835
3: -33.6786270, 77.7695847, -48.5460625, 122.6407471, -156.3193665, 126.3156204
4: -79.9356384, 71.8462219, -133.9883881, 102.2042465, -182.1398773, 205.8346100

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6285558
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -119.7582397, 108.6050873, -224.3563995, 225.5090332
1: -90.5305862, 98.8567810, -93.6877594, 101.5805969, -192.1111755, 192.5444794
2: -131.0461578, 110.5165710, -135.5686035, 113.4887772, -244.5349274, 246.0851440
3: -52.9895172, 132.8755493, -54.3816757, 137.2239532, -190.2134705, 187.2572327
4: -145.9682770, 110.9683304, -151.0151978, 114.0073700, -259.9756470, 261.9835205

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8012999, upper bound: 187.6309205
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -166.6749878, 141.1701660, -203.7691193, 235.3563843
1: -49.0822411, 64.2613220, -130.8176880, 133.2319183, -182.3141632, 195.0790100
2: -71.3749008, 71.9796066, -189.1006165, 146.8185883, -218.1934814, 261.0802307
3: -33.6786270, 77.7695847, -71.6876678, 188.3006744, -221.9792786, 149.4572296
4: -79.9356384, 71.8462219, -210.3767548, 147.4294434, -227.3650818, 282.2229614

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269352, upper bound: 187.6286071
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277222, upper bound: 187.6341480
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -181.9644928, 153.5216522, -269.2729187, 287.7153015
1: -90.5305862, 98.8567810, -142.8848877, 144.5434418, -235.0740204, 241.7416687
2: -131.0461578, 110.5165710, -206.3749084, 159.8457184, -290.8918762, 316.8914490
3: -52.9895172, 132.8755493, -78.2761917, 204.6129150, -257.6024170, 211.1517181
4: -145.9682770, 110.9683304, -229.5056152, 160.6483917, -306.6166077, 340.4739380

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303795, upper bound: 187.6370517
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -119.9573212, 108.7468872, -258.0307922, 254.6466827
1: -116.9905624, 127.1221771, -93.8448410, 101.7164536, -218.7070160, 220.9670105
2: -169.2326202, 141.1778259, -135.7941437, 113.6401825, -282.8728027, 276.9718933
3: -69.0598907, 170.1712341, -54.4526558, 137.4407501, -206.5006256, 224.6238861
4: -188.5170746, 140.5437622, -151.2667084, 114.1601410, -302.6772156, 291.8104248

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286071, upper bound: 187.6269352
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6309569, upper bound: 187.6293981
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -119.0552597, 108.2171707, -322.9401245, 305.0284729
1: -168.6910553, 175.9269562, -93.1325226, 101.2112274, -269.9022827, 269.0594482
2: -243.5518188, 193.9766388, -134.7739410, 113.0966644, -356.6484680, 328.7505493
3: -95.6573029, 241.0182190, -54.1846466, 136.5119934, -231.5868530, 295.2027893
4: -271.0773621, 193.9729767, -150.1368103, 113.6082687, -384.6856079, 344.1097412

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6341480, upper bound: 187.6277222
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6370517, upper bound: 187.6303795
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -182.1602936, 153.6605072, -302.9444580, 316.8496704
1: -116.9905624, 127.1221771, -143.0398865, 144.6755829, -261.6661072, 270.1620178
2: -169.2326202, 141.1778259, -206.5966339, 159.9870605, -329.2196655, 347.7744141
3: -69.0598907, 170.1712341, -78.3434219, 204.8266907, -273.8865967, 248.5146027
4: -188.5170746, 140.5437622, -229.7526093, 160.7959442, -349.3130188, 370.2963867

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6299901
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6299901
time: 0.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -214.7231140, 185.9732056, -181.5190277, 153.3090973, -368.0322266, 367.4922485
1: -168.6910553, 175.9269562, -142.5338440, 144.3416901, -313.0327454, 318.4607849
2: -243.5518188, 193.9766388, -205.8718872, 159.6268616, -403.1786804, 399.8485107
3: -95.6573029, 241.0182190, -78.1683502, 204.1779022, -299.4284973, 319.1865540
4: -271.0773621, 193.9729767, -228.9604187, 160.4309540, -431.5083008, 422.9333801

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6338902, upper bound: 187.6277222
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6303795
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.22 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.7987681, upper bound: 187.6285558
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6280994, upper bound: 187.6282582
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.8012999, upper bound: 187.6309205
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6269352, upper bound: 187.6286071
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6277222, upper bound: 187.6341480
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6303795, upper bound: 187.6370517
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6286071, upper bound: 187.6269352
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6309569, upper bound: 187.6293981
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6341480, upper bound: 187.6277222
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6370517, upper bound: 187.6303795
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6299901
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6300054, upper bound: 187.6299901
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6338902, upper bound: 187.6277222
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -187.6366910, upper bound: 187.6303795

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -79.7371521, 81.4387817, -144.0377197, 148.4185333
1: -49.0822411, 64.2613220, -62.2732544, 76.2692413, -125.3514862, 126.5345612
2: -71.3749008, 71.9796066, -90.3249054, 85.6733170, -157.0482178, 162.3045044
3: -33.6786270, 77.7695847, -41.1421471, 94.7014923, -128.3801117, 118.9117279
4: -79.9356384, 71.8462219, -100.9268417, 85.4481277, -165.3837585, 172.7730713

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279461, upper bound: 187.6282582
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279461, upper bound: 187.6282582
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -61.9585381, 68.2883911, -135.9420776, 125.6306152, -187.5891266, 204.2304382
1: -48.5780029, 63.8934555, -106.3213120, 118.7759705, -167.3539734, 170.2147522
2: -70.6596298, 71.5704880, -154.0769043, 131.6185913, -202.2781830, 225.6473999
3: -33.4631042, 77.1321182, -64.5234070, 156.0122223, -189.4753265, 141.6554871
4: -79.1433411, 71.4262695, -171.9312897, 131.0277863, -210.1710968, 243.3575592

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6271053, upper bound: 187.3741556
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269766, upper bound: 187.6257837
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -93.3917847, 92.7331238, -208.4844208, 199.1425781
1: -90.5305862, 98.8567810, -72.9543228, 86.8698502, -177.4004364, 171.8110504
2: -131.0461578, 110.5165710, -105.7210693, 97.5862808, -228.6324463, 216.2376251
3: -52.9895172, 132.8755493, -47.2158279, 109.3487320, -162.3382568, 180.0913696
4: -145.9682770, 110.9683304, -118.0184097, 97.2374649, -243.2057190, 228.9867401

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -114.8312531, 105.2043762, -149.7934418, 137.2201385, -252.0513916, 254.9978180
1: -89.8055115, 98.3340836, -117.1645584, 129.6675110, -219.4730225, 215.4986420
2: -130.0062866, 109.9501648, -169.6975098, 143.9083099, -273.9146118, 279.6476746
3: -52.7117538, 131.9280548, -70.6193466, 170.8952637, -223.6069794, 202.5473938
4: -144.8175507, 110.3987961, -189.2721252, 143.1866150, -288.0041504, 299.6708984

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350329
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6289659, upper bound: 187.6289659
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -134.2789459, 122.3004990, -184.8993835, 202.9603424
1: -49.0822411, 64.2613220, -105.1659775, 115.5536118, -164.6358490, 169.4272919
2: -71.3749008, 71.9796066, -152.2682495, 128.1363678, -199.5112610, 224.2478333
3: -33.6786270, 77.7695847, -62.5633125, 154.0523529, -187.7309723, 140.3329010
4: -79.9356384, 71.8462219, -169.7571564, 127.5054779, -207.4411163, 241.6033783

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267795, upper bound: 187.6286071
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267795, upper bound: 187.6286071
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -61.9585381, 68.2883911, -197.8566284, 172.1485291, -234.1070709, 266.1450195
1: -48.5780029, 63.8934555, -155.3006287, 163.0055389, -211.5835419, 219.1940765
2: -70.6596298, 71.5704880, -224.5160675, 179.4752197, -250.1347961, 296.0865479
3: -33.4631042, 77.1321182, -88.3512268, 222.9403076, -256.4034119, 164.4918671
4: -79.1433411, 71.4262695, -250.0165558, 179.3800049, -258.5233459, 321.4428101

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267375, upper bound: 187.3784863
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266088, upper bound: 187.6301358
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -149.0803223, 134.5512085, -250.3024902, 254.8311157
1: -90.5305862, 98.8567810, -116.8296204, 126.9912109, -217.5217743, 215.6863708
2: -131.0461578, 110.5165710, -169.0030518, 141.0368652, -272.0830078, 279.5196228
3: -52.9895172, 132.8755493, -68.9912720, 169.9507294, -222.9402466, 201.8668060
4: -145.9682770, 110.9683304, -188.2618713, 140.3968811, -286.3651123, 299.2301941

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -114.8312531, 105.2043762, -214.4722443, 185.7803345, -300.6115417, 319.6766357
1: -89.8055115, 98.3340836, -168.4919128, 175.7428894, -265.5484009, 266.8259888
2: -130.0062866, 109.9501648, -243.2691193, 193.7771149, -323.7833862, 353.2192993
3: -52.7117538, 131.9280548, -95.5593414, 240.7458954, -293.4576416, 226.6427002
4: -144.8175507, 110.3987961, -270.7622681, 193.7694244, -338.5869751, 381.1610413

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6294468, upper bound: 187.4398745
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286043, upper bound: 187.6335490
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -134.2789459, 122.3004990, -62.5989532, 68.6813965, -202.9603424, 184.8993988
1: -105.1659775, 115.5536118, -49.0822411, 64.2613220, -169.4273071, 164.6358490
2: -152.2682495, 128.1363678, -71.3749008, 71.9796066, -224.2478485, 199.5112610
3: -62.5633125, 154.0523529, -33.6786270, 77.7695847, -140.3329010, 187.7309723
4: -169.7571564, 127.5054779, -79.9356384, 71.8462219, -241.6033783, 207.4411163

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6258460
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -149.0803223, 134.5512085, -115.7513199, 105.7508011, -254.8311157, 250.3025208
1: -116.8296204, 126.9912109, -90.5305862, 98.8567810, -215.6863861, 217.5217896
2: -169.0030518, 141.0368652, -131.0461578, 110.5165710, -279.5196228, 272.0830078
3: -68.9912720, 169.9507294, -52.9895172, 132.8755493, -201.8668060, 222.9402466
4: -188.2618713, 140.3968811, -145.9682770, 110.9683304, -299.2301941, 286.3651123

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274614, upper bound: 187.5290925
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6274614, upper bound: 187.6293981
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -197.8566284, 172.1485291, -61.9585381, 68.2883911, -266.1450195, 234.1070404
1: -155.3006287, 163.0055389, -48.5780029, 63.8934555, -219.1940765, 211.5835419
2: -224.5160675, 179.4752197, -70.6596298, 71.5704880, -296.0865479, 250.1348419
3: -88.3512268, 222.9403076, -33.4631042, 77.1321182, -164.4918671, 256.4034119
4: -250.0165558, 179.3800049, -79.1433411, 71.4262695, -321.4428101, 258.5233459

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324278, upper bound: 187.4871570
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6301360, upper bound: 187.6266087
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -214.4722443, 185.7803345, -114.8312531, 105.2043762, -319.6766357, 300.6115417
1: -168.4919128, 175.7428894, -89.8055115, 98.3340836, -266.8259888, 265.5484009
2: -243.2691193, 193.7771149, -130.0062866, 109.9501648, -353.2192993, 323.7833862
3: -95.5593414, 240.7458954, -52.7117538, 131.9280548, -226.6427002, 293.4576416
4: -270.7622681, 193.7694244, -144.8175507, 110.3987961, -381.1610413, 338.5869751

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6303796
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -149.2839355, 134.6893616, -283.9732361, 283.9732056
1: -116.9905624, 127.1221771, -116.9905624, 127.1221771, -244.1127319, 244.1127167
2: -169.2326202, 141.1778259, -169.2326202, 141.1778259, -310.4104614, 310.4104614
3: -69.0598907, 170.1712341, -69.0598907, 170.1712341, -239.2311096, 239.2311096
4: -188.5170746, 140.5437622, -188.5170746, 140.5437622, -329.0608215, 329.0608215

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.5290925
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298520, upper bound: 187.6293981
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -149.2839355, 134.6893616, -214.7231140, 185.9732056, -335.2571411, 349.4124451
1: -116.9905624, 127.1221771, -168.6910553, 175.9269562, -292.9174500, 295.8131714
2: -169.2326202, 141.1778259, -243.5518188, 193.9766388, -363.2092590, 384.7296143
3: -69.0598907, 170.1712341, -95.6573029, 241.0182190, -310.0781250, 265.3165283
4: -188.5170746, 140.5437622, -271.0773621, 193.9729767, -382.4900208, 411.6211243

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263038, upper bound: 187.5290925
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298520, upper bound: 187.6293981
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -198.5921936, 172.9775543, -115.5919189, 107.9167938, -306.5089417, 288.5694580
1: -155.8728333, 163.7884216, -90.7411270, 102.0026550, -257.8754883, 254.5295105
2: -225.3496094, 180.3471985, -131.4843903, 112.6674042, -338.0170288, 311.8315430
3: -88.8018494, 223.7629547, -54.3718262, 135.0808563, -222.9420319, 278.1347656
4: -250.9470367, 180.2328949, -146.7601929, 112.2500839, -363.1971130, 326.9931030

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6321695, upper bound: 187.4915734
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6298772, upper bound: 187.6266087
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -214.5238037, 185.8393707, -177.1518250, 150.2636261, -364.7873840, 362.9912109
1: -168.5326691, 175.7993317, -139.0804749, 141.4425354, -309.9751282, 314.8798218
2: -243.3277893, 193.8392792, -200.9367218, 156.5137939, -399.8415222, 394.7760010
3: -95.5921478, 240.8043213, -76.6745148, 199.4432068, -294.3564758, 317.4788208
4: -270.8276978, 193.8303833, -223.4640808, 157.1941223, -428.0218201, 417.2944641

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300127, upper bound: 187.6279995
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300127, upper bound: 187.6303796
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.33 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6279461, upper bound: 187.6282582
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6279461, upper bound: 187.6282582
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6271053, upper bound: 187.3741556
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6269766, upper bound: 187.6257837
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6307496, upper bound: 187.6307497
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6298076, upper bound: 187.4350329
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6289659, upper bound: 187.6289659
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6267795, upper bound: 187.6286071
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6267795, upper bound: 187.6286071
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6267375, upper bound: 187.3784863
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6266088, upper bound: 187.6301358
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6293981, upper bound: 187.6309569
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6294468, upper bound: 187.4398745
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6286043, upper bound: 187.6335490
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6258460
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6274614, upper bound: 187.5290925
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6274614, upper bound: 187.6293981
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6324278, upper bound: 187.4871570
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6301360, upper bound: 187.6266087
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6279995
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6303900, upper bound: 187.6303796
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6263038, upper bound: 187.5290925
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6298520, upper bound: 187.6293981
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6263038, upper bound: 187.5290925
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6298520, upper bound: 187.6293981
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6321695, upper bound: 187.4915734
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6298772, upper bound: 187.6266087
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6300127, upper bound: 187.6279995
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.33
Output dim: 3, lower bound: -187.6300127, upper bound: 187.6303796

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -79.7371521, 81.4387817, -122.4197693, 134.5162659
1: -32.1188278, 51.3675804, -62.2732544, 76.2692413, -108.3880386, 113.6408234
2: -47.0364113, 57.8066292, -90.3249054, 85.6733170, -132.7097168, 148.1315308
3: -27.2729225, 55.0080185, -41.1421471, 94.7014923, -121.9744034, 96.1501617
4: -53.0424118, 57.1621284, -100.9268417, 85.4481277, -138.4905243, 158.0889587

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6268356
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6259194
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -79.7371521, 81.4387817, -170.1891327, 174.8338318
1: -69.4899979, 90.1278381, -62.2732544, 76.2692413, -145.7592316, 152.4010925
2: -100.9360809, 99.7855835, -90.3249054, 85.6733170, -186.6093903, 190.1104889
3: -48.5418625, 106.7330246, -41.1421471, 94.7014923, -143.2433319, 147.8751526
4: -113.2336960, 98.6521301, -100.9268417, 85.4481277, -198.6817932, 199.5789490

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6268356
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6259194
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -53.8375549, 63.7631683, -135.8187714, 125.5687408, -179.4062958, 199.5819397
1: -42.4029007, 59.6174088, -106.2268372, 118.7171326, -161.1200256, 165.8441772
2: -61.7918816, 66.9551926, -153.9399567, 131.5551300, -193.3470154, 220.8951416
3: -31.1683788, 69.2946548, -64.4921265, 155.8913727, -187.0597534, 133.7867737
4: -69.3926926, 66.5902328, -171.7808990, 130.9613190, -200.3540039, 238.3711243

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4325078, upper bound: 187.3741556
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4325078, upper bound: 187.3741556
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -135.9420776, 125.6306152, -185.0837555, 202.5599365
1: -46.6114120, 62.3146553, -106.3213120, 118.7759705, -165.3873901, 168.6359558
2: -67.8582993, 69.8480682, -154.0769043, 131.6185913, -199.4768524, 223.9249725
3: -32.6405067, 74.5948639, -64.5234070, 156.0122223, -188.6527252, 139.1182556
4: -76.0634003, 69.6337814, -171.9312897, 131.0277863, -207.0911865, 241.5650635

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6257839
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6257839
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -93.3917847, 92.7331238, -182.1810760, 183.2736816
1: -69.8412399, 84.1715393, -72.9543228, 86.8698502, -156.7110748, 157.1258087
2: -101.2739258, 94.6007919, -105.7210693, 97.5862808, -198.8601990, 200.3218536
3: -45.8811989, 105.0105896, -47.2158279, 109.3487320, -155.2299347, 152.2264099
4: -113.0568390, 94.2403870, -118.0184097, 97.2374649, -210.2943115, 212.2587891

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -93.3917847, 92.7331238, -238.7796936, 228.1255493
1: -114.2232513, 127.2852173, -72.9543228, 86.8698502, -201.0930939, 200.2395020
2: -165.4884644, 141.3374176, -105.7210693, 97.5862808, -263.0747375, 247.0584869
3: -69.3965378, 166.8359833, -47.2158279, 109.3487320, -178.7452698, 214.0518188
4: -184.5755615, 140.5543518, -118.0184097, 97.2374649, -281.8130188, 258.5727539

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -103.9031677, 98.6311188, -149.6694489, 137.1584930, -241.0616455, 248.3005676
1: -81.4129181, 92.2082825, -117.0696259, 129.6089478, -211.0218353, 209.2778778
2: -117.8177338, 103.2506866, -169.5599670, 143.8451996, -261.6629333, 272.8106689
3: -49.3840714, 120.8862457, -70.5882416, 170.7739410, -220.1579895, 191.4744873
4: -131.3925323, 103.4162064, -189.1209259, 143.1206360, -274.5131836, 292.5371399

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4350329, upper bound: 187.4350329
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4350329, upper bound: 187.4350329
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -149.7934418, 137.2201385, -248.4728546, 252.6948700
1: -86.9877853, 96.1443176, -117.1645584, 129.6675110, -216.6553040, 213.3088684
2: -125.9503555, 107.5486832, -169.6975098, 143.9083099, -269.8586731, 277.2461853
3: -51.5428505, 128.2184296, -70.6193466, 170.8952637, -222.4380951, 198.8377686
4: -140.3546448, 107.9580917, -189.2721252, 143.1866150, -283.5412598, 297.2302246

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6289660
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -134.2789459, 122.3004990, -163.2814789, 189.0580597
1: -32.1188278, 51.3675804, -105.1659775, 115.5536118, -147.6724243, 156.5335541
2: -47.0364113, 57.8066292, -152.2682495, 128.1363678, -175.1727753, 210.0748749
3: -27.2729225, 55.0080185, -62.5633125, 154.0523529, -181.3252563, 117.5713272
4: -53.0424118, 57.1621284, -169.7571564, 127.5054779, -180.5478821, 226.9192810

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6268869
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256720, upper bound: 187.6259583
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -134.2789459, 122.3004990, -211.0508118, 229.3756561
1: -69.4899979, 90.1278381, -105.1659775, 115.5536118, -185.0435638, 195.2938080
2: -100.9360809, 99.7855835, -152.2682495, 128.1363678, -229.0724487, 252.0538330
3: -48.5418625, 106.7330246, -62.5633125, 154.0523529, -202.5941772, 169.2963409
4: -113.2336960, 98.6521301, -169.7571564, 127.5054779, -240.7391357, 268.4092712

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6268869
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256720, upper bound: 187.6259583
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -53.8375549, 63.7631683, -197.4850159, 171.8398743, -225.6773987, 261.2481384
1: -42.4029007, 59.6174088, -155.0117340, 162.7132111, -205.1161194, 214.6291504
2: -61.7918816, 66.9551926, -224.0996399, 179.1537933, -240.9456635, 291.0548401
3: -31.1683788, 69.2946548, -88.1924057, 222.5468140, -253.7151947, 156.4183960
4: -69.3926926, 66.5902328, -249.5538330, 179.0574036, -248.4501038, 316.1440430

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4871562, upper bound: 187.3784863
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4871562, upper bound: 187.3784863
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -197.6916809, 171.9809418, -231.4340820, 264.3095398
1: -46.6114120, 62.3146553, -155.1719666, 162.8464508, -209.4578552, 217.4866180
2: -67.8582993, 69.8480682, -224.3302612, 179.2990723, -247.1573334, 294.1783142
3: -32.6405067, 74.5948639, -88.2647858, 222.7579803, -255.3984680, 161.8992920
4: -76.0634003, 69.6337814, -249.8088379, 179.2068024, -255.2701874, 319.4425659

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4871579, upper bound: 187.6301360
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4871579, upper bound: 187.6301360
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -149.0803223, 134.5512085, -223.9991455, 238.9622192
1: -69.8412399, 84.1715393, -116.8296204, 126.9912109, -196.8324127, 201.0011597
2: -101.2739258, 94.6007919, -169.0030518, 141.0368652, -242.3107758, 263.6038513
3: -45.8811989, 105.0105896, -68.9912720, 169.9507294, -215.8319244, 174.0018616
4: -113.0568390, 94.2403870, -188.2618713, 140.3968811, -253.4537201, 282.5022583

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6281352
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -149.0803223, 134.5512085, -280.5977783, 283.8140869
1: -114.2232513, 127.2852173, -116.8296204, 126.9912109, -241.2144470, 244.1148376
2: -165.4884644, 141.3374176, -169.0030518, 141.0368652, -306.5253296, 310.3404541
3: -69.3965378, 166.8359833, -68.9912720, 169.9507294, -239.3472595, 235.8272552
4: -184.5755615, 140.5543518, -188.2618713, 140.3968811, -324.9723816, 328.8162231

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6281352
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -103.9031677, 98.6311188, -214.2766418, 185.6470490, -289.5502319, 312.9077148
1: -81.4129181, 92.2082825, -168.3418427, 175.6170502, -257.0299683, 260.5501099
2: -117.8177338, 103.2506866, -243.0501556, 193.6398315, -311.4575500, 346.3008423
3: -49.3840714, 120.8862457, -95.4888153, 240.5458832, -289.9299622, 215.4555206
4: -131.3925323, 103.4162064, -270.5207214, 193.6284180, -325.0209351, 373.9369202

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262787, upper bound: 187.4346769
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262788, upper bound: 187.4392577
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -214.4375305, 185.7405396, -296.9932251, 317.3389587
1: -86.9877853, 96.1443176, -168.4644012, 175.7048950, -262.6926575, 264.6087036
2: -125.9503555, 107.5486832, -243.2296295, 193.7352142, -319.6855774, 350.7781982
3: -51.5428505, 128.2184296, -95.5372314, 240.7064972, -292.2493591, 222.9164124
4: -140.3546448, 107.9580917, -270.7181702, 193.7283325, -334.0829468, 378.6762695

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255247, upper bound: 187.6287692
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6255247, upper bound: 187.6297939
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -124.0620880, 116.4748535, -62.4976425, 68.6312943, -192.6933441, 178.9724731
1: -97.3096466, 110.2341156, -49.0058823, 64.2134705, -161.5231171, 159.2399902
2: -140.8645020, 122.2203827, -71.2653809, 71.9282532, -212.7927246, 193.4857483
3: -59.5883522, 143.7969208, -33.6530342, 77.6752548, -137.2636108, 177.4499207
4: -157.2206116, 121.3138199, -79.8149414, 71.7916336, -229.0122070, 201.1287537

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -130.7532959, 120.0599976, -62.5989532, 68.6813965, -199.4346924, 182.6589203
1: -102.3961792, 113.4313202, -49.0822411, 64.2613220, -166.6575012, 162.5135651
2: -148.2715454, 125.8178711, -71.3749008, 71.9796066, -220.2511444, 197.1927795
3: -61.4442673, 150.3676147, -33.6786270, 77.7695847, -139.2138519, 184.0462341
4: -165.3551483, 125.1434479, -79.9356384, 71.8462219, -237.2013702, 205.0790863

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6256720
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6258460
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -115.7513199, 105.7508011, -194.3585205, 208.5616608
1: -69.5185242, 87.8645935, -90.5305862, 98.8567810, -168.3752747, 178.3951721
2: -100.9003143, 97.3328018, -131.0461578, 110.5165710, -211.4168854, 228.3789520
3: -47.0367928, 106.5991135, -52.9895172, 132.8755493, -179.9123383, 159.5886230
4: -113.0341187, 96.2283783, -145.9682770, 110.9683304, -224.0024414, 242.1966553

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5290925
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5290925
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -115.7513199, 105.7508011, -250.5600586, 247.4103546
1: -113.4549103, 124.2506790, -90.5305862, 98.8567810, -212.3116913, 214.7812653
2: -164.1941833, 138.0722961, -131.0461578, 110.5165710, -274.7107544, 269.1184387
3: -67.5839310, 165.3264618, -52.9895172, 132.8755493, -200.4594727, 218.3159790
4: -182.9160309, 137.3219604, -145.9682770, 110.9683304, -293.8843689, 283.2901917

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6274443
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6276226
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -185.7005920, 165.0380707, -61.8576355, 68.2385712, -253.9391632, 226.8957062
1: -145.8810577, 156.5421906, -48.5019646, 63.8458443, -209.7268982, 204.9551697
2: -210.9440613, 172.2925873, -70.5504990, 71.5194244, -282.4634399, 242.8430786
3: -84.7064209, 210.7245331, -33.4376526, 77.0382080, -160.5814056, 244.1621704
4: -235.0774231, 171.7262268, -79.0231171, 71.3719788, -306.4494019, 250.7493439

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3784863, upper bound: 187.4871570
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3784863, upper bound: 187.4871562
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -193.4654236, 169.3762360, -61.9585381, 68.2883911, -261.7538147, 231.3347321
1: -151.8135376, 160.4064026, -48.5780029, 63.8934555, -215.7070007, 208.9843750
2: -219.5485077, 176.6367798, -70.6596298, 71.5704880, -291.1189880, 247.2963409
3: -86.9621582, 218.4016571, -33.4631042, 77.1321182, -162.9123688, 251.8647614
4: -244.5529480, 176.4509125, -79.1433411, 71.4262695, -315.9792175, 255.5942535

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266088
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3784860, upper bound: 187.6266088
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9180756, 137.5461884, -114.8312531, 105.2043762, -249.1224518, 251.4270935
1: -112.8710327, 130.6539764, -89.8055115, 98.3340836, -211.2051086, 218.6286469
2: -163.7352295, 143.6445923, -130.0062866, 109.9501648, -273.6853943, 271.5076599
3: -70.0499954, 166.9139862, -52.7117538, 131.9280548, -199.9314117, 219.6257019
4: -182.9428101, 142.4112701, -144.8175507, 110.3987961, -293.3415527, 286.1289978

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3740132, upper bound: 187.6262788
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259468, upper bound: 187.6255247
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.0101318, 182.6937714, -114.8312531, 105.2043762, -315.2144470, 297.5250244
1: -164.9488068, 172.8107758, -89.8055115, 98.3340836, -263.2828674, 262.6162415
2: -238.2541351, 190.6086121, -130.0062866, 109.9501648, -348.2042847, 320.6148987
3: -94.0316849, 235.9465790, -52.7117538, 131.9280548, -225.0954437, 288.6582642
4: -265.1723938, 190.4963989, -144.8175507, 110.3987961, -375.5711670, 335.3139648

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3740132, upper bound: 187.6274399
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259468, upper bound: 187.6270481
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -134.2789459, 122.3004990, -210.9081726, 227.0893097
1: -69.5185242, 87.8645935, -105.1659775, 115.5536118, -185.0721130, 193.0305481
2: -100.9003143, 97.3328018, -152.2682495, 128.1363678, -229.0366821, 249.6010437
3: -47.0367928, 106.5991135, -62.5633125, 154.0523529, -201.0891266, 169.1624146
4: -113.0341187, 96.2283783, -169.7571564, 127.5054779, -240.5395966, 265.9855347

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4538069, upper bound: 187.5078077
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252000, upper bound: 187.5064804
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -149.0803223, 134.5512085, -279.3604736, 280.7393799
1: -113.4549103, 124.2506790, -116.8296204, 126.9912109, -240.4461060, 241.0802917
2: -164.1941833, 138.0722961, -169.0030518, 141.0368652, -305.2310486, 307.0753479
3: -67.5839310, 165.3264618, -68.9912720, 169.9507294, -237.5346680, 234.3177338
4: -182.9160309, 137.3219604, -188.2618713, 140.3968811, -323.3128967, 325.5838013

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5294421, upper bound: 187.6263038
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5294421, upper bound: 187.6297745
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -198.5921936, 172.9775543, -261.5852356, 291.4024963
1: -69.5185242, 87.8645935, -155.8728333, 163.7884216, -233.3069153, 243.7373962
2: -100.9003143, 97.3328018, -225.3496094, 180.3471985, -281.2474670, 322.6824036
3: -47.0367928, 106.5991135, -88.8018494, 223.7629547, -270.7997437, 194.2900085
4: -113.0341187, 96.2283783, -250.9470367, 180.2328949, -293.2670288, 347.1754150

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4343131, upper bound: 187.5075312
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259616, upper bound: 187.5063318
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -214.5238037, 185.8393707, -330.6486206, 346.1828308
1: -113.4549103, 124.2506790, -168.5326691, 175.7993317, -289.2542419, 292.7833557
2: -164.1941833, 138.0722961, -243.3277893, 193.8392792, -358.0334473, 381.4000854
3: -67.5839310, 165.3264618, -95.5921478, 240.8043213, -308.3881836, 260.1698914
4: -182.9160309, 137.3219604, -270.8276978, 193.8303833, -376.7463989, 408.1496277

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283487, upper bound: 187.6269352
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6283487, upper bound: 187.6293981
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -186.5197296, 165.9050598, -115.4761429, 107.8600616, -294.3797913, 281.3811951
1: -146.5168304, 157.3769226, -90.6525879, 101.9485703, -248.4653625, 248.0295105
2: -211.8697205, 173.2506714, -131.3562775, 112.6090240, -324.4787292, 304.6068726
3: -85.1997375, 211.6340637, -54.3431664, 134.9683075, -219.0444031, 265.9772034
4: -236.1106415, 172.6641083, -146.6197968, 112.1885986, -348.2991638, 319.2838745

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4009905, upper bound: 187.4915734
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4009905, upper bound: 187.4915734
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -194.2747040, 170.2841644, -115.5919189, 107.9167938, -302.1914978, 285.8760681
1: -152.4441833, 161.2508545, -90.7411270, 102.0026550, -254.4468384, 251.9919739
2: -220.4664917, 177.5887299, -131.4843903, 112.6674042, -333.1339111, 309.0731201
3: -87.4532776, 219.3068390, -54.3718262, 135.0808563, -221.3943329, 273.6786499
4: -245.5769043, 177.3847351, -146.7601929, 112.2500839, -357.8269958, 324.1449280

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -177.1518250, 150.2636261, -294.2210693, 314.2165222
1: -112.9016113, 130.6958466, -139.0804749, 141.4425354, -254.3440704, 268.2952271
2: -163.7802124, 143.6897125, -200.9367218, 156.5137939, -320.2939758, 342.9116821
3: -70.0736237, 166.9585724, -76.6745148, 199.4432068, -267.6372681, 243.6330719
4: -182.9930725, 142.4557343, -223.4640808, 157.1941223, -340.1871948, 365.4053955

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279994
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279994
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -177.1518250, 150.2636261, -360.5443420, 360.1593628
1: -165.1631775, 173.1105804, -139.0804749, 141.4425354, -306.6056213, 312.1910400
2: -238.5625763, 190.9392853, -200.9367218, 156.5137939, -395.0762939, 391.8760071
3: -94.2064362, 236.2539673, -76.6745148, 199.4432068, -292.9205017, 312.9284668
4: -265.5162659, 190.8201752, -223.4640808, 157.1941223, -422.7103882, 414.2842407

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6277689
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279000
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.45 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6268356
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6259194
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6268356
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.7953593, upper bound: 187.6259194
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4325078, upper bound: 187.3741556
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4325078, upper bound: 187.3741556
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6257839
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6257839
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6299725
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.7972518, upper bound: 187.6290361
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4350329, upper bound: 187.4350329
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4350329, upper bound: 187.4350329
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4325078, upper bound: 187.6289660
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6268869
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6256720, upper bound: 187.6259583
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6268869
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6256720, upper bound: 187.6259583
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4871562, upper bound: 187.3784863
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4871562, upper bound: 187.3784863
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4871579, upper bound: 187.6301360
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4871579, upper bound: 187.6301360
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6281352
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6274614
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5290925, upper bound: 187.6281352
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6262787, upper bound: 187.4346769
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6262788, upper bound: 187.4392577
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6255247, upper bound: 187.6287692
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6255247, upper bound: 187.6297939
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6268869, upper bound: 187.4533761
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6256720
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6259583, upper bound: 187.6258460
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5290925
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6253563, upper bound: 187.5290925
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6274443
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6253563, upper bound: 187.6276226
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3784863, upper bound: 187.4871570
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3784863, upper bound: 187.4871562
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266088
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3784860, upper bound: 187.6266088
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3740132, upper bound: 187.6262788
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6259468, upper bound: 187.6255247
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.3740132, upper bound: 187.6274399
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6259468, upper bound: 187.6270481
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4538069, upper bound: 187.5078077
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6252000, upper bound: 187.5064804
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5294421, upper bound: 187.6263038
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5294421, upper bound: 187.6297745
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4343131, upper bound: 187.5075312
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6259616, upper bound: 187.5063318
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6283487, upper bound: 187.6269352
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.6283487, upper bound: 187.6293981
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4009905, upper bound: 187.4915734
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4009905, upper bound: 187.4915734
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279994
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279994
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6277689
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.45
Output dim: 3, lower bound: -187.5306232, upper bound: 187.6279000

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.9018097, 54.7366104, -69.8997116, 75.5575409, -116.4593353, 124.6363220
1: -32.0589752, 51.3275223, -54.7346268, 70.8060379, -102.8650131, 106.0621414
2: -46.9490891, 57.7631111, -79.3983231, 79.7230682, -126.6721497, 137.1614227
3: -27.2523251, 54.9312325, -38.2211113, 84.8334351, -112.0857468, 93.1523438
4: -52.9476547, 57.1160774, -88.9131699, 79.2564697, -132.2041168, 146.0292511

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4862614, upper bound: 187.3896738
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4862614, upper bound: 187.8137650
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -76.5378723, 79.3108215, -120.2917938, 131.3169861
1: -32.1188278, 51.3675804, -59.7672310, 74.2600708, -106.3788757, 111.1348114
2: -47.0364113, 57.8066292, -86.7146301, 83.4780350, -130.5144348, 144.5212555
3: -27.2729225, 55.0080185, -40.0958786, 91.3843842, -118.6573029, 95.1038971
4: -53.0424118, 57.1621284, -96.9501190, 83.2198486, -136.2622223, 154.1122437

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8124576, upper bound: 187.3896738
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8124576, upper bound: 187.8137650
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.6466217, 95.0470734, -69.8997116, 75.5575409, -164.2041626, 164.9467773
1: -69.4114227, 90.0800629, -54.7346268, 70.8060379, -140.2174683, 144.8146973
2: -100.8234787, 99.7343903, -79.3983231, 79.7230682, -180.5465393, 179.1327209
3: -48.5166931, 106.6355896, -38.2211113, 84.8334351, -133.2857361, 144.8567047
4: -113.1097717, 98.5980530, -88.9131699, 79.2564697, -192.3662415, 187.5112305

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4860476, upper bound: 187.3740901
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6259194
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -76.5378723, 79.3108215, -168.0611572, 171.6345825
1: -69.4899979, 90.1278381, -59.7672310, 74.2600708, -143.7500610, 149.8950500
2: -100.9360809, 99.7855835, -86.7146301, 83.4780350, -184.4141235, 186.5002136
3: -48.5418625, 106.7330246, -40.0958786, 91.3843842, -139.9262238, 146.8288879
4: -113.2336960, 98.6521301, -96.9501190, 83.2198486, -196.4534912, 195.6022491

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7953597, upper bound: 187.3740901
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7953597, upper bound: 187.6259194
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -125.3855972, 119.5806885, -179.0338287, 192.0034485
1: -46.6114120, 62.3146553, -98.2052307, 113.1511230, -159.7625427, 160.5198822
2: -67.8582993, 69.8480682, -142.2991333, 125.4865494, -193.3448181, 212.1472015
3: -32.6405067, 74.5948639, -61.4639053, 145.3962860, -178.0367889, 136.0587769
4: -76.0634003, 69.6337814, -158.9818115, 124.6195374, -200.6829376, 228.6156006

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257837
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257837
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -132.5488739, 123.4380493, -182.8911896, 199.1667175
1: -46.6114120, 62.3146553, -103.6525116, 116.6961212, -163.3075256, 165.9671631
2: -67.8582993, 69.8480682, -150.2287750, 129.3436127, -197.2018890, 220.0768433
3: -32.6405067, 74.5948639, -63.4405289, 152.4523621, -185.0928650, 138.0353699
4: -76.0634003, 69.6337814, -167.6948547, 128.7310791, -204.7944794, 237.3286285

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257839
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257839
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.3242798, 89.8176117, -82.5392914, 86.3880310, -175.7122803, 172.3569031
1: -69.7470398, 84.1101379, -64.6071396, 80.9783936, -150.7254333, 148.7172699
2: -101.1371307, 94.5346985, -93.6184311, 91.1768112, -192.3139191, 188.1531372
3: -45.8487396, 104.8884125, -44.0479164, 98.4312515, -144.2799988, 148.9363251
4: -112.9061813, 94.1723709, -104.7108383, 90.5531921, -203.4593811, 198.8832092

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4897811, upper bound: 187.4897811
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4897811, upper bound: 187.8137650
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -90.0182343, 90.5549622, -180.0028839, 179.9001465
1: -69.8412399, 84.1715393, -70.2976913, 84.8072433, -154.6484680, 154.4691772
2: -101.2739258, 94.6007919, -101.8951874, 95.3282242, -196.6021423, 196.4959717
3: -45.8811989, 105.0105896, -46.1052017, 105.8526535, -151.7338562, 151.1157837
4: -113.0568390, 94.2403870, -113.8105698, 94.9274139, -207.9842224, 208.0509644

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8137650, upper bound: 187.4897811
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8137650, upper bound: 187.8137650
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -145.9277954, 134.6750488, -82.5392914, 86.3880310, -232.3157959, 217.2143402
1: -114.1324387, 127.2294235, -64.6071396, 80.9783936, -195.1108398, 191.8365479
2: -165.3568573, 141.2772369, -93.6184311, 91.1768112, -256.5336609, 234.8956604
3: -69.3667374, 166.7200623, -44.0479164, 98.4312515, -167.7979889, 210.7679749
4: -184.4309692, 140.4914246, -104.7108383, 90.5531921, -274.9841614, 245.2022247

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4896723, upper bound: 187.4350361
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6290361
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -90.0182343, 90.5549622, -236.6015167, 224.7519989
1: -114.2232513, 127.2852173, -70.2976913, 84.8072433, -199.0304871, 197.5828857
2: -165.4884644, 141.3374176, -101.8951874, 95.3282242, -260.8166809, 243.2325745
3: -69.3965378, 166.8359833, -46.1052017, 105.8526535, -175.2491913, 212.9411926
4: -184.5755615, 140.5543518, -113.8105698, 94.9274139, -279.5029602, 254.3648682

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7972522, upper bound: 187.4350361
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7972522, upper bound: 187.6290361
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -139.0402222, 131.1075745, -242.3602905, 241.9416504
1: -86.9877853, 96.1443176, -108.8877182, 123.9752121, -210.9629974, 205.0320435
2: -125.9503555, 107.5486832, -157.6835480, 137.7197266, -263.6700745, 265.2322388
3: -51.5428505, 128.2184296, -67.5197983, 160.0581207, -211.6009521, 195.7382202
4: -140.3546448, 107.9580917, -176.0632782, 136.6995392, -277.0541992, 284.0213623

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289659
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289659
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -146.3229065, 134.9822693, -246.2349854, 249.2243347
1: -86.9877853, 96.1443176, -114.4310303, 127.5312195, -214.5190125, 210.5753479
2: -125.9503555, 107.5486832, -165.7576294, 141.5741119, -267.5244751, 273.3063049
3: -51.5428505, 128.2184296, -69.4877243, 167.2463074, -218.7891388, 197.7061462
4: -140.3546448, 107.9580917, -184.9295807, 140.8143768, -281.1689758, 292.8876648

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -40.9018097, 54.7366104, -124.0620880, 116.4748535, -157.3766632, 178.7986603
1: -32.0589752, 51.3275223, -97.3096466, 110.2341156, -142.2930908, 148.6371613
2: -46.9490891, 57.7631111, -140.8645020, 122.2203827, -169.1694641, 198.6276093
3: -27.2523251, 54.9312325, -59.5883522, 143.7969208, -171.0492249, 114.5195847
4: -52.9476547, 57.1160774, -157.2206116, 121.3138199, -174.2614594, 214.3366852

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4537857, upper bound: 187.3895300
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4537857, upper bound: 187.7973624
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -40.9809875, 54.7791138, -130.7532959, 120.0599976, -161.0409851, 185.5324097
1: -32.1188278, 51.3675804, -102.3961792, 113.4313202, -145.5501251, 153.7637634
2: -47.0364113, 57.8066292, -148.2715454, 125.8178711, -172.8542786, 206.0781708
3: -27.2729225, 55.0080185, -61.4442673, 150.3676147, -177.6405334, 116.4522858
4: -53.0424118, 57.1621284, -165.3551483, 125.1434479, -178.1858521, 222.5172729

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256721, upper bound: 187.3895300
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256721, upper bound: 187.7973624
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -88.6466217, 95.0470734, -124.0620880, 116.4748535, -205.1214294, 219.1091461
1: -69.4114227, 90.0800629, -97.3096466, 110.2341156, -179.6455383, 187.3897095
2: -100.8234787, 99.7343903, -140.8645020, 122.2203827, -223.0438538, 240.5988922
3: -48.5166931, 106.6355896, -59.5883522, 143.7969208, -192.3136139, 166.2239380
4: -113.1097717, 98.5980530, -157.2206116, 121.3138199, -234.4235840, 255.8185883

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4533761, upper bound: 187.3739922
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6259583
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -88.7503510, 95.0967102, -130.7532959, 120.0599976, -208.8103485, 225.8499908
1: -69.4899979, 90.1278381, -102.3961792, 113.4313202, -182.9212646, 192.5240021
2: -100.9360809, 99.7855835, -148.2715454, 125.8178711, -226.7539520, 248.0571289
3: -48.5418625, 106.7330246, -61.4442673, 150.3676147, -198.9094238, 168.1772919
4: -113.2336960, 98.6521301, -165.3551483, 125.1434479, -238.3771210, 264.0072632

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258461, upper bound: 187.3739922
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258461, upper bound: 187.6259583
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -185.5335846, 164.8775177, -224.3306580, 252.1514282
1: -46.6114120, 62.3146553, -145.7511902, 156.3802032, -202.9272614, 208.0658417
2: -67.8582993, 69.8480682, -210.7563934, 172.1129303, -239.9711914, 280.6044312
3: -32.6405067, 74.5948639, -84.6182480, 210.5406342, -243.1811218, 158.0781708
4: -76.0634003, 69.6337814, -234.8673401, 171.5497894, -247.6131592, 304.5010986

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6301358
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6259582
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -59.4531441, 66.6178513, -193.4216919, 169.3316650, -228.7847900, 260.0394897
1: -46.6114120, 62.3146553, -151.7793427, 160.3641357, -206.9755402, 214.0939941
2: -67.8582993, 69.8480682, -219.4990845, 176.5899963, -244.4482880, 289.3471680
3: -32.6405067, 74.5948639, -86.9392014, 218.3532867, -250.9937897, 160.3694763
4: -76.0634003, 69.6337814, -244.4977722, 176.4049377, -252.4683380, 314.1315613

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6301360
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4532943, upper bound: 187.6259583
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -88.6077271, 92.8103714, -182.2582855, 178.4896240
1: -69.8412399, 84.1715393, -69.5185242, 87.8645935, -157.7057953, 153.6900330
2: -101.2739258, 94.6007919, -100.9003143, 97.3328018, -198.6067200, 195.5010986
3: -45.8811989, 105.0105896, -47.0367928, 106.5991135, -152.4803009, 152.0473785
4: -113.0568390, 94.2403870, -113.0341187, 96.2283783, -209.2852020, 207.2745056

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5078077, upper bound: 187.4862798
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5064803, upper bound: 187.7947934
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -144.8092651, 131.6590424, -221.1069946, 234.6911469
1: -69.8412399, 84.1715393, -113.4549103, 124.2506790, -194.0919037, 197.6264496
2: -101.2739258, 94.6007919, -164.1941833, 138.0722961, -239.3461914, 258.7949829
3: -45.8811989, 105.0105896, -67.5839310, 165.3264618, -211.2076569, 172.5945129
4: -113.0568390, 94.2403870, -182.9160309, 137.3219604, -250.3787994, 277.1564331

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5078077, upper bound: 187.4895571
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5064804, upper bound: 187.7947934
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -88.6077271, 92.8103714, -238.8569336, 223.3414917
1: -114.2232513, 127.2852173, -69.5185242, 87.8645935, -202.0878448, 196.8037415
2: -165.4884644, 141.3374176, -100.9003143, 97.3328018, -262.8212585, 242.2377319
3: -69.3965378, 166.8359833, -47.0367928, 106.5991135, -175.8508911, 213.8727570
4: -184.5755615, 140.5543518, -113.0341187, 96.2283783, -280.8039246, 253.5884705

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6263293
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -144.8092651, 131.6590424, -277.7055969, 279.5430298
1: -114.2232513, 127.2852173, -113.4549103, 124.2506790, -238.4739380, 240.7401276
2: -165.4884644, 141.3374176, -164.1941833, 138.0722961, -303.5607605, 305.5316162
3: -69.3965378, 166.8359833, -67.5839310, 165.3264618, -234.7229919, 234.4199219
4: -184.5755615, 140.5543518, -182.9160309, 137.3219604, -321.8974609, 323.4703979

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4347185
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6269313
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -103.9031677, 98.6311188, -143.7563324, 137.4443665, -240.1795044, 242.3874512
1: -81.4129181, 92.2082825, -112.7467041, 130.5567780, -209.9925995, 204.9549408
2: -117.8177338, 103.2506866, -163.5547180, 143.5398254, -259.0214844, 266.8053894
3: -49.3840714, 120.8862457, -69.9975510, 166.7504578, -216.1344910, 188.7614746
4: -131.3925323, 103.4162064, -182.7438660, 142.3035278, -272.3574219, 286.1600647

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -103.9031677, 98.6311188, -209.7880859, 182.5311127, -286.4342651, 308.4191895
1: -81.4129181, 92.2082825, -164.7771606, 172.6552429, -254.0681610, 256.9853821
2: -117.8177338, 103.2506866, -238.0054321, 190.4399872, -308.2577209, 341.2561035
3: -49.3840714, 120.8862457, -93.9459839, 235.7164307, -285.1004639, 213.8959808
4: -131.3925323, 103.4162064, -264.8973694, 190.3248901, -321.7174072, 368.3135681

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4392577
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4347185
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -143.8914948, 137.5167542, -247.8019562, 246.7928925
1: -86.9877853, 96.1443176, -112.8503876, 130.6257629, -215.7660675, 208.9947052
2: -125.9503555, 107.5486832, -163.7048645, 143.6141357, -267.4067078, 271.2535095
3: -51.5428505, 128.2184296, -70.0340729, 166.8839417, -218.4267731, 196.2104340
4: -140.3546448, 107.9580917, -182.9088287, 142.3812714, -281.6063538, 290.8669128

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6287692
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6263293
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -111.2527161, 102.9014282, -209.9721832, 182.6498566, -293.9025574, 312.8735657
1: -86.9877853, 96.1443176, -164.9187622, 172.7687683, -259.7565308, 261.0630798
2: -125.9503555, 107.5486832, -238.2108917, 190.5623016, -316.5126648, 345.7595215
3: -51.5428505, 128.2184296, -94.0072021, 235.9033966, -287.4462585, 221.3673401
4: -140.3546448, 107.9580917, -265.1242371, 190.4510345, -330.8056641, 373.0823364

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6297895
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6269313
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -124.0620880, 116.4748535, -40.9018097, 54.7366104, -178.7986450, 157.3766632
1: -97.3096466, 110.2341156, -32.0589752, 51.3275223, -148.6371613, 142.2930908
2: -140.8645020, 122.2203827, -46.9490891, 57.7631111, -198.6275940, 169.1694641
3: -59.5883522, 143.7969208, -27.2523251, 54.9312325, -114.5195847, 171.0492249
4: -157.2206116, 121.3138199, -52.9476547, 57.1160774, -214.3366852, 174.2614441

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252281, upper bound: 187.4531642
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260387, upper bound: 187.4530820
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -124.0620880, 116.4748535, -88.6466217, 95.0470734, -219.1091461, 205.1214600
1: -97.3096466, 110.2341156, -69.4114227, 90.0800629, -187.3897095, 179.6455383
2: -140.8645020, 122.2203827, -100.8234787, 99.7343903, -240.5988922, 223.0438538
3: -59.5883522, 143.7969208, -48.5166931, 106.6355896, -166.2239380, 192.3136139
4: -157.2206116, 121.3138199, -113.1097717, 98.5980530, -255.8186646, 234.4235840

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252281, upper bound: 187.4531642
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6260387, upper bound: 187.4530820
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -130.7532959, 120.0599976, -40.9809875, 54.7791138, -185.5324097, 161.0409851
1: -102.3961792, 113.4313202, -32.1188278, 51.3675804, -153.7637634, 145.5501251
2: -148.2715454, 125.8178711, -47.0364113, 57.8066292, -206.0781708, 172.8542786
3: -61.4442673, 150.3676147, -27.2729225, 55.0080185, -116.4522858, 177.6405334
4: -165.3551483, 125.1434479, -53.0424118, 57.1621284, -222.5172729, 178.1858521

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244703, upper bound: 187.5878173
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250591, upper bound: 187.6255823
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -130.7532959, 120.0599976, -88.7503510, 95.0967102, -225.8499908, 208.8103485
1: -102.3961792, 113.4313202, -69.4899979, 90.1278381, -192.5239868, 182.9212799
2: -148.2715454, 125.8178711, -100.9360809, 99.7855835, -248.0571289, 226.7539520
3: -61.4442673, 150.3676147, -48.5418625, 106.7330246, -168.1772919, 198.9094543
4: -165.3551483, 125.1434479, -113.2336960, 98.6521301, -264.0072632, 238.3771210

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244703, upper bound: 187.5878478
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250591, upper bound: 187.6257636
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -89.4479752, 89.8819046, -178.4896240, 182.2583160
1: -69.5185242, 87.8645935, -69.8412399, 84.1715393, -153.6900330, 157.7058105
2: -100.9003143, 97.3328018, -101.2739258, 94.6007919, -195.5010986, 198.6067047
3: -47.0367928, 106.5991135, -45.8811989, 105.0105896, -152.0473785, 152.4803009
4: -113.0341187, 96.2283783, -113.0568390, 94.2403870, -207.2745056, 209.2851868

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264685, upper bound: 187.3659618
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263294, upper bound: 187.5063314
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -146.0465698, 134.7337646, -223.3414917, 238.8569336
1: -69.5185242, 87.8645935, -114.2232513, 127.2852173, -196.8037262, 202.0878296
2: -100.9003143, 97.3328018, -165.4884644, 141.3374176, -242.2377319, 262.8212585
3: -47.0367928, 106.5991135, -69.3965378, 166.8359833, -213.8727417, 175.8508759
4: -113.0341187, 96.2283783, -184.5755615, 140.5543518, -253.5884705, 280.8038940

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6264685, upper bound: 187.3659618
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263294, upper bound: 187.5063314
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -89.4479752, 89.8819046, -234.6911621, 221.1069946
1: -113.4549103, 124.2506790, -69.8412399, 84.1715393, -197.6264496, 194.0919189
2: -164.1941833, 138.0722961, -101.2739258, 94.6007919, -258.7949829, 239.3461914
3: -67.5839310, 165.3264618, -45.8811989, 105.0105896, -172.5945129, 211.2076569
4: -182.9160309, 137.3219604, -113.0568390, 94.2403870, -277.1564331, 250.3787842

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300111, upper bound: 187.4672258
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6261783
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -146.0465698, 134.7337646, -279.5430298, 277.7056274
1: -113.4549103, 124.2506790, -114.2232513, 127.2852173, -240.7401276, 238.4739380
2: -164.1941833, 138.0722961, -165.4884644, 141.3374176, -305.5316162, 303.5607605
3: -67.5839310, 165.3264618, -69.3965378, 166.8359833, -234.4199219, 234.7229919
4: -182.9160309, 137.3219604, -184.5755615, 140.5543518, -323.4703979, 321.8974609

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300111, upper bound: 187.4672258
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6263517
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -193.3603668, 169.2692261, -53.8375549, 63.7631683, -257.1235046, 223.1067810
1: -151.7314453, 160.3050079, -42.4029007, 59.6174088, -211.3488312, 202.5927582
2: -219.4299011, 176.5243683, -61.7918816, 66.9551926, -286.3851013, 238.1765594
3: -86.9069595, 218.2855377, -31.1683788, 69.2946548, -154.9298859, 249.4538727
4: -244.4205475, 176.3404083, -69.3926926, 66.5902328, -311.0107727, 245.7330933

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3727329, upper bound: 187.6086995
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266087
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -193.4216919, 169.3316650, -59.4531441, 66.6178513, -260.0394897, 228.7848053
1: -151.7793427, 160.3641357, -46.6114120, 62.3146553, -214.0939789, 206.9755096
2: -219.4990845, 176.5899963, -67.8582993, 69.8480682, -289.3471680, 244.4482880
3: -86.9392014, 218.3532867, -32.6405067, 74.5948639, -160.3694916, 250.9937897
4: -244.4977722, 176.4049377, -76.0634003, 69.6337814, -314.1315002, 252.4683380

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3727329, upper bound: 187.6086996
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266088
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.7563324, 137.4443665, -103.9031677, 98.6311188, -242.3874512, 240.1795349
1: -112.7467041, 130.5567780, -81.4129181, 92.2082825, -204.9549561, 209.9926147
2: -163.5547180, 143.5398254, -117.8177338, 103.2506866, -266.8053894, 259.0215149
3: -69.9975510, 166.7504578, -49.3840714, 120.8862457, -188.7614899, 216.1345062
4: -182.7438660, 142.3035278, -131.3925323, 103.4162064, -286.1600647, 272.3574524

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4346822, upper bound: 187.3965469
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346822, upper bound: 187.6255247
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.8914948, 137.5167542, -111.2527161, 102.9014282, -246.7929230, 247.8019714
1: -112.8503876, 130.6257629, -86.9877853, 96.1443176, -208.9947052, 215.7660522
2: -163.7048645, 143.6141357, -125.9503555, 107.5486832, -271.2535400, 267.4067078
3: -70.0340729, 166.8839417, -51.5428505, 128.2184296, -196.2104340, 218.4267731
4: -182.9088287, 142.3812714, -140.3546448, 107.9580917, -290.8669128, 281.6063538

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292772, upper bound: 187.3965467
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6292772, upper bound: 187.6255247
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -209.7880859, 182.5311127, -103.9031677, 98.6311188, -308.4191895, 286.4342651
1: -164.7771606, 172.6552429, -81.4129181, 92.2082825, -256.9853821, 254.0681610
2: -238.0054321, 190.4399872, -117.8177338, 103.2506866, -341.2560730, 308.2577209
3: -93.9459839, 235.7164307, -49.3840714, 120.8862457, -213.8959808, 285.1004639
4: -264.8973694, 190.3248901, -131.3925323, 103.4162064, -368.3135681, 321.7174072

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4398745, upper bound: 187.4965127
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6270481
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -209.9721832, 182.6498566, -111.2527161, 102.9014282, -312.8735962, 293.9025269
1: -164.9187622, 172.7687683, -86.9877853, 96.1443176, -261.0630798, 259.7565308
2: -238.2108917, 190.5623016, -125.9503555, 107.5486832, -345.7595215, 316.5126343
3: -94.0072021, 235.9033966, -51.5428505, 128.2184296, -221.3673401, 287.4462585
4: -265.1242371, 190.4510345, -140.3546448, 107.9580917, -373.0823364, 330.8056641

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6335490, upper bound: 187.4965127
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6335490, upper bound: 187.6270481
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -130.7532959, 120.0599976, -208.6677094, 223.5636444
1: -69.5185242, 87.8645935, -102.3961792, 113.4313202, -182.9498291, 190.2607422
2: -100.9003143, 97.3328018, -148.2715454, 125.8178711, -226.7181854, 245.6043243
3: -47.0367928, 106.5991135, -61.4442673, 150.3676147, -197.4043884, 168.0433807
4: -113.0341187, 96.2283783, -165.3551483, 125.1434479, -238.1775665, 261.5835266

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252001, upper bound: 187.3660477
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6252001, upper bound: 187.5064804
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -88.6077271, 92.8103714, -237.6196289, 220.2667694
1: -113.4549103, 124.2506790, -69.5185242, 87.8645935, -201.3195038, 193.7691956
2: -164.1941833, 138.0722961, -100.9003143, 97.3328018, -261.5269775, 238.9726105
3: -67.5839310, 165.3264618, -47.0367928, 106.5991135, -174.1830444, 212.3632507
4: -182.9160309, 137.3219604, -113.0341187, 96.2283783, -279.1444092, 250.3560791

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5078807, upper bound: 187.4538067
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5065233, upper bound: 187.6252000
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -144.8092651, 131.6590424, -276.4683228, 276.4683228
1: -113.4549103, 124.2506790, -113.4549103, 124.2506790, -237.7055969, 237.7055969
2: -164.1941833, 138.0722961, -164.1941833, 138.0722961, -302.2664795, 302.2664795
3: -67.5839310, 165.3264618, -67.5839310, 165.3264618, -232.9104004, 232.9104004
4: -182.9160309, 137.3219604, -182.9160309, 137.3219604, -320.2379456, 320.2379150

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5078807, upper bound: 187.4677463
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5065234, upper bound: 187.6259112
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -88.6077271, 92.8103714, -194.2747040, 170.2841644, -258.8919067, 287.0850830
1: -69.5185242, 87.8645935, -152.4441833, 161.2508545, -230.7411957, 240.3087616
2: -100.9003143, 97.3328018, -220.4664917, 177.5887299, -278.4454041, 317.7992859
3: -47.0367928, 106.5991135, -87.4532776, 219.3068390, -266.3436279, 192.7422943
4: -113.0341187, 96.2283783, -245.5769043, 177.3847351, -290.4188538, 341.8052979

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259617, upper bound: 187.3659618
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259617, upper bound: 187.5063318
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -143.9574738, 137.5898285, -281.7225647, 275.6165161
1: -113.4549103, 124.2506790, -112.9016113, 130.6958466, -242.5797119, 237.1522827
2: -164.1941833, 138.0722961, -163.7802124, 143.6897125, -306.0038452, 301.8525085
3: -67.5839310, 165.3264618, -70.0736237, 166.9585724, -234.5424957, 233.4506531
4: -182.9160309, 137.3219604, -182.9930725, 142.4557343, -324.6633911, 320.3149719

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266280, upper bound: 187.4540569
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256993, upper bound: 187.6257329
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -144.8092651, 131.6590424, -210.2807465, 183.0075378, -327.8168030, 341.9397888
1: -113.4549103, 124.2506790, -165.1631775, 173.1105804, -286.5654907, 289.4138489
2: -164.1941833, 138.0722961, -238.5625763, 190.9392853, -355.1334839, 376.6348877
3: -67.5839310, 165.3264618, -94.2064362, 236.2539673, -303.8377991, 258.7338257
4: -182.9160309, 137.3219604, -265.5162659, 190.8201752, -373.7362061, 402.8381958

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266280, upper bound: 187.4680263
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256994, upper bound: 187.6263517
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -194.2747040, 170.2841644, -105.6108093, 102.6108704, -296.8855591, 275.8949585
1: -152.4441833, 161.2508545, -83.0685120, 96.9844360, -249.4285583, 244.3193665
2: -220.4664917, 177.5887299, -120.3844452, 107.2560120, -327.7225037, 297.9731750
3: -87.4532776, 219.3068390, -51.7198486, 125.1831512, -211.3801422, 271.0266724
4: -245.5769043, 177.3847351, -134.5889587, 106.6198120, -352.1967163, 311.9736938

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3926270, upper bound: 187.6091843
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266087
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -194.2747040, 170.2841644, -112.2545319, 105.8760834, -300.1507874, 282.5386963
1: -152.4441833, 161.2508545, -88.1243439, 100.0631561, -252.5073242, 249.3751984
2: -220.4664917, 177.5887299, -127.7198334, 110.5493774, -331.0158386, 305.3085327
3: -87.4532776, 219.3068390, -53.3567505, 131.6137695, -217.9225006, 272.6635742
4: -245.5769043, 177.3847351, -142.6180115, 110.0738602, -355.6507568, 320.0027466

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3926270, upper bound: 187.6091843
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -144.8092651, 131.6590424, -275.6165161, 281.7225342
1: -112.9016113, 130.6958466, -113.4549103, 124.2506790, -237.1522827, 242.5797119
2: -163.7802124, 143.6897125, -164.1941833, 138.0722961, -301.8525085, 306.0038147
3: -70.0736237, 166.9585724, -67.5839310, 165.3264618, -233.4506531, 234.5424957
4: -182.9930725, 142.4557343, -182.9160309, 137.3219604, -320.3150024, 324.6633911

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282394, upper bound: 187.3967780
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281344, upper bound: 187.6255246
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -210.2807465, 183.0075378, -326.9650269, 347.5987854
1: -112.9016113, 130.6958466, -165.1631775, 173.1105804, -285.3846741, 294.5191040
2: -163.7802124, 143.6897125, -238.5625763, 190.9392853, -353.9878540, 380.6860962
3: -70.0736237, 166.9585724, -94.2064362, 236.2539673, -304.4873352, 259.4671631
4: -182.9930725, 142.4557343, -265.5162659, 190.8201752, -373.8132324, 407.7260132

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6282394, upper bound: 187.3999444
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281344, upper bound: 187.6255245
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -144.8092651, 131.6590424, -341.9397888, 327.8168030
1: -165.1631775, 173.1105804, -113.4549103, 124.2506790, -289.4138489, 286.5654907
2: -238.5625763, 190.9392853, -164.1941833, 138.0722961, -376.6348877, 355.1334839
3: -94.2064362, 236.2539673, -67.5839310, 165.3264618, -258.7338867, 303.8378296
4: -265.5162659, 190.8201752, -182.9160309, 137.3219604, -402.8381958, 373.7362061

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6353343, upper bound: 187.4966750
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324061, upper bound: 187.6265713
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -210.2807465, 183.0075378, -393.2882690, 393.2882690
1: -165.1631775, 173.1105804, -165.1631775, 173.1105804, -338.2737427, 338.2737427
2: -238.5625763, 190.9392853, -238.5625763, 190.9392853, -429.5018616, 429.5018311
3: -94.2064362, 236.2539673, -94.2064362, 236.2539673, -329.7705078, 329.7705383
4: -265.5162659, 190.8201752, -265.5162659, 190.8201752, -456.3364258, 456.3364258

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6353343, upper bound: 187.4966770
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324061, upper bound: 187.6266687
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.70 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4862614, upper bound: 187.3896738
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4862614, upper bound: 187.8137650
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.8124576, upper bound: 187.3896738
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.8124576, upper bound: 187.8137650
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4860476, upper bound: 187.3740901
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4860476, upper bound: 187.6259194
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.7953597, upper bound: 187.3740901
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.7953597, upper bound: 187.6259194
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257837
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257837
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257839
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4324371, upper bound: 187.6257839
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4897811, upper bound: 187.4897811
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4897811, upper bound: 187.8137650
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.8137650, upper bound: 187.4897811
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.8137650, upper bound: 187.8137650
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4896723, upper bound: 187.4350361
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4896723, upper bound: 187.6290361
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.7972522, upper bound: 187.4350361
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.7972522, upper bound: 187.6290361
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289659
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289659
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4350329, upper bound: 187.6289660
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4537857, upper bound: 187.3895300
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4537857, upper bound: 187.7973624
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6256721, upper bound: 187.3895300
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6256721, upper bound: 187.7973624
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4533761, upper bound: 187.3739922
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4533761, upper bound: 187.6259583
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6258461, upper bound: 187.3739922
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6258461, upper bound: 187.6259583
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6301358
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6259582
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4532944, upper bound: 187.6301360
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4532943, upper bound: 187.6259583
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5078077, upper bound: 187.4862798
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5064803, upper bound: 187.7947934
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5078077, upper bound: 187.4895571
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5064804, upper bound: 187.7947934
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6263293
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4347185
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6269313
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075311, upper bound: 187.4317035
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4392577
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5075312, upper bound: 187.4347185
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6287692
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6263293
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6297895
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5063318, upper bound: 187.6269313
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6252281, upper bound: 187.4531642
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6260387, upper bound: 187.4530820
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6252281, upper bound: 187.4531642
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6260387, upper bound: 187.4530820
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6244703, upper bound: 187.5878173
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6250591, upper bound: 187.6255823
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6244703, upper bound: 187.5878478
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6250591, upper bound: 187.6257636
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6264685, upper bound: 187.3659618
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6263294, upper bound: 187.5063314
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6264685, upper bound: 187.3659618
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6263294, upper bound: 187.5063314
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6300111, upper bound: 187.4672258
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6261783
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6300111, upper bound: 187.4672258
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6290607, upper bound: 187.6263517
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3727329, upper bound: 187.6086995
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266087
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3727329, upper bound: 187.6086996
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3784863, upper bound: 187.6266088
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4346822, upper bound: 187.3965469
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4346822, upper bound: 187.6255247
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6292772, upper bound: 187.3965467
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6292772, upper bound: 187.6255247
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4398745, upper bound: 187.4965127
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4398745, upper bound: 187.6270481
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6335490, upper bound: 187.4965127
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6335490, upper bound: 187.6270481
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6252001, upper bound: 187.3660477
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6252001, upper bound: 187.5064804
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5078807, upper bound: 187.4538067
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5065233, upper bound: 187.6252000
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5078807, upper bound: 187.4677463
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.5065234, upper bound: 187.6259112
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6259617, upper bound: 187.3659618
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6259617, upper bound: 187.5063318
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6266280, upper bound: 187.4540569
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6256993, upper bound: 187.6257329
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6266280, upper bound: 187.4680263
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6256994, upper bound: 187.6263517
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3926270, upper bound: 187.6091843
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266087
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.3926270, upper bound: 187.6091843
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.4009905, upper bound: 187.6266088
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6282394, upper bound: 187.3967780
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6281344, upper bound: 187.6255246
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6282394, upper bound: 187.3999444
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6281344, upper bound: 187.6255245
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6353343, upper bound: 187.4966750
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6324061, upper bound: 187.6265713
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6353343, upper bound: 187.4966770
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 3, lower bound: -187.6324061, upper bound: 187.6266687

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -39.0077286, 53.3303146, -69.8997116, 75.5575409, -114.5652618, 123.2300262
1: -30.5722923, 50.0091782, -54.7346268, 70.8060379, -101.3783264, 104.7437668
2: -44.8006439, 56.3047638, -79.3983231, 79.7230682, -124.5237122, 135.7030945
3: -26.5797043, 52.9133911, -38.2211113, 84.8334351, -111.4131317, 91.1345062
4: -50.5919037, 55.6129456, -88.9131699, 79.2564697, -129.8483734, 144.5261230

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860403, upper bound: 187.7409287
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860403, upper bound: 187.8136488
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -34.2662277, 50.8938408, -76.5378723, 79.3108215, -113.5770493, 127.4316940
1: -27.0025425, 47.7687073, -59.7672310, 74.2600708, -101.2626038, 107.5359344
2: -39.5765114, 53.9266510, -86.7146301, 83.4780350, -123.0545349, 140.6412811
3: -25.4885406, 48.3789978, -40.0958786, 91.3843842, -116.8728943, 88.4748764
4: -44.9341927, 53.1512604, -96.9501190, 83.2198486, -128.1540222, 150.1013794

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4860403, upper bound: 187.3685378
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4860403, upper bound: 187.3888774
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -39.0077286, 53.3303146, -76.5378723, 79.3108215, -118.3185425, 129.8681946
1: -30.5722923, 50.0091782, -59.7672310, 74.2600708, -104.8323669, 109.7763977
2: -44.8006439, 56.3047638, -86.7146301, 83.4780350, -128.2786865, 143.0193939
3: -26.5797043, 52.9133911, -40.0958786, 91.3843842, -117.9640808, 93.0092697
4: -50.5919037, 55.6129456, -96.9501190, 83.2198486, -133.8117371, 152.5630646

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4860403, upper bound: 187.7399719
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4860403, upper bound: 187.3888774
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -86.0194931, 93.4221649, -69.8997116, 75.5575409, -161.5770111, 163.3218689
1: -67.3570480, 88.5131531, -54.7346268, 70.8060379, -138.1630859, 143.2477722
2: -97.8890457, 98.0217743, -79.3983231, 79.7230682, -177.6121216, 177.4200745
3: -47.6930962, 103.9667816, -38.2211113, 84.8334351, -132.2406311, 142.1878662
4: -109.8827209, 96.8530350, -88.9131699, 79.2564697, -189.1391907, 185.7662048

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4858462, upper bound: 187.6249973
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4858462, upper bound: 187.6259857
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -79.6487732, 90.3730392, -76.5378723, 79.3108215, -158.9595947, 166.9109192
1: -62.5632401, 85.6208954, -59.7672310, 74.2600708, -136.8233032, 145.3881226
2: -90.9939499, 94.9724960, -86.7146301, 83.4780350, -174.4719543, 181.6871338
3: -46.1946678, 97.9734116, -40.0958786, 91.3843842, -137.3304443, 138.0692749
4: -102.2944336, 93.6362610, -96.9501190, 83.2198486, -185.5142822, 190.5863800

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4858462, upper bound: 187.3673777
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4858462, upper bound: 187.3732252
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -86.0194931, 93.4221649, -76.5378723, 79.3108215, -165.3302765, 169.9600372
1: -67.3570480, 88.5131531, -59.7672310, 74.2600708, -141.6171265, 148.2803802
2: -97.8890457, 98.0217743, -86.7146301, 83.4780350, -181.3670807, 184.7364044
3: -47.6930962, 103.9667816, -40.0958786, 91.3843842, -138.9037323, 144.0626221
4: -109.8827209, 96.8530350, -96.9501190, 83.2198486, -193.1025696, 193.8031616

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4858462, upper bound: 187.3673777
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4858462, upper bound: 187.6250198
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -39.0077286, 53.3303146, -125.3074646, 119.4819336, -158.4896393, 178.6377563
1: -30.5722923, 50.0091782, -98.1450348, 113.0652542, -143.6375427, 148.1542053
2: -44.8006439, 56.3047638, -142.2108917, 125.3818207, -170.1824646, 198.5156555
3: -26.5797043, 52.9133911, -61.4081421, 145.3101044, -171.8898010, 114.3215332
4: -50.5919037, 55.6129456, -158.8825073, 124.5176773, -175.1095886, 214.4954529

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3718044, upper bound: 187.6242571
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3718044, upper bound: 187.6265378
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -84.4384613, 92.6079865, -125.3855972, 119.5806885, -204.0191498, 217.9935760
1: -66.1660461, 87.7723999, -98.2052307, 113.1511230, -179.3171692, 185.9776154
2: -96.2040405, 97.2211151, -142.2991333, 125.4865494, -221.6905670, 239.5202332
3: -47.2600517, 102.6184387, -61.4639053, 145.3962860, -192.5695953, 164.0823364
4: -107.9720001, 96.0083313, -158.9818115, 124.6195374, -232.5915222, 254.9901428

Time for backsubstitution: 2.00 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=232.61239624023438
rel_dist={3: [-187.90965608592424, 187.9096560859242]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.7833230
time: 0.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6687624
time: 0.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.46 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -187.6676341, upper bound: 187.7833230
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.46
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6687624

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -146.9069672, 125.0689240, -245.0262451, 255.6538544
1: -93.8448410, 101.7164536, -115.1647415, 116.8796234, -210.7244568, 216.8811951
2: -135.7941437, 113.6401825, -166.5767059, 129.9570160, -265.7510986, 280.2168884
3: -54.4526558, 137.4407501, -62.5208206, 166.3341980, -220.7868347, 199.9615784
4: -151.2667084, 114.1601410, -185.2062378, 131.6817169, -282.9483643, 299.3663635

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
time: 0.61 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6687624
time: 0.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -182.1602936, 153.6605072, -147.2906494, 125.2701874, -307.4304810, 300.9511719
1: -143.0398865, 144.6755829, -115.4979095, 117.0591202, -260.0989685, 260.1734314
2: -206.5966339, 159.9870605, -167.0516052, 130.1336060, -336.7301636, 327.0386658
3: -78.3434219, 204.8266907, -62.6109695, 166.8166809, -245.1600952, 267.4376526
4: -229.7526093, 160.7959442, -185.7081909, 131.9127808, -361.6654053, 346.5041199

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6513714, upper bound: 187.6627654
time: 0.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6683139, upper bound: 187.6683139
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6676341
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6676341, upper bound: 187.6687624
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6513714, upper bound: 187.6627654
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 3, lower bound: -187.6683139, upper bound: 187.6683139

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -119.9573212, 108.7468872, -228.7041931, 228.7042084
1: -93.8448410, 101.7164536, -93.8448410, 101.7164536, -195.5612946, 195.5612946
2: -135.7941437, 113.6401825, -135.7941437, 113.6401825, -249.4343262, 249.4343262
3: -54.4526558, 137.4407501, -54.4526558, 137.4407501, -191.8934021, 191.8934021
4: -151.2667084, 114.1601410, -151.2667084, 114.1601410, -265.4268188, 265.4268188

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6622823, upper bound: 187.7825379
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673384, upper bound: 187.7826767
time: 0.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.9573212, 108.7468872, -180.2966766, 152.7804108, -272.7377319, 289.0435181
1: -93.8448410, 101.7164536, -141.5775146, 143.8618774, -237.7067261, 243.2939301
2: -135.7941437, 113.6401825, -204.5590515, 159.1097717, -294.9038391, 318.1992188
3: -54.4526558, 137.4407501, -77.8814087, 203.1510773, -257.6037292, 215.3221588
4: -151.2667084, 114.1601410, -227.4683685, 159.8488617, -311.1155396, 341.6284485

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6622823, upper bound: 187.7825379
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6673384, upper bound: 187.7826767
time: 0.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -155.1980591, 132.6422729, -86.6551056, 83.2883453, -238.4863892, 219.2973785
1: -121.8151321, 125.2165375, -67.9913635, 77.8647232, -199.6798553, 193.2079010
2: -176.1829376, 138.0224457, -98.6177521, 86.7079773, -262.8909302, 236.6401825
3: -66.9484940, 176.2607269, -40.8936234, 103.2635422, -170.2120361, 217.1543579
4: -196.0498505, 138.5000458, -110.0529633, 87.4945526, -283.5444031, 248.5529938

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6245132, upper bound: 187.6243218
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6251496
time: 0.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -180.1434021, 152.2499695, -142.9368286, 122.0671463, -302.2105408, 295.1867981
1: -141.4430695, 143.3300781, -112.0409241, 114.0217285, -255.4647980, 255.3710022
2: -204.3124390, 158.5519562, -162.1338043, 126.7824326, -331.0948792, 320.6857605
3: -77.6579437, 202.6321259, -60.9158020, 162.0751801, -239.7331238, 263.5479126
4: -227.2089844, 159.2999115, -180.2363281, 128.4469299, -355.6559143, 339.5361938

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6283582
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6290606
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6622823, upper bound: 187.7825379
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6673384, upper bound: 187.7826767
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6622823, upper bound: 187.7825379
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6673384, upper bound: 187.7826767
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6245132, upper bound: 187.6243218
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6251496
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6269162, upper bound: 187.6283582
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.17
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6290606

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -96.7168961, 90.5865326, -153.1854553, 165.3982849
1: -49.0822411, 64.2613220, -75.6750565, 84.9021606, -133.9844055, 139.9363708
2: -71.3749008, 71.9796066, -109.6172333, 94.5198593, -165.8947601, 181.5967865
3: -33.6786270, 77.7695847, -44.7764587, 112.7566147, -146.4352417, 122.5460434
4: -79.9356384, 71.8462219, -122.1720047, 95.0657730, -175.0014038, 194.0182190

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7290234, upper bound: 187.6247612
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248497, upper bound: 187.6246367
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -117.9716415, 107.3318253, -223.0831146, 223.7224274
1: -90.5305862, 98.8567810, -92.2776642, 100.3613434, -190.8919373, 191.1344452
2: -131.0461578, 110.5165710, -133.5452423, 112.1558685, -243.2020264, 244.0617676
3: -52.9895172, 132.8755493, -53.7486954, 135.2761688, -188.2656860, 186.6242371
4: -145.9682770, 110.9683304, -148.7582245, 112.6418839, -258.6101685, 259.7265625

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272868
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272885, upper bound: 187.6272885
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -62.5989532, 68.6813965, -153.9510651, 132.0519714, -194.6509094, 222.6324615
1: -49.0822411, 64.2613220, -120.8303986, 124.6718750, -173.7541199, 185.0917206
2: -71.3749008, 71.9796066, -174.8085480, 137.4363708, -208.8112793, 246.7881470
3: -33.6786270, 77.7695847, -66.6439514, 175.1317749, -208.8103943, 144.4135284
4: -79.9356384, 71.8462219, -194.5200195, 137.8570251, -217.7926636, 266.3662415

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6247712
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6244726, upper bound: 187.6265682
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -115.7513199, 105.7508011, -178.3051147, 151.3810120, -267.1322937, 284.0559082
1: -90.5305862, 98.8567810, -140.0000000, 142.5263672, -233.0569458, 238.8567810
2: -131.0461578, 110.5165710, -202.3019562, 157.6860352, -288.7321472, 312.8185120
3: -52.9895172, 132.8755493, -77.2020187, 200.9746399, -253.9641571, 210.0775757
4: -145.9682770, 110.9683304, -224.9540405, 158.3651886, -304.3334656, 335.9223633

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6294224
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -123.6794357, 114.4391174, -85.1616135, 82.3770905, -206.0565186, 199.6007080
1: -96.9159088, 108.2451935, -66.8189545, 77.0076523, -173.9235535, 175.0641479
2: -140.3470459, 119.6642990, -96.9166870, 85.7805176, -226.1275635, 216.5809937
3: -58.0654373, 142.9347076, -40.4399643, 101.6775208, -159.7429199, 183.3746643
4: -156.5294342, 119.0380554, -108.1737442, 86.5136642, -243.0430908, 227.2117920

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -185.4330750, 162.9749908, -86.0705566, 83.0001373, -268.4331970, 249.0455475
1: -145.5177002, 154.5120087, -67.5341873, 77.5772705, -223.0949249, 222.0462036
2: -210.5110321, 169.7353973, -97.9673538, 86.3945694, -296.9056091, 267.7027588
3: -83.1596222, 209.8917694, -40.7519760, 102.7173386, -185.2352295, 250.6437378
4: -234.5242004, 169.4522705, -109.3389359, 87.2138519, -321.7380371, 278.7911377

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6244726
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6244726
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -147.2054901, 133.2802734, -141.2478180, 121.0332184, -268.2387085, 274.5280457
1: -115.3470154, 125.7866287, -110.7067719, 113.0551910, -228.4022064, 236.4933929
2: -166.8895874, 139.7389832, -160.2056122, 125.7446060, -292.6341858, 299.9445801
3: -68.3726730, 167.9203033, -60.3968620, 160.2678375, -228.6405029, 228.3171692
4: -185.9134521, 139.0450897, -178.1076355, 127.3299408, -313.2433777, 317.1527100

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6262339
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6283582
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -212.5227051, 184.4253540, -141.6368256, 121.3328629, -333.8555298, 326.0621643
1: -166.9429779, 174.4524536, -111.0363083, 113.3082275, -280.2511597, 285.4887695
2: -241.0757599, 192.3849487, -160.6801453, 126.0063171, -367.0820312, 353.0650330
3: -94.8915787, 238.6475372, -60.5473785, 160.7881927, -254.9078064, 299.1949158
4: -268.3185425, 192.3304749, -178.6283112, 127.7026901, -396.0212097, 370.9587708

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6269199
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6269199
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.15 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.7290234, upper bound: 187.6247612
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6248497, upper bound: 187.6246367
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.7309013, upper bound: 187.6272868
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6272885, upper bound: 187.6272885
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6236270, upper bound: 187.6247712
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6244726, upper bound: 187.6265682
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6269199, upper bound: 187.6294224
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.5272371, upper bound: 187.6228919
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6244726
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6263104, upper bound: 187.6244726
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6262339
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6262340, upper bound: 187.6283582
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6269199
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 3, lower bound: -187.6290606, upper bound: 187.6269199

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -61.3601875, 67.9426346, -70.8088837, 74.5321655, -135.8923187, 138.7515259
1: -48.1152992, 63.5708046, -55.3257141, 69.9133606, -118.0286331, 118.8965149
2: -69.9924850, 71.2292633, -80.3077316, 78.2066879, -148.1991730, 151.5369873
3: -33.3106194, 76.4981613, -37.1244087, 85.3472137, -118.6578369, 113.6225662
4: -78.4040070, 71.0635452, -89.7878342, 78.0283203, -156.4323273, 160.8513489

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6172087, upper bound: 187.3729503
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7222837, upper bound: 187.6231596
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -61.5466385, 68.0616608, -125.1854477, 117.4320831, -178.9787140, 193.2471008
1: -48.2574577, 63.6742630, -97.9450531, 111.1974106, -159.4548492, 161.6193237
2: -70.2038116, 71.3310165, -141.9799957, 122.8201141, -193.0239258, 213.3110046
3: -33.3431396, 76.7382965, -59.8747673, 144.6909485, -178.0340881, 136.6130676
4: -78.6394806, 71.1897736, -158.4964294, 122.1660309, -200.8054962, 229.6862030

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5305762, upper bound: 187.3730317
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6239683, upper bound: 187.6230269
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -114.2607040, 104.8580704, -91.5989838, 91.4505310, -205.7112427, 196.4570465
1: -89.3597031, 98.0290298, -71.5373001, 85.6562042, -175.0158997, 169.5663147
2: -129.3549042, 109.6179657, -103.6967926, 96.2342453, -225.5891113, 213.3147583
3: -52.5823441, 131.2979889, -46.6185226, 107.3869629, -159.9692993, 177.9165039
4: -144.1015930, 110.0179825, -115.7632828, 95.8914185, -239.9930115, 225.7812195

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -114.1841736, 104.8378830, -148.1500549, 136.1284180, -250.3125916, 252.9879456
1: -89.2987900, 97.9758301, -115.8733826, 128.6213989, -217.9201813, 213.8492126
2: -129.2783661, 109.5616150, -167.8495941, 142.7842712, -272.0626221, 277.4111633
3: -52.5245056, 131.2808075, -70.0773010, 169.1162109, -221.6406860, 201.3581085
4: -144.0146179, 110.0222931, -187.2107849, 142.0318451, -286.0464478, 297.2330322

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5797557, upper bound: 187.4346536
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265252, upper bound: 187.6265252
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -61.3601875, 67.9426346, -123.6794357, 114.4391174, -175.7992554, 191.6220703
1: -48.1152992, 63.5708046, -96.9159088, 108.2451935, -156.3604889, 160.4867096
2: -69.9924850, 71.2292633, -140.3470459, 119.6642990, -189.6567841, 211.5763092
3: -33.3106194, 76.4981613, -58.0654373, 142.9347076, -176.2453003, 134.5635681
4: -78.4040070, 71.0635452, -156.5294342, 119.0380554, -197.4420624, 227.5929565

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5354784, upper bound: 187.3727671
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6227843, upper bound: 187.6231698
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -61.5466385, 68.0616608, -182.0503540, 160.9244690, -222.4710846, 250.1120148
1: -48.2574577, 63.6742630, -142.8406067, 152.5797882, -200.8241119, 206.5148621
2: -70.2038116, 71.3310165, -206.7423401, 167.6624756, -237.8662872, 278.0733643
3: -33.3431396, 76.7382965, -82.0790634, 206.6145477, -239.9576416, 158.0980377
4: -78.6394806, 71.1897736, -230.3176575, 167.2851868, -245.9246216, 301.5073853

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5503722, upper bound: 187.3761342
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6236011, upper bound: 187.6251408
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -114.2607040, 104.8580704, -147.2054901, 133.2802734, -247.5409851, 252.0635681
1: -89.3597031, 98.0290298, -115.3470154, 125.7866287, -215.1463318, 213.3760376
2: -129.3549042, 109.6179657, -166.8895874, 139.7389832, -269.0938721, 276.5075684
3: -52.5823441, 131.2979889, -68.3726730, 167.9203033, -220.5026245, 199.6706543
4: -144.1015930, 110.0179825, -185.9134521, 139.0450897, -283.1466675, 295.9314270

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -114.1841736, 104.8378830, -209.2748718, 182.7306061, -296.9147644, 314.1127625
1: -89.2987900, 97.9758301, -164.3894653, 172.8815308, -262.1802979, 262.3652954
2: -129.2783661, 109.5616150, -237.5092163, 190.6833801, -319.9617310, 347.0707703
3: -52.5245056, 131.2808075, -94.0009003, 235.6711731, -288.1956482, 224.4425964
4: -144.0146179, 110.0222931, -264.3191528, 190.5205231, -334.5351562, 374.3413696

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5899946, upper bound: 187.4394221
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261665, upper bound: 187.6289531
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -123.6794357, 114.4391174, -60.7287216, 67.5609207, -191.2403107, 175.1678314
1: -96.9159088, 108.2451935, -47.6900101, 63.1847534, -160.1006622, 155.9351959
2: -140.3470459, 119.6642990, -69.3438416, 70.8050919, -211.1521301, 189.0081024
3: -58.0654373, 142.9347076, -33.0708771, 76.1239090, -134.1893463, 176.0055847
4: -156.5294342, 119.0380554, -77.6507797, 70.6399307, -227.1693726, 196.6888275

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5057449, upper bound: 187.4232876
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -123.6794357, 114.4391174, -115.9698257, 110.1377106, -233.8171082, 230.4089355
1: -96.9159088, 108.2451935, -90.9609222, 104.2408829, -201.1567993, 199.2061157
2: -140.3470459, 119.6642990, -131.9120789, 115.0171890, -255.3642273, 251.5763855
3: -58.0654373, 142.9347076, -55.6980286, 135.7971954, -193.8625946, 198.6327362
4: -156.5294342, 119.0380554, -147.3544006, 114.5274887, -271.0569153, 266.3924561

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5057449, upper bound: 187.4232876
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6235391
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -184.3393555, 161.9893188, -61.5466385, 68.0616608, -252.4009857, 223.5359497
1: -144.6589966, 153.5589447, -48.2574577, 63.6742630, -208.3332520, 201.8067474
2: -209.2718964, 168.7155762, -70.2038116, 71.3310165, -280.6029053, 238.9193878
3: -82.6229553, 208.6881104, -33.3431396, 76.7382965, -158.6567230, 242.0312500
4: -233.1399536, 168.4353333, -78.6394806, 71.1897736, -304.3297119, 247.0748138

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5714726, upper bound: 187.4519846
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248828, upper bound: 187.6236010
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -186.3793030, 163.9709625, -114.9476852, 107.5441284, -293.9234314, 278.9186401
1: -146.2506104, 155.4834290, -90.2363510, 101.6353912, -247.8860016, 245.7197876
2: -211.5796509, 170.7777863, -130.7589264, 112.2637711, -323.8433838, 301.5367126
3: -83.7342224, 210.9392853, -54.1695328, 134.4276276, -217.3653412, 265.1088257
4: -235.7194214, 170.4737244, -145.9635925, 111.8617706, -347.5811768, 316.4373169

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5714726, upper bound: 187.4588528
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6248828, upper bound: 187.6236010
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -147.2054901, 133.2802734, -113.0714035, 104.0583725, -251.2638550, 246.3516846
1: -115.3470154, 125.7866287, -88.5095215, 97.2490158, -212.5960388, 214.2961426
2: -166.8895874, 139.7389832, -128.0974579, 108.7795715, -275.6691589, 267.8364258
3: -68.3726730, 167.9203033, -52.1716537, 130.2787628, -198.6514282, 220.0919495
4: -185.9134521, 139.0450897, -142.6618195, 109.1446457, -295.0580750, 281.7069092

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5927841, upper bound: 187.4671709
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254944
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -147.2054901, 133.2802734, -176.0679321, 151.9378662, -299.1433716, 309.3482056
1: -115.3470154, 125.7866287, -138.1637573, 143.1842499, -258.5312500, 263.9503784
2: -166.8895874, 139.7389832, -199.7817993, 158.3654480, -325.2550354, 339.5207825
3: -68.3726730, 167.9203033, -77.6897659, 198.9140625, -267.2867432, 245.6100769
4: -185.9134521, 139.0450897, -222.2691956, 158.8623505, -344.7757874, 361.3142700

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5927841, upper bound: 187.4711091
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254944
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -212.2793274, 184.1467438, -114.1841736, 104.8378830, -317.1172180, 298.3309326
1: -166.7501068, 174.1857758, -89.2987900, 97.9758301, -264.7259521, 263.4845581
2: -240.7987976, 192.0914459, -129.2783661, 109.5616150, -350.3603821, 321.3698120
3: -94.7369537, 238.3717041, -52.5245056, 131.2808075, -225.1919708, 290.8961792
4: -268.0097961, 192.0428162, -144.0146179, 110.0222931, -378.0320740, 336.0574036

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6243781
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6252037
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -212.6791992, 184.6053772, -174.8538513, 149.1306763, -361.8098755, 359.4591675
1: -167.0668182, 174.6245880, -137.2773285, 140.3892365, -307.4560547, 311.9019165
2: -241.2538605, 192.5746002, -198.4066315, 155.3790741, -396.6329346, 390.9812317
3: -94.9916992, 238.8250122, -76.0881729, 197.3055115, -291.5916748, 314.9131470
4: -268.5172729, 192.5163574, -220.6388245, 155.9823303, -424.4996033, 413.1551819

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6244065
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6252037
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6172087, upper bound: 187.3729503
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.7222837, upper bound: 187.6231596
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5305762, upper bound: 187.3730317
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6239683, upper bound: 187.6230269
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6272868, upper bound: 187.6272868
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5797557, upper bound: 187.4346536
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6265252, upper bound: 187.6265252
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5354784, upper bound: 187.3727671
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6227843, upper bound: 187.6231698
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5503722, upper bound: 187.3761342
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6236011, upper bound: 187.6251408
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6258430, upper bound: 187.6272853
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5899946, upper bound: 187.4394221
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6261665, upper bound: 187.6289531
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5057449, upper bound: 187.4232876
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5057449, upper bound: 187.4232876
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6235391
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5714726, upper bound: 187.4519846
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6248828, upper bound: 187.6236010
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5714726, upper bound: 187.4588528
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6248828, upper bound: 187.6236010
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5927841, upper bound: 187.4671709
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254944
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.5927841, upper bound: 187.4711091
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254944
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6243781
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6252037
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6244065
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 3, lower bound: -187.6231432, upper bound: 187.6252037

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -53.2885323, 63.4290314, -68.3843155, 73.2792206, -126.5677490, 131.8133392
1: -41.9746780, 59.3109512, -53.4824524, 68.7188416, -110.6935196, 112.7934036
2: -61.1721077, 66.6260986, -77.6385040, 76.9220657, -138.0941467, 144.2646027
3: -31.0217991, 68.6886520, -36.5083466, 82.9867096, -114.0084839, 105.1969986
4: -68.7092209, 66.2392120, -86.8633118, 76.6831131, -145.3923340, 153.1025238

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4663681, upper bound: 187.3729503
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4663681, upper bound: 187.3729503
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -70.2204819, 74.1192627, -133.0376129, 136.5172119
1: -46.1953201, 62.0187683, -54.8636055, 69.5222702, -115.7175903, 116.8823624
2: -67.2575836, 69.5302048, -79.6419601, 77.7782288, -145.0358124, 149.1721649
3: -32.4994354, 74.0081787, -36.9180298, 84.7186584, -117.2180786, 110.9262085
4: -75.4006653, 69.2964325, -89.0545197, 77.5919342, -152.9925995, 158.3509521

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604095
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -124.4439926, 116.9407120, -175.9891052, 190.8355560
1: -46.2957001, 62.0957146, -97.3622208, 110.7306213, -157.0263214, 159.4578857
2: -67.4097137, 69.6082687, -141.1396027, 122.3107147, -189.7204285, 210.7478485
3: -32.5202141, 74.2043915, -59.6231804, 143.9089203, -176.4291382, 133.8275757
4: -75.5679550, 69.3979874, -157.5689697, 121.6483917, -197.2163391, 226.9669495

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4170011, upper bound: 187.5592682
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4170011, upper bound: 187.6230270
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -91.5989838, 91.4505310, -180.8984528, 181.4808655
1: -69.8412399, 84.1715393, -71.5373001, 85.6562042, -155.4974213, 155.7088318
2: -101.2739258, 94.6007919, -103.6967926, 96.2342453, -197.5081177, 198.2975769
3: -45.8811989, 105.0105896, -46.6185226, 107.3869629, -153.2681580, 151.6291199
4: -113.0568390, 94.2403870, -115.7632828, 95.8914185, -208.9482422, 210.0036621

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7241665, upper bound: 187.6265245
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -145.8793182, 134.5294037, -91.5989838, 91.4505310, -237.3298492, 226.1283875
1: -114.0937805, 127.0895996, -71.5373001, 85.6562042, -199.7499695, 198.6268921
2: -165.2985535, 141.1205444, -103.6967926, 96.2342453, -261.5328064, 244.8173065
3: -69.2801971, 166.6479492, -46.6185226, 107.3869629, -176.6671600, 213.2664795
4: -184.3617096, 140.3413849, -115.7632828, 95.8914185, -280.2530823, 256.1046448

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7241665, upper bound: 187.6265245
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -103.4500961, 98.3777466, -145.3360901, 134.7312317, -238.1813354, 243.7138062
1: -81.0576172, 91.9590759, -113.7183151, 127.2939301, -208.3515472, 205.6773529
2: -117.3086472, 102.9799500, -164.7272797, 141.3537750, -258.6624146, 267.7071838
3: -49.2532768, 120.4372559, -69.3720016, 166.3632355, -215.6165161, 189.8092194
4: -130.8305664, 103.1562576, -183.7806396, 140.5365906, -271.3671570, 286.9368896

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4346536, upper bound: 187.4346536
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4346536, upper bound: 187.4346536
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -110.6350174, 102.5497208, -147.3744202, 135.6159363, -246.2509460, 249.9241333
1: -86.5048218, 95.8006287, -115.2622528, 128.1344147, -214.6392212, 211.0628510
2: -125.2559204, 107.1758804, -166.9682312, 142.2507629, -267.5066833, 274.1440125
3: -51.3641205, 127.6001511, -69.8214188, 168.2931366, -219.6572571, 197.4215698
4: -139.5886078, 107.5979919, -186.2393188, 141.4892578, -281.0778198, 293.8373108

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.6265252
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -122.9845581, 113.9999542, -172.9183044, 189.2813110
1: -46.1953201, 62.0187683, -96.3705902, 107.8279572, -154.0232849, 158.3893585
2: -67.2575836, 69.5302048, -139.5614929, 119.2086182, -186.4662018, 209.0917053
3: -32.4994354, 74.0081787, -57.8425446, 142.2068939, -174.7063293, 131.8507233
4: -75.4006653, 69.2964325, -155.6630554, 118.5721436, -193.9728088, 224.9594727

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604555
time: 0.57 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -53.4291878, 63.5372925, -178.4697723, 158.8351746, -212.2643585, 242.0070496
1: -42.0848007, 59.3990631, -140.0687714, 150.6054077, -192.5673523, 199.4678040
2: -61.3394165, 66.7152710, -202.7481384, 165.5332489, -226.8726501, 269.4634094
3: -31.0467987, 68.9024734, -81.0083389, 203.0190582, -234.0657959, 149.0968170
4: -68.8936691, 66.3546677, -225.9217987, 165.0424805, -233.9361420, 292.2764282

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4519861, upper bound: 187.3761342
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4519861, upper bound: 187.3761342
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -180.8627777, 160.0867767, -219.1351624, 247.2543335
1: -46.2957001, 62.0957146, -141.9023132, 151.7810974, -198.0233002, 203.9980164
2: -67.4097137, 69.6082687, -205.3982391, 166.8008575, -234.2105713, 275.0065002
3: -32.5202141, 74.2043915, -81.6491470, 205.3548431, -237.8750458, 155.1134338
4: -75.5679550, 69.3979874, -228.8333740, 166.4052734, -241.9732056, 298.2313538

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4519861, upper bound: 187.5714770
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4519861, upper bound: 187.6251409
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -89.4479752, 89.8819046, -147.2054901, 133.2802734, -222.7282104, 237.0874023
1: -69.8412399, 84.1715393, -115.3470154, 125.7866287, -195.6278381, 199.5185547
2: -101.2739258, 94.6007919, -166.8895874, 139.7389832, -241.0128784, 261.4903870
3: -45.8811989, 105.0105896, -68.3726730, 167.9203033, -213.8014832, 173.3832703
4: -113.0568390, 94.2403870, -185.9134521, 139.0450897, -252.1019287, 280.1538391

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4597248, upper bound: 187.5834151
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250969, upper bound: 187.6265237
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -146.0465698, 134.7337646, -147.2054901, 133.2802734, -279.3268433, 281.9392700
1: -114.2232513, 127.2852173, -115.3470154, 125.7866287, -240.0098572, 242.6322327
2: -165.4884644, 141.3374176, -166.8895874, 139.7389832, -305.2274475, 308.2269897
3: -69.3965378, 166.8359833, -68.3726730, 167.9203033, -237.3168335, 235.2086487
4: -184.5755615, 140.5543518, -185.9134521, 139.0450897, -323.6206665, 326.4678040

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4597249, upper bound: 187.5834151
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6250969, upper bound: 187.6265237
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -103.4500961, 98.3777466, -205.8919830, 180.7538757, -284.2039185, 304.2697144
1: -81.0576172, 91.9590759, -161.7704315, 171.0195618, -252.0771790, 253.7295074
2: -117.3086472, 102.9799500, -233.6740723, 188.6710510, -305.9796448, 336.6539001
3: -49.2532768, 120.4372559, -92.9916077, 232.2173309, -281.4705505, 212.4981995
4: -130.8305664, 103.1562576, -260.1016235, 188.4030457, -319.2336121, 363.2578735

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5613204, upper bound: 187.4222963
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5613205, upper bound: 187.4389173
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -110.6350174, 102.5497208, -208.0514069, 181.8733215, -292.5083008, 310.6011353
1: -86.5048218, 95.8006287, -163.3760223, 172.0660858, -258.5709229, 259.1766052
2: -125.2559204, 107.1758804, -236.1184235, 189.7994080, -315.0553284, 343.2943115
3: -51.3641205, 127.6001511, -93.5683594, 234.3666077, -285.7306824, 220.2883911
4: -139.5886078, 107.5979919, -262.7874451, 189.6113892, -329.1999817, 370.3854370

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6227679, upper bound: 187.6247160
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6227680, upper bound: 187.6261711
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -60.1746635, 67.1882477, -187.6997223, 172.6346741
1: -94.4305878, 106.3638000, -47.2545242, 62.8339310, -157.2645264, 153.6183167
2: -136.7642822, 117.6114349, -68.7233124, 70.4207535, -207.1850281, 186.3347473
3: -57.0565109, 139.6298981, -32.8888016, 75.5571136, -132.6136169, 172.5186768
4: -152.5781097, 116.9367599, -76.9693756, 70.2401505, -222.8182678, 193.9061127

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -115.2989349, 109.7152939, -230.2267609, 227.7589417
1: -94.4305878, 106.3638000, -90.4330292, 103.8395538, -198.2701416, 196.7968140
2: -136.7642822, 117.6114349, -131.1525726, 114.5792770, -251.3435669, 248.7640076
3: -57.0565109, 139.6298981, -55.4904213, 135.0913086, -192.1478271, 195.1203156
4: -152.5781097, 116.9367599, -146.5191650, 114.0791931, -266.6572876, 263.4559021

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6229115, upper bound: 187.6227843
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6229115, upper bound: 187.6227843
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -172.0058899, 154.9357758, -59.3307190, 66.9583130, -238.9641876, 214.2664642
1: -135.1005859, 146.9785767, -46.5870438, 62.6211090, -197.7216797, 193.1010437
2: -195.5021362, 161.5928802, -67.8055954, 70.1998596, -265.7019958, 229.1622009
3: -78.9017715, 196.2789307, -32.7779007, 74.6654663, -152.5998840, 229.0568237
4: -217.9936676, 160.9057312, -76.0007706, 69.9863968, -287.9800415, 236.9064484

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3761342, upper bound: 187.4519846
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3761342, upper bound: 187.4519846
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -179.8314514, 159.1389923, -60.9842834, 67.6798553, -247.5112762, 220.1232605
1: -141.0920868, 150.8634949, -47.8167267, 63.3138695, -204.4059601, 198.4385376
2: -204.1809692, 165.7945862, -69.5755615, 70.9376373, -275.1185913, 235.3701172
3: -81.1755676, 204.0319824, -33.1568451, 76.1665726, -156.4582672, 237.1888275
4: -227.5329285, 165.4163971, -77.9480286, 70.7807236, -298.3136292, 243.3643799

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3761342, upper bound: 187.5503713
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3761342, upper bound: 187.6236011
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -174.3274689, 157.1601257, -112.4117584, 106.3142319, -280.6416931, 269.5718994
1: -136.9112091, 149.1395874, -88.2992859, 100.4639435, -237.3751526, 236.9550781
2: -198.1261749, 163.9049072, -127.9554672, 111.0022125, -309.1283569, 291.6834717
3: -80.1450500, 198.8348846, -53.5516510, 131.9685822, -211.0159302, 252.3865356
4: -220.9257355, 163.1945190, -142.8901672, 110.5265503, -331.4522705, 306.0846558

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3986336, upper bound: 187.4588515
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3986336, upper bound: 187.4588515
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -182.1927795, 161.4089661, -114.2524414, 107.1134415, -289.3062134, 275.6613770
1: -142.9349670, 153.0692139, -89.6911850, 101.2265320, -244.1614532, 242.5100708
2: -206.8510437, 168.1558075, -129.9748383, 111.8170776, -318.6681213, 298.1306458
3: -82.4409790, 206.6351013, -53.9571991, 133.7005768, -215.1360168, 260.5922852
4: -230.5150452, 167.7524261, -145.1003723, 111.4027252, -341.9177856, 312.8527832

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3986336, upper bound: 187.5568501
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3986336, upper bound: 187.6236011
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -136.4013824, 126.9338074, -110.3064423, 102.6005936, -239.0019836, 237.2402496
1: -106.9984512, 119.9301987, -86.4005814, 95.8578262, -202.8562775, 206.3307343
2: -154.7998047, 133.3471069, -125.0340805, 107.2737427, -262.0735474, 258.3811646
3: -65.1247711, 156.9957428, -51.4286499, 127.5505447, -192.6753082, 208.4243927
4: -172.6357117, 132.2774811, -139.2882690, 107.5847092, -280.2204285, 271.5657349

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4669166
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4671710
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -143.0573883, 130.6116943, -112.2053223, 103.4862823, -246.5436707, 242.8170166
1: -112.0597992, 123.2650528, -87.8261642, 96.7073975, -208.7671967, 211.0912170
2: -162.1757965, 136.9888611, -127.1142273, 108.1839523, -270.3597412, 264.1030884
3: -67.0134430, 163.5671539, -51.8846245, 129.3755493, -196.3889771, 215.4517822
4: -180.7325897, 136.1973724, -141.5809631, 108.5352859, -289.2678528, 277.7783203

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -136.4013824, 126.9338074, -173.1470490, 150.3465271, -286.7479248, 300.0808411
1: -106.9984512, 119.9301987, -135.9157104, 141.6991425, -248.6976013, 255.8459015
2: -154.7998047, 133.3471069, -196.5363464, 156.7561188, -311.5559082, 329.8834229
3: -65.1247711, 156.9957428, -76.8711777, 196.0257874, -261.1505432, 233.8669128
4: -172.6357117, 132.2774811, -218.6999054, 157.1408539, -329.7765503, 350.9773865

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4597248
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4672622
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -143.0573883, 130.6116943, -175.1234741, 151.3096313, -294.3670044, 305.7351379
1: -112.0597992, 123.2650528, -137.4121399, 142.5909424, -254.6507111, 260.6771851
2: -162.1757965, 136.9888611, -198.7072906, 157.7182770, -319.8940735, 335.6961670
3: -67.0134430, 163.5671539, -77.3728180, 197.9146881, -264.9280701, 240.9399719
4: -180.7325897, 136.1973724, -221.0904694, 158.1954193, -338.9279480, 357.2877502

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -142.8028412, 136.6226196, -113.6884842, 104.5215073, -247.3243408, 249.4350891
1: -111.9867935, 129.7888641, -88.9057007, 97.6754227, -209.6622009, 216.9283295
2: -162.4577789, 142.7068176, -128.7084198, 109.2332916, -271.6910400, 269.3512878
3: -69.5772781, 165.6665039, -52.3516312, 130.7516937, -198.3083801, 218.0181274
4: -181.5161743, 141.4585419, -143.3885193, 109.6835403, -291.1997070, 283.8351440

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5613203
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6227680
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -208.7499237, 181.4654846, -114.1841736, 104.8378830, -313.5877686, 295.6496582
1: -163.9384003, 171.6240845, -89.2987900, 97.9758301, -261.9142151, 260.9228821
2: -236.8227234, 189.3229065, -129.2783661, 109.5616150, -346.3842773, 318.6012268
3: -93.3814468, 234.5383148, -52.5245056, 131.2808075, -223.8942871, 287.0628052
4: -263.5765991, 189.2199860, -144.0146179, 110.0222931, -373.5988770, 333.2345886

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5755471
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6244861
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -143.9574738, 137.5898285, -174.0307159, 148.5609131, -292.5183716, 311.0789185
1: -112.9016113, 130.6958466, -136.6201477, 139.8596649, -252.7612762, 265.8226624
2: -163.7802124, 143.6897125, -197.4616241, 154.7978210, -318.5780334, 339.4266663
3: -70.0736237, 166.9585724, -75.7977142, 196.4233398, -264.6083069, 242.7562561
4: -182.9930725, 142.4557343, -219.6032104, 155.3710327, -338.3640747, 361.5292969

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6244065
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6223351
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -210.2807465, 183.0075378, -174.8538513, 149.1306763, -359.4114380, 357.8613281
1: -165.1631775, 173.1105804, -137.2773285, 140.3892365, -305.5524292, 310.3879089
2: -238.5625763, 190.9392853, -198.4066315, 155.3790741, -393.9416504, 389.3458862
3: -94.2064362, 236.2539673, -76.0881729, 197.3055115, -290.7794495, 312.3421326
4: -265.5162659, 190.8201752, -220.6388245, 155.9823303, -421.4985962, 411.4589844

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251408
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251853
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.44 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4663681, upper bound: 187.3729503
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4663681, upper bound: 187.3729503
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604095
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4170011, upper bound: 187.5592682
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4170011, upper bound: 187.6230270
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.7241665, upper bound: 187.6265245
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.7241665, upper bound: 187.6265245
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4346536, upper bound: 187.4346536
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4346536, upper bound: 187.4346536
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4346536, upper bound: 187.6265252
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604555
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4519861, upper bound: 187.3761342
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4519861, upper bound: 187.3761342
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4519861, upper bound: 187.5714770
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4519861, upper bound: 187.6251409
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4597248, upper bound: 187.5834151
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6250969, upper bound: 187.6265237
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.4597249, upper bound: 187.5834151
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6250969, upper bound: 187.6265237
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5613204, upper bound: 187.4222963
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5613205, upper bound: 187.4389173
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6227679, upper bound: 187.6247160
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6227680, upper bound: 187.6261711
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5051512, upper bound: 187.6220017
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6229115, upper bound: 187.6227843
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6229115, upper bound: 187.6227843
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3761342, upper bound: 187.4519846
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3761342, upper bound: 187.4519846
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3761342, upper bound: 187.5503713
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3761342, upper bound: 187.6236011
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3986336, upper bound: 187.4588515
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3986336, upper bound: 187.4588515
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3986336, upper bound: 187.5568501
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3986336, upper bound: 187.6236011
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4669166
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4671710
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4597248
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4672622
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5613203
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6227680
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5755471
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6244861
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6244065
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6223351
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251408
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251853

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -62.0726891, 69.5171814, -128.4355164, 128.3694305
1: -46.1953201, 62.0187683, -48.6528091, 65.1671600, -111.3624802, 110.6715775
2: -67.2575836, 69.5302048, -70.6422958, 73.1033783, -140.3609619, 140.1725006
3: -32.4994354, 74.0081787, -34.5528183, 76.6450348, -109.1444626, 108.5609970
4: -75.4006653, 69.2964325, -79.1664734, 72.7514496, -148.1521149, 148.4629059

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604085
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604085
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -68.3037109, 72.8281784, -131.7464905, 134.6004333
1: -46.1953201, 62.0187683, -53.3555756, 68.2969208, -114.4922409, 115.3743439
2: -67.2575836, 69.5302048, -77.4724121, 76.4414444, -143.6990356, 147.0026245
3: -32.4994354, 74.0081787, -36.2477036, 82.7130051, -115.2124329, 110.2558670
4: -75.4006653, 69.2964325, -86.6691895, 76.2220230, -151.6226807, 155.9656219

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -114.4981918, 111.5533295, -170.6017151, 180.8897552
1: -46.2957001, 62.0957146, -89.7365723, 105.6287766, -151.9244690, 151.8322906
2: -67.4097137, 69.6082687, -130.0746307, 116.8258362, -184.2355499, 199.6828766
3: -32.5202141, 74.2043915, -56.7771721, 133.9736786, -166.4938812, 130.9815674
4: -75.5679550, 69.3979874, -145.4030151, 115.9632645, -191.5312195, 214.8009796

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4169875, upper bound: 187.5592672
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4169875, upper bound: 187.5592672
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -121.8557358, 115.2981262, -174.3464813, 188.2472687
1: -46.2957001, 62.0957146, -95.3280029, 109.1697159, -155.4653931, 157.4236908
2: -67.4097137, 69.6082687, -138.2064056, 120.6092834, -188.0189972, 207.8146667
3: -32.5202141, 74.2043915, -58.7909508, 141.1980438, -173.7182617, 132.9953461
4: -75.5679550, 69.3979874, -154.3339233, 119.9142532, -195.4821625, 223.7319031

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4169875, upper bound: 187.6230270
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4169875, upper bound: 187.6230270
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -86.7375641, 88.4901047, -80.7479401, 85.0878143, -171.8253174, 169.2380219
1: -67.7763672, 82.8458176, -63.1928024, 79.7481689, -147.5245361, 146.0386200
2: -98.2780533, 93.1718140, -91.5943222, 89.8128357, -188.0908813, 184.7661438
3: -45.1791992, 102.3533783, -43.4452820, 96.4653091, -141.6444702, 145.7986603
4: -109.7604904, 92.7649384, -102.4525833, 89.1877670, -198.9482422, 195.2175293

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4890677, upper bound: 187.4890677
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4890677, upper bound: 187.6949280
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6560287, 89.3579025, -88.1947784, 89.2497559, -177.9057922, 177.5526733
1: -69.2166138, 83.6777191, -68.8570099, 83.5735397, -152.7901306, 152.5347290
2: -100.3754883, 94.0587158, -99.8368530, 93.9555893, -194.3310547, 193.8955536
3: -45.6202240, 104.1861343, -45.4965744, 103.8610992, -149.4813232, 149.6826935
4: -112.0700226, 93.6839752, -111.5177612, 93.5536041, -205.6235962, 205.2017212

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6949278, upper bound: 187.4890677
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6949280, upper bound: 187.8026410
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -143.1246948, 133.1262817, -80.7479401, 85.0878143, -228.2124634, 213.8742218
1: -111.9873199, 125.7556381, -63.1928024, 79.7481689, -191.7354889, 188.9483795
2: -162.2446136, 139.6806793, -91.5943222, 89.8128357, -252.0574493, 231.2749939
3: -68.5647812, 163.9500732, -43.4452820, 96.4653091, -165.0300446, 207.3953400
4: -181.0043945, 138.8412018, -102.4525833, 89.1877670, -270.1921692, 241.2937927

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4883843, upper bound: 187.4346654
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -145.0540161, 133.9693756, -88.1947784, 89.2497559, -234.3037720, 222.1641541
1: -113.4438171, 126.5572662, -68.8570099, 83.5735397, -197.0173340, 195.4142609
2: -164.3603973, 140.5375671, -99.8368530, 93.9555893, -258.3159790, 240.3744202
3: -68.9971237, 165.7720337, -45.4965744, 103.8610992, -172.8581848, 211.2686005
4: -183.3270569, 139.7488708, -111.5177612, 93.5536041, -276.8806763, 251.2666321

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6563819, upper bound: 187.4346654
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6563819, upper bound: 187.6265245
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -110.6350174, 102.5497208, -137.4818420, 130.0579834, -240.6929932, 240.0315552
1: -86.5048218, 95.8006287, -107.6646576, 122.9698868, -209.4746857, 203.4652710
2: -125.2559204, 107.1758804, -155.9312439, 136.6390381, -261.8949585, 263.1070862
3: -51.3641205, 127.6001511, -66.9975357, 158.3645020, -209.7286072, 194.5976715
4: -139.5886078, 107.5979919, -174.1071167, 135.5888519, -275.1774597, 281.7051086

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797171
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797515
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -110.6350174, 102.5497208, -144.6886292, 133.8912964, -244.5263062, 247.2383423
1: -86.5048218, 95.8006287, -113.1467667, 126.4890060, -212.9938049, 208.9473877
2: -125.2559204, 107.1758804, -163.9182129, 140.4533539, -265.7092896, 271.0939941
3: -51.3641205, 127.6001511, -68.9462738, 165.4748535, -216.8389740, 196.5464172
4: -139.5886078, 107.5979919, -182.8783569, 139.6602325, -279.2488403, 290.4763489

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.6265252
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -113.9726257, 109.2072830, -168.1256256, 180.2693176
1: -46.1953201, 62.0187683, -89.4736710, 103.3015976, -149.4969177, 151.4924164
2: -67.2575836, 69.5302048, -129.5501556, 114.3353577, -181.5929108, 199.0803528
3: -32.4994354, 74.0081787, -55.3262329, 133.2557526, -165.7551575, 129.3344116
4: -75.4006653, 69.2964325, -144.6617126, 113.5089493, -188.9096069, 213.9581299

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604545
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604545
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -58.9183464, 66.2967453, -120.5114899, 112.4600143, -171.3783569, 186.8081970
1: -46.1953201, 62.0187683, -94.4305878, 106.3638000, -152.5591125, 156.4493256
2: -67.2575836, 69.5302048, -136.7642822, 117.6114349, -184.8690186, 206.2944946
3: -32.4994354, 74.0081787, -57.0565109, 139.6298981, -172.1293335, 131.0646973
4: -75.4006653, 69.2964325, -152.5781097, 116.9367599, -192.3374176, 221.8745422

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -170.1371918, 153.9749908, -213.0233612, 236.5287476
1: -46.2957001, 62.0957146, -133.6181335, 146.0840759, -191.8556976, 195.7138367
2: -67.4097137, 69.6082687, -193.4321594, 160.6344757, -227.7147980, 263.0404358
3: -32.5202141, 74.2043915, -78.4069366, 194.5491943, -227.0693970, 151.5801086
4: -75.5679550, 69.3979874, -215.6818542, 159.8786774, -235.4466248, 285.0798340

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4224109, upper bound: 187.5714759
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4224109, upper bound: 187.5604631
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -59.0483932, 66.3915634, -177.8010559, 158.1509705, -217.1993713, 244.1926270
1: -46.2957001, 62.0957146, -139.4800568, 149.9519501, -196.0126038, 201.5757599
2: -67.4097137, 69.6082687, -201.9336853, 164.8150940, -232.2247925, 271.5419312
3: -32.5202141, 74.2043915, -80.6691971, 202.1686249, -234.6888123, 153.9914703
4: -75.5679550, 69.3979874, -225.0239563, 164.3520050, -239.9199524, 294.4219055

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4224110, upper bound: 187.6251375
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4224109, upper bound: 187.6231699
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -86.7375641, 88.4901047, -136.4013824, 126.9338074, -213.6713562, 224.8914795
1: -67.7763672, 82.8458176, -106.9984512, 119.9301987, -187.7065735, 189.8442688
2: -98.2780533, 93.1718140, -154.7998047, 133.3471069, -231.6251526, 247.9716187
3: -45.1791992, 102.3533783, -65.1247711, 156.9957428, -202.1749420, 167.4781189
4: -109.7604904, 92.7649384, -172.6357117, 132.2774811, -242.0379486, 265.4006348

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4669166, upper bound: 187.4892407
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4669166, upper bound: 187.6719509
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -88.6560287, 89.3579025, -143.0573883, 130.6116943, -219.2677307, 232.4152832
1: -69.2166138, 83.6777191, -112.0597992, 123.2650528, -192.4816589, 195.7375183
2: -100.3754883, 94.0587158, -162.1757965, 136.9888611, -237.3643494, 256.2344971
3: -45.6202240, 104.1861343, -67.0134430, 163.5671539, -209.1873779, 171.1995392
4: -112.0700226, 93.6839752, -180.7325897, 136.1973724, -248.2673950, 274.4165649

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4892407
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5933491, upper bound: 187.7246656
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -143.3497925, 133.4030457, -136.4013824, 126.9338074, -270.2835999, 269.8044434
1: -112.1615753, 126.0206375, -106.9984512, 119.9301987, -232.0917511, 233.0190887
2: -162.5002594, 139.9744263, -154.7998047, 133.3471069, -295.8473511, 294.7742310
3: -68.7219391, 164.2039642, -65.1247711, 156.9957428, -225.7176819, 229.3287201
4: -181.2921753, 139.1296387, -172.6357117, 132.2774811, -313.5696411, 311.7653198

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.4597248, upper bound: 187.4346985
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.4597248, upper bound: 187.5834151
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -145.2531586, 134.2131195, -143.0573883, 130.6116943, -275.8648682, 277.2705078
1: -113.5979767, 126.7905884, -112.0597992, 123.2650528, -236.8630219, 238.8503876
2: -164.5865936, 140.7962494, -162.1757965, 136.9888611, -301.5754395, 302.9720459
3: -69.1358871, 165.9959717, -67.0134430, 163.5671539, -232.7030182, 233.0094147
4: -183.5817108, 140.0029297, -180.7325897, 136.1973724, -319.7790222, 320.7355347

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -102.9278564, 98.0382767, -139.6066132, 134.9120178, -236.7263794, 237.6448975
1: -80.6436691, 91.6370316, -109.5284042, 128.1481018, -206.8662415, 201.1654358
2: -116.7074127, 102.6258392, -158.9042969, 140.9403381, -255.3727112, 261.5301208
3: -49.0654907, 119.8790131, -68.7114487, 162.5013123, -211.5668030, 186.4791412
4: -130.1715698, 102.7921143, -177.6101685, 139.6204529, -268.5254211, 280.4022217

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4222963
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4148367
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -103.4500961, 98.3777466, -202.3826599, 178.0569763, -281.5069885, 300.7603760
1: -81.0576172, 91.9590759, -159.0069885, 168.4462280, -249.5038452, 250.9660492
2: -117.3086472, 102.9799500, -229.7624512, 185.8877106, -303.1963501, 332.7424011
3: -49.2532768, 120.4372559, -91.6285629, 228.4269867, -277.6802368, 211.2062531
4: -130.8305664, 103.1562576, -255.7361298, 185.5626221, -316.3931580, 358.8923950

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4389173
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4343763
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -110.1260223, 102.2211380, -141.7084961, 135.8848419, -245.0078125, 243.9296265
1: -86.1022797, 95.4882812, -111.1242142, 129.0836182, -213.3345184, 206.6124878
2: -124.6722107, 106.8343506, -161.2149658, 141.9429169, -264.4393616, 268.0492859
3: -51.1841621, 127.0574265, -69.2084732, 164.5036926, -215.6878510, 194.1904144
4: -138.9471130, 107.2469330, -180.1426392, 140.6805115, -278.4614258, 287.3895569

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6228842
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6238455
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -110.6350174, 102.5497208, -204.4423828, 179.1090088, -289.7440186, 306.9920959
1: -86.5048218, 95.8006287, -160.4992523, 169.4315948, -255.9364166, 256.2998657
2: -125.2559204, 107.1758804, -232.0440369, 186.9504395, -312.2063599, 339.2199097
3: -51.3641205, 127.6001511, -92.1728210, 230.4297943, -281.7938843, 218.9549713
4: -139.5886078, 107.5979919, -258.2484741, 186.6992950, -326.2879028, 365.8464661

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6261711
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6248332
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -40.3613243, 54.2850304, -174.7965088, 152.8213196
1: -94.4305878, 106.3638000, -31.6318417, 50.9045906, -145.3351593, 137.9956360
2: -136.7642822, 117.6114349, -46.3339272, 57.2926102, -194.0568848, 163.9453583
3: -57.0565109, 139.6298981, -27.0197716, 54.3477020, -111.4042130, 166.6496735
4: -152.5781097, 116.9367599, -52.2693520, 56.6317139, -209.2098236, 169.2060852

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5037126, upper bound: 187.5037783
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5037126, upper bound: 187.6220017
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -86.1480865, 91.6282730, -212.1397705, 198.6080933
1: -94.4305878, 106.3638000, -67.5872040, 86.7583618, -181.1889038, 173.9509888
2: -136.7642822, 117.6114349, -98.0902328, 96.1821594, -232.9464417, 215.7016602
3: -57.0565109, 139.6298981, -46.5560608, 104.1080551, -161.1645660, 186.1859436
4: -152.5781097, 116.9367599, -109.9391327, 95.0345612, -247.6126556, 226.8758698

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.5037126, upper bound: 187.5037783
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.5037126, upper bound: 187.6220017
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -89.3759232, 95.3453140, -215.8567810, 201.8359375
1: -94.4305878, 106.3638000, -70.0084152, 90.3265457, -184.7571411, 176.3721924
2: -136.7642822, 117.6114349, -101.7516785, 99.9713974, -236.7356873, 219.3630981
3: -57.0565109, 139.6298981, -48.5536575, 107.5397034, -164.5962219, 188.1835632
4: -152.5781097, 116.9367599, -114.0960464, 98.8496094, -251.4277191, 231.0328064

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6190618, upper bound: 187.5043674
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6190618, upper bound: 187.6227843
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -120.5114899, 112.4600143, -141.6408844, 136.1860046, -255.8318176, 254.1008911
1: -94.4305878, 106.3638000, -111.1266632, 129.3168640, -222.0037842, 217.4904480
2: -136.7642822, 117.6114349, -161.2119293, 142.1561890, -277.0828857, 278.8233643
3: -57.0565109, 139.6298981, -69.3030243, 164.6562958, -221.7127991, 207.0476990
4: -152.5781097, 116.9367599, -180.1463013, 140.9905701, -292.7587891, 297.0830688

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6190618, upper bound: 187.5043674
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6190618, upper bound: 187.6227843
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -179.5513916, 158.9033813, -53.4291878, 63.5372925, -243.0886841, 212.3325653
1: -140.8713379, 150.6365967, -42.0848007, 59.3990631, -200.2703705, 192.3703308
2: -203.8637695, 165.5520325, -61.3394165, 66.7152710, -270.5790405, 226.8031464
3: -81.0509415, 203.7251282, -31.0467987, 68.9024734, -148.9833527, 234.7718964
4: -227.1792450, 165.1725616, -68.8936691, 66.3546677, -293.5339050, 234.0662079

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5345709
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5345709
time: 0.71 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604085
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4663681, upper bound: 187.5604085
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4663681, upper bound: 187.6231597
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4169875, upper bound: 187.5592672
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4169875, upper bound: 187.5592672
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4169875, upper bound: 187.6230270
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4169875, upper bound: 187.6230270
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4890677, upper bound: 187.4890677
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4890677, upper bound: 187.6949280
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6949278, upper bound: 187.4890677
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6949280, upper bound: 187.8026410
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4883843, upper bound: 187.4346654
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4883843, upper bound: 187.5833507
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6563819, upper bound: 187.4346654
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6563819, upper bound: 187.6265245
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797171
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797515
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4346536, upper bound: 187.6265252
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4346536, upper bound: 187.5797557
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604545
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4225394, upper bound: 187.5604545
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4225394, upper bound: 187.6231699
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4224109, upper bound: 187.5714759
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4224109, upper bound: 187.5604631
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4224110, upper bound: 187.6251375
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4224109, upper bound: 187.6231699
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4669166, upper bound: 187.4892407
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4669166, upper bound: 187.6719509
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5933491, upper bound: 187.4892407
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5933491, upper bound: 187.7246656
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4597248, upper bound: 187.4346985
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.4597248, upper bound: 187.5834151
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5765998, upper bound: 187.4346985
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4222963
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4148367
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4389173
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5053603, upper bound: 187.4343763
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6228842
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6238455
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6261711
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5048692, upper bound: 187.6248332
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5037126, upper bound: 187.5037783
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5037126, upper bound: 187.6220017
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5037126, upper bound: 187.5037783
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.5037126, upper bound: 187.6220017
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6190618, upper bound: 187.5043674
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6190618, upper bound: 187.6227843
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6190618, upper bound: 187.5043674
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.6190618, upper bound: 187.6227843
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5345709
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.18
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5345709
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.3761342, upper bound: 187.6236011
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.3986336, upper bound: 187.5568501
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.3986336, upper bound: 187.6236011
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4669166
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5936253, upper bound: 187.4671710
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6254945, upper bound: 187.6254605
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4597248
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5834151, upper bound: 187.4672622
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6261644, upper bound: 187.6250969
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5613203
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6227680
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.3737669, upper bound: 187.5755471
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.6218708, upper bound: 187.6244861
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6244065
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6223351
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251408
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.18
Output dim: 3, lower bound: -187.5280726, upper bound: 187.6251853
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=232.61239624023438
rel_dist={3: [-187.89872093335524, 187.8987209333552]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1136.78 seconds
