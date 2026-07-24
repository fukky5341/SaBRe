## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 466.672919624136


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-200.7323151, 313.5108337, -200.7323151, 313.5108337, -514.2431641, 514.2431641)
1: (-154.8143463, 288.4224854, -154.8143463, 288.4224854, -443.2368164, 443.2368164)
2: (-136.0845337, 298.5967407, -136.0845337, 298.5967407, -434.6812744, 434.6812744)
3: (-210.5117645, 294.8744812, -210.5117645, 294.8744812, -505.3862305, 505.3862305)
4: (-164.3547821, 316.4496765, -164.3547821, 316.4496765, -480.8044434, 480.8044434)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.49 + 1.91 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -466.6775864, upper bound: 466.6775864

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748479, upper bound: 466.6743041
time: 0.93 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743005, upper bound: 466.6743005
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.94 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -466.6748479, upper bound: 466.6743041
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -466.6743005, upper bound: 466.6743005

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -200.7323151, 313.5108337, -182.0487671, 281.9833984, -482.7156677, 495.5596008
1: -154.8143463, 288.4224854, -140.1936951, 259.6987915, -414.5131226, 428.6160889
2: -136.0845337, 298.5967407, -123.1656647, 269.2485962, -405.3331299, 421.7623291
3: -210.5117645, 294.8744812, -191.1080322, 266.1956787, -476.7074585, 485.9824829
4: -164.3547821, 316.4496765, -148.8809204, 286.0896301, -450.4443665, 465.3305969

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742277
time: 0.76 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742989
time: 0.60 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -197.9027405, 309.1196899, -511.5330200, 773.8687134, -970.5332642, 815.0314331
1: -152.6286011, 284.3979187, -389.6335754, 708.8753662, -860.5180054, 670.9335327
2: -134.1536713, 294.3879395, -342.8289795, 738.1580200, -870.3645020, 634.1976318
3: -207.4603577, 290.7559509, -532.8462524, 731.9035034, -937.8784790, 819.4674683
4: -162.0148315, 311.9281616, -416.0361328, 788.5063477, -947.5808716, 725.4478149

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742989, upper bound: 466.6742295
time: 0.69 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742989, upper bound: 466.6743005
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.94 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742277
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742989
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 0, lower bound: -466.6742989, upper bound: 466.6742295
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 0, lower bound: -466.6742989, upper bound: 466.6743005

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -182.0487671, 281.9833984, -182.0487671, 281.9833984, -464.0321350, 464.0321350
1: -140.1936951, 259.6987915, -140.1936951, 259.6987915, -399.8924255, 399.8924255
2: -123.1656647, 269.2485962, -123.1656647, 269.2485962, -392.4142151, 392.4142151
3: -191.1080322, 266.1956787, -191.1080322, 266.1956787, -457.3036804, 457.3036804
4: -148.8809204, 286.0896301, -148.8809204, 286.0896301, -434.9705200, 434.9705200

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6734473, upper bound: 466.6717465
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6734473, upper bound: 466.6742059
time: 0.79 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -504.5384521, 761.9297485, -182.0487671, 281.9833984, -780.9023438, 942.8170776
1: -384.0342407, 698.1108398, -140.1936951, 259.6987915, -640.6350708, 837.3647461
2: -337.7992554, 726.9852295, -123.1656647, 269.2485962, -604.0620728, 848.2469482
3: -525.3290405, 721.0104980, -191.1080322, 266.1956787, -787.4302368, 910.6485596
4: -409.9612732, 776.8063965, -148.8809204, 286.0896301, -693.5400391, 922.7835083

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6718435, upper bound: 466.6731358
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748206, upper bound: 466.6742785
time: 0.77 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -182.0487671, 281.9833984, -504.5384827, 761.9297485, -942.8170776, 780.9023438
1: -140.1936951, 259.6987915, -384.0342407, 698.1107788, -837.3646851, 640.6350708
2: -123.1656647, 269.2485962, -337.7992554, 726.9851685, -848.2468872, 604.0621338
3: -191.1080322, 266.1956787, -525.3290405, 721.0104980, -910.6485596, 787.4302368
4: -148.8809204, 286.0896301, -409.9613037, 776.8063354, -922.7834473, 693.5399780

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6717450
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6742054
time: 0.76 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -535.4615479, 813.5322266, -535.4615479, 813.5322266, -1341.4660645, 1341.4660645
1: -408.6176147, 744.8612061, -408.6176147, 744.8612061, -1148.9763184, 1148.9763184
2: -359.8183594, 775.3198853, -359.8183594, 775.3198853, -1129.8016357, 1129.8016357
3: -558.2575073, 768.5037231, -558.2575073, 768.5037231, -1320.4251709, 1320.4252930
4: -436.6341858, 827.5334473, -436.6341858, 827.5334473, -1258.1224365, 1258.1224365

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6720120
time: 0.68 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742024, upper bound: 466.6742764
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.08 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6734473, upper bound: 466.6717465
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6734473, upper bound: 466.6742059
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6718435, upper bound: 466.6731358
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6748206, upper bound: 466.6742785
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6717450
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6742054
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6730019, upper bound: 466.6720120
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.08
Output dim: 0, lower bound: -466.6742024, upper bound: 466.6742764

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -184.6222687, 286.1645813, -180.9404144, 280.3025513, -464.9247437, 467.1049500
1: -142.3757935, 263.5417175, -139.3454590, 258.1489258, -400.5246887, 402.8870850
2: -125.0713654, 273.2555237, -122.4199371, 267.6402893, -392.7116699, 395.6754761
3: -194.0663452, 270.1241455, -189.9491425, 264.5991821, -458.6654968, 460.0733032
4: -151.1575623, 290.3919067, -147.9775391, 284.3712158, -435.5287781, 438.3693542

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6440895, upper bound: 466.6604896
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6734456, upper bound: 466.6718440
time: 1.00 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -178.8057404, 277.0134583, -182.0487671, 281.9833984, -460.7891235, 459.0621948
1: -137.6861877, 255.1209106, -140.1936951, 259.6987915, -397.3849792, 395.3145447
2: -120.9570541, 264.5043335, -123.1656647, 269.2485962, -390.2056580, 387.6699219
3: -187.6841125, 261.4776917, -191.1080322, 266.1956787, -453.8797607, 452.5856628
4: -146.2034760, 281.0205078, -148.8809204, 286.0896301, -432.2930908, 429.9014282

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6603939, upper bound: 466.6457852
time: 0.74 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6453754, upper bound: 466.6453753
time: 0.70 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -503.3503113, 759.9846191, -184.6222687, 286.1645813, -783.8101196, 943.4393921
1: -383.0332642, 696.3745117, -142.3757935, 263.5417175, -643.4003296, 837.7830200
2: -336.9135742, 725.2073975, -125.0713654, 273.2555237, -607.1408081, 848.3577271
3: -523.9440918, 719.2279053, -194.0663452, 270.1241455, -789.8715820, 911.7379761
4: -408.8747864, 774.8745117, -151.1575623, 290.3919067, -696.7062988, 923.0387573

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6706720, upper bound: 466.6728195
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6718313, upper bound: 466.6730881
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -503.9179077, 760.9266968, -178.8057404, 277.0134583, -775.3056030, 938.5753174
1: -383.5465088, 697.2071533, -137.6861877, 255.1209106, -635.5726318, 833.9620361
2: -337.3607788, 726.0389404, -120.9570541, 264.5043335, -598.8823242, 845.1024170
3: -524.6547852, 720.0855103, -187.6841125, 261.4776917, -782.0521240, 906.3048096
4: -409.4287720, 775.7975464, -146.2034760, 281.0205078, -687.9444580, 919.1113892

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6745217, upper bound: 466.6741228
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748136, upper bound: 466.6742358
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -184.6222687, 286.1645813, -503.3503113, 759.9846191, -943.4393921, 783.8101196
1: -142.3757935, 263.5417175, -383.0332642, 696.3745117, -837.7830200, 643.4003296
2: -125.0713654, 273.2555237, -336.9135742, 725.2073975, -848.3577271, 607.1408081
3: -194.0663452, 270.1241455, -523.9440918, 719.2279053, -911.7379761, 789.8715820
4: -151.1575623, 290.3919067, -408.8747864, 774.8745117, -923.0387573, 696.7062988

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6728195, upper bound: 466.6706720
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730881, upper bound: 466.6718313
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -178.8057404, 277.0134583, -503.9179077, 760.9266968, -938.5753174, 775.3056030
1: -137.6861877, 255.1209106, -383.5465088, 697.2071533, -833.9620361, 635.5726318
2: -120.9570541, 264.5043335, -337.3607788, 726.0389404, -845.1024170, 598.8823242
3: -187.6841125, 261.4776917, -524.6547852, 720.0855103, -906.3048096, 782.0521240
4: -146.2034760, 281.0205078, -409.4287720, 775.7975464, -919.1113892, 687.9444580

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6741228, upper bound: 466.6745217
time: 0.92 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742358, upper bound: 466.6748136
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -538.7124634, 818.6244507, -534.2661133, 811.6925659, -1342.9010010, 1345.2103271
1: -411.3185730, 749.7399902, -407.6927490, 743.1719360, -1149.9205322, 1152.8310547
2: -362.1410828, 780.3756714, -359.0053101, 773.5693359, -1130.3474121, 1133.9838867
3: -562.0626221, 773.6127930, -556.9993286, 766.7658081, -1322.3157959, 1324.1209717
4: -439.3822327, 833.0296631, -435.6494141, 825.6678467, -1258.9282227, 1262.5482178

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6629529, upper bound: 466.6546961
time: 0.62 seconds

## Relational analysis of NS_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6720846, upper bound: 466.6647702
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6709129, upper bound: 466.6667007
time: 0.67 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6727696, upper bound: 466.6710418
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -532.5598145, 809.0067139, -535.4615479, 813.5322266, -1338.5256348, 1336.8946533
1: -406.3498535, 740.7058105, -408.6176147, 744.8612061, -1146.6971436, 1144.7966309
2: -357.8172607, 771.0238647, -359.8183594, 775.3198853, -1127.7883301, 1125.4812012
3: -555.1863403, 764.2391357, -558.2575073, 768.5037231, -1317.3258057, 1316.1429443
4: -434.2049255, 822.9564209, -436.6341858, 827.5334473, -1255.6898193, 1253.5238037

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6722845, upper bound: 466.6691425
time: 0.79 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6740302, upper bound: 466.6740318
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.39 seconds
NS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6440895, upper bound: 466.6604896
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6734456, upper bound: 466.6718440
NS_B1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6603939, upper bound: 466.6457852
NS_B1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6453754, upper bound: 466.6453753
NS_B1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6706720, upper bound: 466.6728195
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6718313, upper bound: 466.6730881
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6745217, upper bound: 466.6741228
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6748136, upper bound: 466.6742358
NS_B2_A1_A1_A1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6728195, upper bound: 466.6706720
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6730881, upper bound: 466.6718313
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6741228, upper bound: 466.6745217
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6742358, upper bound: 466.6748136
NS_B2_A2_A1_A1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6709129, upper bound: 466.6667007
NS_B2_A2_A1_A2, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6727696, upper bound: 466.6710418
NS_B2_A2_A2_A1, status: Status.VERIFIED, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6722845, upper bound: 466.6691425
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.39
Output dim: 0, lower bound: -466.6740302, upper bound: 466.6740318

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -184.6222687, 286.1645813, -180.4205322, 279.4895630, -464.1117554, 466.5850830
1: -142.3757935, 263.5417175, -138.9455261, 257.3991089, -399.7748718, 402.4871521
2: -125.0713654, 273.2555237, -122.0685501, 266.8666077, -391.9379578, 395.3240662
3: -194.0663452, 270.1241455, -189.4079742, 263.8320007, -457.8983154, 459.5321045
4: -151.1575623, 290.3919067, -147.5521240, 283.5542603, -434.7118225, 437.9439087

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6618177, upper bound: 466.6531913
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6448121, upper bound: 466.6525228
time: 0.79 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -506.0240784, 767.3338623, -341.1174927, 519.7399902, -1019.4884033, 1105.2902832
1: -386.0679016, 702.4789429, -261.9853210, 477.2941589, -859.7298584, 962.4024658
2: -339.8491211, 731.3196411, -230.7795258, 496.6697998, -832.6799316, 959.0236816
3: -527.8622437, 725.1848145, -358.9993286, 491.1134644, -1013.7828979, 1080.9741211
4: -412.4129333, 781.1448364, -279.7787476, 531.3535156, -939.8012085, 1056.5117188

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6536946, upper bound: 466.6629557
time: 0.82 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6682426, upper bound: 466.6713922
time: 0.68 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674464, upper bound: 466.6674723
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -497.3532715, 750.4360962, -163.7498474, 252.4905853, -744.2727661, 913.0855103
1: -378.4034119, 687.5358887, -126.0533066, 232.6803436, -608.0014038, 812.7019653
2: -332.7658081, 716.0266724, -110.6711273, 241.4557648, -571.2608643, 824.8433838
3: -517.6959839, 710.1900635, -172.0185699, 238.7624054, -752.4233398, 880.7614746
4: -403.8456421, 765.1777344, -133.8149567, 256.8652039, -658.2209473, 896.1412964

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6710503, upper bound: 466.6689679
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743596, upper bound: 466.6738753
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -507.2097473, 769.1473389, -329.5036011, 502.2285156, -1003.2465820, 1095.5812988
1: -386.9778748, 704.1373291, -252.8796844, 461.0344543, -844.4756470, 954.9912720
2: -340.6484680, 733.0398560, -222.8173065, 479.8270569, -816.7131348, 952.7937012
3: -529.1030884, 726.8908691, -346.5607605, 474.2444763, -998.2620850, 1070.4321289
4: -413.3792725, 782.9776611, -270.1044006, 513.2299194, -922.7573853, 1048.6958008

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741920
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6741893
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -341.1174927, 519.7399902, -506.0240784, 767.3338623, -1105.2902832, 1019.4884033
1: -261.9853210, 477.2941589, -386.0679016, 702.4789429, -962.4024658, 859.7298584
2: -230.7795258, 496.6697998, -339.8491211, 731.3196411, -959.0236816, 832.6799316
3: -358.9993286, 491.1134644, -527.8622437, 725.1848145, -1080.9741211, 1013.7828979
4: -279.7787476, 531.3535156, -412.4129333, 781.1448364, -1056.5117188, 939.8012085

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6617804, upper bound: 466.6536946
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6713923, upper bound: 466.6682426
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674723, upper bound: 466.6682153
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -163.7498474, 252.4905853, -497.3532715, 750.4360962, -913.0855103, 744.2727051
1: -126.0533066, 232.6803436, -378.4034119, 687.5358887, -812.7019653, 608.0014038
2: -110.6711273, 241.4557648, -332.7658081, 716.0266724, -824.8433838, 571.2608643
3: -172.0185699, 238.7624054, -517.6959839, 710.1900635, -880.7614746, 752.4233398
4: -133.8149567, 256.8652039, -403.8456421, 765.1777344, -896.1412964, 658.2209473

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6689674, upper bound: 466.6710503
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6738753, upper bound: 466.6743596
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -329.5036011, 502.2285156, -507.2097473, 769.1473389, -1095.5812988, 1003.2465820
1: -252.8796844, 461.0344543, -386.9778748, 704.1373291, -954.9912720, 844.4756470
2: -222.8173065, 479.8270569, -340.6484680, 733.0398560, -952.7937012, 816.7131348
3: -346.5607605, 474.2444763, -529.1030884, 726.8908691, -1070.4322510, 998.2620850
4: -270.1044006, 513.2299194, -413.3792725, 782.9776611, -1048.6956787, 922.7573853

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A2_A2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6741920, upper bound: 466.6705615
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6741893, upper bound: 466.6748114
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -529.7191772, 804.6381836, -535.4615479, 813.5322266, -1335.6461182, 1332.5032959
1: -404.1515198, 736.6981812, -408.6176147, 744.8612061, -1144.4960938, 1140.7713623
2: -355.8838501, 766.8588867, -359.8183594, 775.3198853, -1125.8493652, 1121.2963867
3: -552.2074585, 760.1257935, -558.2575073, 768.5037231, -1314.3127441, 1312.0100098
4: -431.8552856, 818.5110474, -436.6341858, 827.5334473, -1253.3382568, 1249.0593262

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6691424, upper bound: 466.6722845
time: 0.74 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6687238, upper bound: 466.6740318
time: 0.90 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.18 seconds
NS_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6618177, upper bound: 466.6531913
NS_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6448121, upper bound: 466.6525228
NS_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6682426, upper bound: 466.6713922
NS_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6674464, upper bound: 466.6674723
NS_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6710503, upper bound: 466.6689679
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6743596, upper bound: 466.6738753
NS_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741920
NS_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6741893
NS_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6713923, upper bound: 466.6682426
NS_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6674723, upper bound: 466.6682153
NS_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6689674, upper bound: 466.6710503
NS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6738753, upper bound: 466.6743596
NS_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6741920, upper bound: 466.6705615
NS_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6741893, upper bound: 466.6748114
NS_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6691424, upper bound: 466.6722845
NS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 0, lower bound: -466.6687238, upper bound: 466.6740318

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -494.0023499, 745.2824707, -163.7498474, 252.4905853, -740.8752441, 907.9065552
1: -375.8238831, 682.8123169, -126.0533066, 232.6803436, -605.4147339, 807.9555054
2: -330.5021973, 711.1200562, -110.6711273, 241.4557648, -568.9829102, 819.9124146
3: -514.1983643, 705.3467407, -172.0185699, 238.7624054, -748.8858032, 875.8942261
4: -401.1003418, 759.9479980, -133.8149567, 256.8652039, -655.4688721, 890.8905029

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6655329, upper bound: 466.6712133
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6654927, upper bound: 466.6651483
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -502.4072266, 761.3099976, -305.2243347, 462.8858337, -959.1259155, 1063.3336182
1: -383.1831665, 696.9084473, -233.7934265, 424.7604980, -804.3529053, 928.5755615
2: -337.3149109, 725.5808105, -206.1056519, 442.3580017, -775.8322144, 928.5475464
3: -523.9473877, 719.5294189, -320.5698242, 437.2911072, -956.0181274, 1036.8833008
4: -409.3670654, 775.1167603, -249.9573212, 473.6711426, -879.0523682, 1020.5637817

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6605354, upper bound: 466.6677500
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6699745
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741887
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -501.3897400, 760.3381958, -348.8673401, 533.6099854, -1028.7371826, 1106.1333008
1: -382.5330505, 696.0931396, -267.8384094, 489.7584534, -868.7968750, 961.9151001
2: -336.7343750, 724.6937866, -236.0829468, 509.5709229, -842.5686646, 957.6826172
3: -523.0307617, 718.6187134, -366.7487488, 503.3572693, -1021.3637085, 1082.4708252
4: -408.6310425, 774.0482178, -286.1520081, 544.5222168, -949.3831787, 1055.8078613

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6699761
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6741892
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -163.7498474, 252.4905853, -494.0023193, 745.2824707, -907.9064941, 740.8752441
1: -126.0533066, 232.6803436, -375.8238525, 682.8123169, -807.9555054, 605.4146729
2: -110.6711273, 241.4557648, -330.5021973, 711.1200562, -819.9123535, 568.9829102
3: -172.0185699, 238.7624054, -514.1983643, 705.3467407, -875.8942261, 748.8858032
4: -133.8149567, 256.8652039, -401.1003418, 759.9479980, -890.8905029, 655.4688721

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6712133, upper bound: 466.6655329
time: 0.76 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6640114, upper bound: 466.6654927
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -305.2243347, 462.8858337, -502.4072266, 761.3099976, -1063.3336182, 959.1259155
1: -233.7934265, 424.7604980, -383.1831665, 696.9084473, -928.5755615, 804.3529053
2: -206.1056519, 442.3580017, -337.3149109, 725.5808105, -928.5475464, 775.8322144
3: -320.5698242, 437.2911072, -523.9473877, 719.5294189, -1036.8833008, 956.0181274
4: -249.9573212, 473.6711426, -409.3670654, 775.1167603, -1020.5637817, 879.0523682

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6677500, upper bound: 466.6605354
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A2_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6699745, upper bound: 466.6705615
time: 0.87 seconds

## Relational analysis of NS_B2_A1_A2_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6697074, upper bound: 466.6705615
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -348.8673401, 533.6099854, -501.3897400, 760.3381958, -1106.1333008, 1028.7371826
1: -267.8384094, 489.7584534, -382.5330505, 696.0931396, -961.9151001, 868.7968750
2: -236.0829468, 509.5709229, -336.7343750, 724.6937866, -957.6826172, 842.5686646
3: -366.7487488, 503.3572693, -523.0307617, 718.6187134, -1082.4708252, 1021.3637085
4: -286.1520081, 544.5222168, -408.6310425, 774.0482178, -1055.8078613, 949.3831787

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A2_A2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
time: 0.72 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -529.7191772, 804.6381836, -532.6256104, 809.1684570, -1331.2600098, 1329.6291504
1: -404.1515198, 736.6981812, -406.4233398, 740.8583374, -1140.4759521, 1138.5745850
2: -355.8838501, 766.8588867, -357.8869019, 771.1603394, -1121.6697998, 1119.3605957
3: -552.2074585, 760.1257935, -555.2838745, 764.3944092, -1310.1839600, 1309.0028076
4: -431.8552856, 818.5110474, -434.2860413, 823.0908813, -1248.8803711, 1246.7110596

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6666998, upper bound: 466.6727579
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6660637, upper bound: 466.6740318
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.66 seconds
NS_B1_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6655329, upper bound: 466.6712133
NS_B1_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6654927, upper bound: 466.6651483
NS_B1_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6699745
NS_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741887
NS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6699761
NS_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6748114, upper bound: 466.6741892
NS_B2_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6712133, upper bound: 466.6655329
NS_B2_A1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6640114, upper bound: 466.6654927
NS_B2_A1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6699745, upper bound: 466.6705615
NS_B2_A1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6697074, upper bound: 466.6705615
NS_B2_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
NS_B2_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
NS_B2_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6666998, upper bound: 466.6727579
NS_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 0, lower bound: -466.6660637, upper bound: 466.6740318

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -554.2918701, 839.1940918, -305.2243347, 462.8858337, -1011.7447510, 1141.7562256
1: -422.8950195, 769.0070190, -233.7934265, 424.7604980, -844.1547852, 1001.0273438
2: -372.2871094, 800.1646118, -206.1056519, 442.3580017, -810.9429321, 1003.4324951
3: -578.3071899, 793.7550049, -320.5698242, 437.2911072, -1010.7477417, 1111.3706055
4: -451.5564575, 854.9867554, -249.9573212, 473.6711426, -921.3270874, 1100.7736816

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6680235, upper bound: 466.6716202
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683399, upper bound: 466.6732599
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -489.8486938, 741.2465820, -348.8673401, 533.6099854, -1017.0054321, 1086.8873291
1: -373.3805847, 678.4306030, -267.8384094, 489.7584534, -859.5251465, 944.1469116
2: -328.7257385, 706.4754639, -236.0829468, 509.5709229, -834.4689941, 939.3543091
3: -510.6364136, 700.6336670, -366.7487488, 503.3572693, -1008.7499390, 1064.3546143
4: -398.9835510, 754.9353638, -286.1520081, 544.5222168, -939.6313477, 1036.5129395

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6671228, upper bound: 466.6699528
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6699761
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -554.5989990, 839.8128052, -348.8673401, 533.6099854, -1082.6947021, 1186.1414795
1: -423.1687622, 769.5574341, -267.8384094, 489.7584534, -909.5159302, 1035.7365723
2: -372.5292969, 800.7277222, -236.0829468, 509.5709229, -878.5030518, 1034.0229492
3: -578.6651001, 794.2982178, -366.7487488, 503.3572693, -1077.3769531, 1158.4141846
4: -451.8421936, 855.5605469, -286.1520081, 544.5222168, -992.6725464, 1137.6501465

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6671228, upper bound: 466.6741628
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741854
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -348.8673401, 533.6099854, -489.8486938, 741.2466431, -1086.8874512, 1017.0054321
1: -267.8384094, 489.7584534, -373.3805847, 678.4306030, -944.1469116, 859.5251465
2: -236.0829468, 509.5709229, -328.7257385, 706.4755859, -939.3543091, 834.4689941
3: -366.7487488, 503.3572693, -510.6364136, 700.6337280, -1064.3546143, 1008.7499390
4: -286.1520081, 544.5222168, -398.9835510, 754.9353638, -1036.5129395, 939.6313477

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6699525, upper bound: 466.6683964
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
time: 1.08 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -348.8673401, 533.6099854, -554.5986938, 839.8118896, -1186.1406250, 1082.6943359
1: -267.8384094, 489.7584534, -423.1684265, 769.5567627, -1035.7358398, 909.5156860
2: -236.0829468, 509.5709229, -372.5289917, 800.7270508, -1034.0224609, 878.5027466
3: -366.7487488, 503.3572693, -578.6646118, 794.2976074, -1158.4135742, 1077.3764648
4: -286.1520081, 544.5222168, -451.8417969, 855.5598755, -1137.6492920, 992.6721802

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6699528, upper bound: 466.6683964
time: 0.72 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
time: 0.92 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -529.7191772, 804.6381836, -529.7191772, 804.6381836, -1326.6833496, 1326.6834717
1: -404.1515198, 736.6981812, -404.1515198, 736.6981812, -1136.2911377, 1136.2911377
2: -355.8838501, 766.8588867, -355.8838501, 766.8588867, -1117.3438721, 1117.3438721
3: -552.2074585, 760.1257935, -552.2074585, 760.1257935, -1305.8975830, 1305.8975830
4: -431.8552856, 818.5110474, -431.8552856, 818.5110474, -1244.2750244, 1244.2750244

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6653703, upper bound: 466.6701288
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6653419, upper bound: 466.6653375
time: 0.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.05 seconds
NS_B1_A2_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6680235, upper bound: 466.6716202
NS_B1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6683399, upper bound: 466.6732599
NS_B1_A2_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6671228, upper bound: 466.6699528
NS_B1_A2_B2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6699761
NS_B1_A2_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6671228, upper bound: 466.6741628
NS_B1_A2_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6705615, upper bound: 466.6741854
NS_B2_A1_A2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6699525, upper bound: 466.6683964
NS_B2_A1_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
NS_B2_A1_A2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6699528, upper bound: 466.6683964
NS_B2_A1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6699761, upper bound: 466.6748114
NS_B2_A2_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6653703, upper bound: 466.6701288
NS_B2_A2_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.05
Output dim: 0, lower bound: -466.6653419, upper bound: 466.6653375

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -553.0928955, 837.2928467, -299.2965393, 453.4571228, -1001.0858154, 1133.8864746
1: -421.9629517, 767.2556152, -229.1671448, 415.9996643, -834.4420776, 994.6275635
2: -371.4670410, 798.3548584, -202.0482330, 433.3185120, -801.0646362, 997.5378418
3: -577.0523071, 791.9658203, -314.2959900, 428.3584900, -1000.5268555, 1103.2834473
4: -450.5682068, 853.0767822, -245.0795746, 464.1211853, -910.7655029, 1093.9473877

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6672070, upper bound: 466.6732381
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683399, upper bound: 466.6732599
time: 0.79 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -554.5228882, 839.6699219, -339.4084473, 519.1185303, -1068.1250000, 1176.4866943
1: -423.1038208, 769.4295044, -260.6213379, 476.3722534, -896.0554810, 1028.3792725
2: -372.4715576, 800.5967407, -229.7625275, 495.6573181, -864.5154419, 1027.5512695
3: -578.5794678, 794.1707153, -356.8570251, 489.5912170, -1063.5144043, 1148.3782959
4: -451.7732239, 855.4255981, -278.5040283, 529.7271118, -977.7724609, 1129.8420410

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6683555, upper bound: 466.6674788
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683555, upper bound: 466.6741628
time: 0.95 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -553.2535400, 837.7460938, -344.0790100, 526.2192993, -1073.9600830, 1179.3524170
1: -422.1426392, 767.6522217, -264.1509094, 482.9623718, -901.7033691, 1030.1663818
2: -371.6253967, 798.7530518, -232.8102722, 502.5033569, -870.5503540, 1028.8017578
3: -577.2723389, 792.3446045, -361.7051086, 496.3778381, -1069.0133057, 1151.4295654
4: -450.7466125, 853.4645386, -282.1394043, 536.9675903, -984.0415649, 1131.5610352

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6747063, upper bound: 466.6674969
time: 0.75 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6747063, upper bound: 466.6741854
time: 0.88 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -488.7911987, 739.6100464, -1080.5261230, 1008.5583496
1: -264.1509094, 482.9623718, -372.5681763, 676.9367065, -938.9862671, 851.9265137
2: -232.8102722, 502.5033569, -328.0088806, 704.9291382, -934.5601807, 826.7040405
3: -361.7051086, 496.3778381, -509.5301514, 699.0974731, -1057.7911377, 1000.6756592
4: -282.1394043, 536.9675903, -398.1129761, 753.2936401, -1030.8796387, 931.2252808

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6696592, upper bound: 466.6748023
time: 0.98 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6691104, upper bound: 466.6748136
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -553.2532349, 837.7451782, -1179.3516846, 1073.9598389
1: -264.1509094, 482.9623718, -422.1422729, 767.6517334, -1030.1658936, 901.7030029
2: -232.8102722, 502.5033569, -371.6251526, 798.7523193, -1028.8010254, 870.5501099
3: -361.7051086, 496.3778381, -577.2719116, 792.3439941, -1151.4288330, 1069.0128174
4: -282.1394043, 536.9675903, -450.7462769, 853.4638672, -1131.5604248, 984.0412598

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6674953, upper bound: 466.6747868
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6674246, upper bound: 466.6748114
time: 0.80 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.92 seconds
NS_B1_A2_B2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6672070, upper bound: 466.6732381
NS_B1_A2_B2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6683399, upper bound: 466.6732599
NS_B1_A2_B2_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6683555, upper bound: 466.6674788
NS_B1_A2_B2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6683555, upper bound: 466.6741628
NS_B1_A2_B2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6747063, upper bound: 466.6674969
NS_B1_A2_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6747063, upper bound: 466.6741854
NS_B2_A1_A2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6696592, upper bound: 466.6748023
NS_B2_A1_A2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6691104, upper bound: 466.6748136
NS_B2_A1_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6674953, upper bound: 466.6747868
NS_B2_A1_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.92
Output dim: 0, lower bound: -466.6674246, upper bound: 466.6748114

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -553.0087891, 837.1351318, -290.7890320, 440.5244446, -988.0598145, 1125.1711426
1: -421.8909912, 767.1142578, -222.6928406, 404.0790100, -822.4257202, 987.9957886
2: -371.4034424, 798.2092285, -196.3810120, 420.9020386, -788.5573120, 991.7061768
3: -576.9569092, 791.8248291, -305.4374695, 416.1007385, -988.1466064, 1094.2490234
4: -450.4924927, 852.9265747, -238.2182007, 450.8859558, -897.4196777, 1086.9133301

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6671784, upper bound: 466.6664039
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6671784, upper bound: 466.6732381
time: 0.70 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -551.6849365, 835.1094360, -296.1350098, 448.4288025, -994.6458740, 1128.5770264
1: -420.8834229, 765.2458496, -226.6773376, 411.3562317, -828.7258301, 990.1279297
2: -370.5160217, 796.2713623, -199.8285828, 428.5183411, -795.3351440, 993.2360229
3: -575.5880127, 789.9078369, -310.9472351, 423.5663757, -994.2771606, 1097.8851318
4: -449.4166565, 850.8682251, -242.3560944, 459.0430908, -904.5552979, 1089.0137939

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6683158, upper bound: 466.6664039
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683158, upper bound: 466.6732599
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -550.8919678, 833.7911377, -339.4084473, 519.1185303, -1064.4107666, 1170.5716553
1: -420.2763367, 763.9958496, -260.6213379, 476.3722534, -893.1499023, 1022.9010010
2: -369.9736938, 794.9967041, -229.7625275, 495.6573181, -861.9556885, 1021.8927002
3: -574.8028564, 788.6419067, -356.8570251, 489.5912170, -1059.6007080, 1142.7547607
4: -448.7453918, 849.5626221, -278.5040283, 529.7271118, -974.6417236, 1123.8813477

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6733109
time: 1.16 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664859, upper bound: 466.6730520
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -547.6615601, 829.2694702, -344.0790100, 526.2192993, -1068.3153076, 1170.8338623
1: -417.8936157, 759.8475952, -264.1509094, 482.9623718, -897.4292603, 1022.3353271
2: -367.9224243, 790.6256104, -232.8102722, 502.5033569, -866.8237305, 1020.6436768
3: -571.4342041, 784.3201904, -361.7051086, 496.3778381, -1063.1406250, 1143.3836670
4: -446.2666016, 844.7935791, -282.1394043, 536.9675903, -979.5443726, 1122.8627930

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6662124
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6664859, upper bound: 466.6663346
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -550.9490356, 833.9036255, -344.0790100, 526.2192993, -1071.5889893, 1175.4815674
1: -420.3265076, 764.0960693, -264.1509094, 482.9623718, -899.8178101, 1026.5729980
2: -370.0181580, 795.0993652, -232.8102722, 502.5033569, -868.8900757, 1025.0987549
3: -574.8687134, 788.7407227, -361.7051086, 496.3778381, -1066.4896240, 1147.7438965
4: -448.7978210, 849.6672363, -282.1394043, 536.9675903, -982.0007935, 1127.6807861

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6733239
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664598, upper bound: 466.6730633
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -482.9952393, 730.7723389, -1071.6406250, 1002.7236328
1: -264.1509094, 482.9623718, -368.1610107, 668.7924805, -930.8171387, 847.4995728
2: -232.8102722, 502.5033569, -324.1664124, 696.4382935, -926.0397339, 822.8416748
3: -361.7051086, 496.3778381, -503.4935303, 690.7434082, -1049.4171143, 994.6063232
4: -282.1394043, 536.9675903, -393.4659424, 744.2617798, -1021.8143311, 926.5657959

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6688139, upper bound: 466.6735554
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6732014
time: 1.02 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -488.6470032, 739.1542969, -1080.0560303, 1008.4119873
1: -264.1509094, 482.9623718, -372.4525452, 676.4891968, -938.5259399, 851.7744751
2: -232.8102722, 502.5033569, -327.8976440, 704.5051270, -934.1152954, 826.5771484
3: -361.7051086, 496.3778381, -509.4729309, 698.6383667, -1057.3122559, 1000.5684204
4: -282.1394043, 536.9675903, -397.9746399, 752.9923096, -1030.5383301, 931.0521240

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675440, upper bound: 466.6719416
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6670531, upper bound: 466.6738772
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -547.6505127, 829.2471313, -1170.8123779, 1068.3041992
1: -264.1509094, 482.9623718, -417.8836975, 759.8277588, -1022.3156128, 897.4194946
2: -232.8102722, 502.5033569, -367.9137268, 790.6050415, -1020.6230469, 866.8149414
3: -361.7051086, 496.3778381, -571.4215088, 784.3006592, -1143.3638916, 1063.1276855
4: -282.1394043, 536.9675903, -446.2563782, 844.7727661, -1122.8419189, 979.5340576

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6662174, upper bound: 466.6735969
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
time: 0.72 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -550.9532471, 833.9098511, -1175.4876709, 1071.5933838
1: -264.1509094, 482.9623718, -420.3295898, 764.1019897, -1026.5788574, 899.8209229
2: -232.8102722, 502.5033569, -370.0209961, 795.1051025, -1025.1044922, 868.8927612
3: -361.7051086, 496.3778381, -574.8729248, 788.7468872, -1147.7501221, 1066.4937744
4: -282.1394043, 536.9675903, -448.8014526, 849.6735840, -1127.6873779, 982.0043945

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6662174, upper bound: 466.6736288
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736109
time: 1.17 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.35 seconds
NS_B1_A2_B2_B2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6671784, upper bound: 466.6664039
NS_B1_A2_B2_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6671784, upper bound: 466.6732381
NS_B1_A2_B2_B2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6683158, upper bound: 466.6664039
NS_B1_A2_B2_B2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6683158, upper bound: 466.6732599
NS_B1_A2_B2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6733109
NS_B1_A2_B2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664859, upper bound: 466.6730520
NS_B1_A2_B2_B2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6662124
NS_B1_A2_B2_B2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664859, upper bound: 466.6663346
NS_B1_A2_B2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664771, upper bound: 466.6733239
NS_B1_A2_B2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6664598, upper bound: 466.6730633
NS_B2_A1_A2_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6688139, upper bound: 466.6735554
NS_B2_A1_A2_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6732014
NS_B2_A1_A2_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6675440, upper bound: 466.6719416
NS_B2_A1_A2_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6670531, upper bound: 466.6738772
NS_B2_A1_A2_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6662174, upper bound: 466.6735969
NS_B2_A1_A2_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
NS_B2_A1_A2_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6662174, upper bound: 466.6736288
NS_B2_A1_A2_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.35
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736109

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -549.3738403, 831.2535400, -290.7890320, 440.5244446, -984.3402710, 1119.2503662
1: -419.0617981, 761.6783447, -222.6928406, 404.0790100, -819.5175171, 982.5126343
2: -368.9036255, 792.6073608, -196.3810120, 420.9020386, -785.9949951, 986.0441895
3: -573.1794434, 786.2919922, -305.4374695, 416.1007385, -984.2305298, 1088.6203613
4: -447.4618225, 847.0618896, -238.2182007, 450.8859558, -894.2856445, 1080.9498291

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6650940, upper bound: 466.6721956
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6650982, upper bound: 466.6722549
time: 1.02 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -549.3801880, 831.2720947, -296.1350098, 448.4288025, -992.2736816, 1124.7084961
1: -419.0690002, 761.6941528, -226.6773376, 411.3562317, -826.8409424, 986.5369263
2: -368.9100342, 792.6233521, -199.8285828, 428.5183411, -793.6752319, 989.5369263
3: -573.1884155, 786.3070068, -310.9472351, 423.5663757, -991.7558594, 1094.2010498
4: -447.4687195, 847.0773315, -242.3560944, 459.0430908, -902.5147705, 1085.1385498

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6650940, upper bound: 466.6721144
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6650982, upper bound: 466.6721725
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -543.7746582, 822.6301270, -339.4084473, 519.1185303, -1057.2091064, 1159.3450928
1: -414.7484131, 753.7407837, -260.6213379, 476.3722534, -887.5764160, 1012.5866699
2: -365.1218872, 784.3876343, -229.7625275, 495.6573181, -857.0594482, 1011.2210693
3: -567.3039551, 778.1112061, -356.8570251, 489.5912170, -1052.0146484, 1132.1650391
4: -442.8753967, 838.3087158, -278.5040283, 529.7271118, -968.7234497, 1112.5507812

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
time: 0.99 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -544.8867188, 824.3342896, -337.6278687, 516.3663940, -1055.6406250, 1159.3519287
1: -415.4660339, 755.0897827, -259.2262573, 473.8384705, -885.8006592, 1012.6281738
2: -365.7870789, 785.7985840, -228.5378113, 493.0277100, -855.1321411, 1011.4808350
3: -568.2041016, 779.4776001, -354.9654541, 486.9896240, -1050.4208984, 1131.7071533
4: -443.6148376, 839.7804565, -277.0122681, 526.9191284, -966.7031860, 1112.6207275

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -543.8307495, 822.7410889, -344.0790100, 526.2192993, -1064.3862305, 1164.2539062
1: -414.7979126, 753.8394775, -264.1509094, 482.9623718, -894.2435913, 1016.2575073
2: -365.1658020, 784.4889526, -232.8102722, 502.5033569, -863.9930420, 1014.4258423
3: -567.3690796, 778.2088623, -361.7051086, 496.3778381, -1058.9028320, 1137.1529541
4: -442.9271545, 838.4118042, -282.1394043, 536.9675903, -976.0816650, 1116.3489990

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
time: 0.70 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -544.9423828, 824.4448853, -342.2144470, 523.3282471, -1062.6787109, 1164.1755371
1: -415.5152893, 755.1881714, -262.7015686, 480.2991028, -892.3373413, 1016.2437744
2: -365.8307495, 785.8995361, -231.5363464, 499.7409668, -861.9325562, 1014.6358643
3: -568.2688599, 779.5746460, -359.7274170, 493.6423950, -1057.1748047, 1136.6085205
4: -443.6664124, 839.8834229, -280.5881042, 534.0244751, -973.9192505, 1116.3588867

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
time: 0.73 seconds

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
time: 0.75 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -476.1569519, 720.0625610, -1060.8591309, 995.8090210
1: -264.1509094, 482.9623718, -362.8572998, 658.9207153, -920.8952026, 842.1497192
2: -232.8102722, 502.5033569, -319.5157776, 686.2137451, -915.7531738, 818.1466675
3: -361.7051086, 496.3778381, -496.3040466, 680.6254272, -1039.2497559, 987.3379517
4: -282.1394043, 536.9675903, -387.8558350, 733.4370728, -1010.9155273, 920.9055786

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6715705
time: 0.92 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6734956
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -342.2144470, 523.3282471, -472.6814575, 712.9640503, -1052.2783203, 989.5947876
1: -262.7015686, 480.2991028, -359.7912292, 652.4592285, -913.1787109, 836.5023193
2: -231.5363464, 499.7409668, -316.8333130, 679.5402222, -907.9756470, 812.7887573
3: -359.7274170, 493.6423950, -492.1355286, 674.2324219, -1030.9927979, 980.6322632
4: -280.5881042, 534.0244751, -384.6148071, 726.5900269, -1002.7216187, 914.7991943

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6635243, upper bound: 466.6680002
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6557957, upper bound: 466.6680517
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6735422
time: 0.91 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6735422
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -342.8888245, 524.3237305, -483.2924500, 730.7142334, -1070.3630371, 1001.0819092
1: -263.2195740, 481.2023926, -368.2741699, 668.6846313, -929.7385254, 845.7898560
2: -231.9944000, 500.6903381, -324.2255859, 696.4290161, -925.1658325, 821.0473633
3: -360.4470520, 494.5859985, -503.8159485, 690.6456909, -1048.0097656, 993.0414429
4: -281.1574707, 535.0511475, -393.5464783, 744.4370728, -1020.9296875, 924.6614990

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6672118, upper bound: 466.6728779
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6656811, upper bound: 466.6725284
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -540.5716553, 818.1635742, -1159.6608887, 1061.1434326
1: -264.1509094, 482.9623718, -412.3824768, 749.6259766, -1012.0539551, 891.8745728
2: -232.8102722, 502.5033569, -363.0892029, 780.0614624, -1010.0166016, 861.9477539
3: -361.7051086, 496.3778381, -563.9638062, 773.8261719, -1132.8293457, 1055.5841064
4: -282.1394043, 536.9675903, -440.4205627, 833.5879517, -1111.5808105, 973.6504517

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6660630, upper bound: 466.6735969
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6660630, upper bound: 466.6735969
time: 1.20 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -342.2144470, 523.3282471, -542.2124023, 820.7822266, -1160.5133057, 1059.9721680
1: -262.7015686, 480.2991028, -413.5153503, 751.8162842, -1012.8929443, 890.3825073
2: -231.5363464, 499.7409668, -364.1148071, 782.3270874, -1011.0913086, 860.2471924
3: -359.7274170, 493.6423950, -565.4003906, 776.0462036, -1133.1375732, 1054.3975830
4: -280.5881042, 534.0244751, -441.5898743, 835.9288330, -1112.4660645, 971.9169922

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
time: 0.73 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
time: 3.45 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -543.8351440, 822.7476196, -1164.2603760, 1064.3908691
1: -264.1509094, 482.9623718, -414.8011780, 753.8455200, -1016.2635498, 894.2468872
2: -232.8102722, 502.5033569, -365.1686707, 784.4951782, -1014.4320068, 863.9959106
3: -361.7051086, 496.3778381, -567.3734131, 778.2152710, -1137.1593018, 1058.9071045
4: -282.1394043, 536.9675903, -442.9308777, 838.4185791, -1116.3557129, 976.0853271

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -342.2144470, 523.3282471, -544.9468384, 824.4514771, -1164.1820068, 1062.6831055
1: -262.7015686, 480.2991028, -415.5185852, 755.1943970, -1016.2500000, 892.3405762
2: -231.5363464, 499.7409668, -365.8337402, 785.9057617, -1014.6420898, 861.9355469
3: -359.7274170, 493.6423950, -568.2732544, 779.5810547, -1136.6149902, 1057.1790771
4: -280.5881042, 534.0244751, -443.6699829, 839.8901978, -1116.3656006, 973.9229126

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
time: 0.83 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.74 seconds
NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6650940, upper bound: 466.6721956
NS_B1_A2_B2_B2_B1_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6650982, upper bound: 466.6722549
NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6650940, upper bound: 466.6721144
NS_B1_A2_B2_B2_B1_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6650982, upper bound: 466.6721725
NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6665040, upper bound: 466.6730520
NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6735842, upper bound: 466.6730633
NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6715705
NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6734956
NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6735422
NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6645750, upper bound: 466.6735422
NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6672118, upper bound: 466.6728779
NS_B2_A1_A2_A2_A2_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6656811, upper bound: 466.6725284
NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6660630, upper bound: 466.6735969
NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6660630, upper bound: 466.6735969
NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6663341, upper bound: 466.6736011
NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109
NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 3.74
Output dim: 0, lower bound: -466.6730478, upper bound: 466.6736109

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -543.7182617, 822.5239868, -331.5548096, 506.7290649, -1044.7496338, 1151.3264160
1: -414.7001038, 753.6457520, -254.5330048, 464.9100952, -876.0460815, 1006.3740234
2: -365.0790100, 784.2901001, -224.4178925, 483.8345947, -845.1493530, 1005.7503052
3: -567.2403564, 778.0164795, -348.5742493, 477.8603516, -1040.1829834, 1123.7537842
4: -442.8242798, 838.2083130, -272.0354614, 517.2217407, -956.1164551, 1105.9412842

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -543.8657837, 822.7984619, -333.9684753, 510.8715210, -1049.0852051, 1154.0864258
1: -414.8256226, 753.8917847, -256.3548279, 468.7024841, -880.0139160, 1008.4877930
2: -365.1906128, 784.5426636, -226.0290375, 487.6890869, -849.1803589, 1007.6564331
3: -567.4060669, 778.2620850, -351.0267639, 481.6855774, -1044.2415771, 1126.5166016
4: -442.9574890, 838.4684448, -273.9320374, 521.1901245, -960.3192749, 1108.1680908

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -544.8336792, 824.2344360, -331.5548096, 506.7290649, -1045.9332275, 1153.1175537
1: -415.4207764, 755.0005493, -254.5330048, 464.9100952, -876.8029175, 1007.8134155
2: -365.7467957, 785.7066650, -224.4178925, 483.8345947, -845.8522339, 1007.2382812
3: -568.1442261, 779.3884888, -348.5742493, 477.8603516, -1041.1907959, 1125.1900635
4: -443.5666809, 839.6863403, -272.0354614, 517.2217407, -956.8973389, 1107.5028076

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -544.9935303, 824.5310059, -333.9684753, 510.8715210, -1050.2813721, 1155.8992920
1: -415.5563965, 755.2663574, -256.3548279, 468.7024841, -880.7808838, 1009.9469604
2: -365.8674927, 785.9797363, -226.0290375, 487.6890869, -849.8922119, 1009.1647949
3: -568.3236084, 779.6537476, -351.0267639, 481.6855774, -1045.2630615, 1127.9727783
4: -443.7108459, 839.9674683, -273.9320374, 521.1901245, -961.1111450, 1109.7500000

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -543.7805176, 822.6475220, -336.1526794, 513.7417603, -1051.8446045, 1156.1760254
1: -414.7551880, 753.7556152, -257.9609375, 471.4176941, -882.6426392, 1009.9561157
2: -365.1278076, 784.4030151, -227.3839417, 490.5706177, -851.9948730, 1008.8846436
3: -567.3126221, 778.1250610, -353.2991333, 484.5562134, -1047.0057373, 1128.6301270
4: -442.8818665, 838.3232422, -275.5871277, 524.3247681, -963.3658447, 1109.6682129

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -543.9254761, 822.9164429, -337.9484253, 516.7664185, -1055.0568848, 1158.3076172
1: -414.8783569, 753.9967651, -259.3567810, 474.1774902, -885.5709839, 1011.6334229
2: -365.2373352, 784.6504517, -228.6060791, 493.3951416, -854.9731445, 1010.3955688
3: -567.4754028, 778.3658447, -355.1456299, 487.3579407, -1050.0203857, 1130.7849121
4: -443.0124817, 838.5782471, -277.0073242, 527.2607422, -966.4882202, 1111.4152832

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -544.8954468, 824.3577271, -336.1526794, 513.7417603, -1053.0278320, 1157.9666748
1: -415.4754944, 755.1100464, -257.9609375, 471.4176941, -883.3991089, 1011.3950806
2: -365.7953796, 785.8192749, -227.3839417, 490.5706177, -852.6974487, 1010.3722534
3: -568.2162476, 779.4967041, -353.2991333, 484.5562134, -1048.0130615, 1130.0657959
4: -443.6241455, 839.8007812, -275.5871277, 524.3247681, -964.1466064, 1111.2290039

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -545.0524902, 824.6482544, -337.9484253, 516.7664185, -1056.2525635, 1160.1198730
1: -415.6084900, 755.3706055, -259.3567810, 474.1774902, -886.3374634, 1013.0917358
2: -365.9138489, 786.0868530, -228.6060791, 493.3951416, -855.6846313, 1011.9031982
3: -568.3922729, 779.7566528, -355.1456299, 487.3579407, -1051.0415039, 1132.2398682
4: -443.7653809, 840.0763550, -277.0073242, 527.2607422, -967.2796631, 1112.9963379

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_B2_B2_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -344.0790100, 526.2192993, -463.8714294, 699.8893433, -1040.9073486, 983.5418091
1: -264.1509094, 482.9623718, -353.1831665, 640.5910034, -902.6372070, 832.5026245
2: -232.8102722, 502.5033569, -311.0032043, 667.1951294, -896.7804565, 809.6564941
3: -361.7051086, 496.3778381, -483.0985718, 662.0110474, -1020.6475220, 974.1846313
4: -282.1394043, 536.9675903, -377.5960999, 713.3724365, -990.9194336, 910.6546021

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6734956
time: 0.90 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668152, upper bound: 466.6734956
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -336.1526794, 513.7417603, -472.6361389, 712.8795776, -1046.0721436, 979.9461060
1: -257.9609375, 471.4176941, -359.7526245, 652.3817139, -908.3306885, 827.5657349
2: -227.3839417, 490.5706177, -316.7988281, 679.4606934, -903.7128906, 803.5549927
3: -353.2991333, 484.5562134, -492.0852051, 674.1552124, -1024.4517822, 971.4730225
4: -275.5871277, 524.3247681, -384.5737915, 726.5083008, -997.5927734, 905.0281372

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## BFS NS instance: NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -337.9484253, 516.7664185, -472.7756958, 713.1390381, -1048.1948242, 983.1531372
1: -259.3567810, 474.1774902, -359.8715820, 652.6199951, -910.0053711, 830.4899292
2: -228.6060791, 493.3951416, -316.9048767, 679.7048340, -905.2205200, 806.5299072
3: -355.1456299, 487.3579407, -492.2402344, 674.3920288, -1026.6021729, 974.4803467
4: -277.0073242, 527.2607422, -384.6999512, 726.7593994, -999.3357544, 908.1463013

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A2_A2_A2_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -336.1526794, 513.7417603, -540.5137939, 818.0545044, -1151.5681152, 1048.5942383
1: -257.9609375, 471.4176941, -412.3328552, 749.5284424, -1005.7389526, 880.2668457
2: -227.3839417, 490.5706177, -363.0451660, 779.9612427, -1004.4613647, 849.9437256
3: -353.2991333, 484.5562134, -563.8985596, 773.7288208, -1124.2930908, 1043.6779785
4: -275.5871277, 524.3247681, -440.3680725, 833.4848633, -1104.8856201, 960.9274902

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -337.9484253, 516.7664185, -540.6554565, 818.3171387, -1153.6936035, 1051.8029785
1: -259.3567810, 474.1774902, -412.4531860, 749.7640381, -1007.4107056, 883.1923218
2: -228.6060791, 493.3951416, -363.1522217, 780.2032471, -1005.9667358, 852.9194336
3: -355.1456299, 487.3579407, -564.0576172, 773.9639893, -1126.4417725, 1046.6889648
4: -277.0073242, 527.2607422, -440.4957886, 833.7341309, -1106.6267090, 964.0469360

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -336.1526794, 513.7417603, -542.1428223, 820.6491089, -1154.2597656, 1050.2989502
1: -257.9609375, 471.4176941, -413.4552002, 751.6972656, -1008.0037231, 881.4240723
2: -227.3839417, 490.5706177, -364.0613708, 782.2052002, -1006.7861938, 850.9941406
3: -353.2991333, 484.5562134, -565.3211060, 775.9279785, -1126.5548096, 1045.2094727
4: -275.5871277, 524.3247681, -441.5264282, 835.8037109, -1107.2937012, 962.1231079

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -337.9484253, 516.7664185, -542.3090210, 820.9616089, -1156.4335938, 1053.5324707
1: -259.3567810, 474.1774902, -413.5975037, 751.9771118, -1009.7192383, 884.3715210
2: -228.6060791, 493.3951416, -364.1878357, 782.4921875, -1008.3363037, 853.9891968
3: -355.1456299, 487.3579407, -565.5092163, 776.2065430, -1128.7474365, 1048.2492676
4: -277.0073242, 527.2607422, -441.6769714, 836.0988770, -1109.0805664, 965.2653809

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -336.1526794, 513.7417603, -543.7848511, 822.6539917, -1156.1824951, 1051.8491211
1: -257.9609375, 471.4176941, -414.7583618, 753.7616577, -1009.9620972, 882.6458740
2: -227.3839417, 490.5706177, -365.1307373, 784.4090576, -1008.8906860, 851.9978027
3: -353.2991333, 484.5562134, -567.3169556, 778.1312256, -1128.6363525, 1047.0100098
4: -275.5871277, 524.3247681, -442.8855591, 838.3297119, -1109.6748047, 963.3696289

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -337.9484253, 516.7664185, -543.9299927, 822.9234009, -1158.3145752, 1055.0614014
1: -259.3567810, 474.1774902, -414.8816833, 754.0030518, -1011.6396484, 885.5744019
2: -228.6060791, 493.3951416, -365.2403564, 784.6568604, -1010.4019775, 854.9761963
3: -355.1456299, 487.3579407, -567.4798584, 778.3724365, -1130.7913818, 1050.0247803
4: -277.0073242, 527.2607422, -443.0162964, 838.5852051, -1111.4221191, 966.4921265

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -336.1526794, 513.7417603, -544.8999023, 824.3643188, -1157.9731445, 1053.0323486
1: -257.9609375, 471.4176941, -415.4786987, 755.1160889, -1011.4011230, 883.4024048
2: -227.3839417, 490.5706177, -365.7982483, 785.8253784, -1010.3783569, 852.7003784
3: -353.2991333, 484.5562134, -568.2206421, 779.5030518, -1130.0722656, 1048.0174561
4: -275.5871277, 524.3247681, -443.6277466, 839.8073730, -1111.2355957, 964.1503296

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -337.9484253, 516.7664185, -545.0444336, 824.6330566, -1160.1044922, 1056.2442627
1: -259.3567810, 474.1774902, -415.6015930, 755.3571167, -1013.0781860, 886.3305664
2: -228.6060791, 493.3951416, -365.9075623, 786.0728149, -1011.8892212, 855.6785278
3: -355.1456299, 487.3579407, -568.3829956, 779.7434082, -1132.2266846, 1051.0318604
4: -277.0073242, 527.2607422, -443.7581177, 840.0621948, -1112.9818115, 967.2723999

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_A2_A2_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.40 + 291.99 = 295.39 seconds
