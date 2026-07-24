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
execution time: IAR + RelationalAnalysis = 1.58 + 1.91 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -466.6775864, upper bound: 466.6775864

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743041, upper bound: 466.6748479
time: 0.75 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743005, upper bound: 466.6743005
time: 0.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -466.6743041, upper bound: 466.6748479
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -466.6743005, upper bound: 466.6743005

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -182.0487671, 281.9833984, -200.7323151, 313.5108337, -495.5596008, 482.7156677
1: -140.1936951, 259.6987915, -154.8143463, 288.4224854, -428.6160889, 414.5131226
2: -123.1656647, 269.2485962, -136.0845337, 298.5967407, -421.7623291, 405.3331299
3: -191.1080322, 266.1956787, -210.5117645, 294.8744812, -485.9824829, 476.7074585
4: -148.8809204, 286.0896301, -164.3547821, 316.4496765, -465.3305969, 450.4443665

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742277
time: 0.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742295
time: 0.86 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -511.5330200, 773.8687134, -197.9027405, 309.1196899, -815.0314331, 970.5332642
1: -389.6335754, 708.8753662, -152.6286011, 284.3979187, -670.9335327, 860.5180054
2: -342.8289795, 738.1580200, -134.1536713, 294.3879395, -634.1976318, 870.3645020
3: -532.8462524, 731.9035034, -207.4603577, 290.7559509, -819.4674683, 937.8784790
4: -416.0361328, 788.5063477, -162.0148315, 311.9281616, -725.4478149, 947.5808716

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742295, upper bound: 466.6742989
time: 0.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6742295, upper bound: 466.6743005
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.01 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742277
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -466.6742277, upper bound: 466.6742295
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -466.6742295, upper bound: 466.6742989
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 0, lower bound: -466.6742295, upper bound: 466.6743005

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -182.0487671, 281.9833984, -182.0487671, 281.9833984, -464.0321350, 464.0321350
1: -140.1936951, 259.6987915, -140.1936951, 259.6987915, -399.8924255, 399.8924255
2: -123.1656647, 269.2485962, -123.1656647, 269.2485962, -392.4142151, 392.4142151
3: -191.1080322, 266.1956787, -191.1080322, 266.1956787, -457.3036804, 457.3036804
4: -148.8809204, 286.0896301, -148.8809204, 286.0896301, -434.9705200, 434.9705200

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6740737, upper bound: 466.6745173
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6741891, upper bound: 466.6748089
time: 0.81 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -182.0487671, 281.9833984, -504.5384827, 761.9297485, -942.8170776, 780.9023438
1: -140.1936951, 259.6987915, -384.0342407, 698.1107788, -837.3646851, 640.6350708
2: -123.1656647, 269.2485962, -337.7992554, 726.9851685, -848.2468872, 604.0621338
3: -191.1080322, 266.1956787, -525.3290405, 721.0104980, -910.6485596, 787.4302368
4: -148.8809204, 286.0896301, -409.9613037, 776.8063354, -922.7834473, 693.5399780

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6740737, upper bound: 466.6745455
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6741891, upper bound: 466.6748372
time: 1.00 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -504.5384521, 761.9297485, -182.0487671, 281.9833984, -780.9023438, 942.8170776
1: -384.0342407, 698.1108398, -140.1936951, 259.6987915, -640.6350708, 837.3647461
2: -337.7992554, 726.9852295, -123.1656647, 269.2485962, -604.0620728, 848.2469482
3: -525.3290405, 721.0104980, -191.1080322, 266.1956787, -787.4302368, 910.6485596
4: -409.9612732, 776.8063965, -148.8809204, 286.0896301, -693.5400391, 922.7835083

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675874, upper bound: 466.6716479
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675785, upper bound: 466.6675820
time: 0.99 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -535.4615479, 813.5322266, -535.4615479, 813.5322266, -1341.4660645, 1341.4660645
1: -408.6176147, 744.8612061, -408.6176147, 744.8612061, -1148.9763184, 1148.9763184
2: -359.8183594, 775.3198853, -359.8183594, 775.3198853, -1129.8016357, 1129.8016357
3: -558.2575073, 768.5037231, -558.2575073, 768.5037231, -1320.4251709, 1320.4252930
4: -436.6341858, 827.5334473, -436.6341858, 827.5334473, -1258.1224365, 1258.1224365

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675874, upper bound: 466.6716479
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675785, upper bound: 466.6675858
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.28 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6740737, upper bound: 466.6745173
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6741891, upper bound: 466.6748089
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6740737, upper bound: 466.6745455
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6741891, upper bound: 466.6748372
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6675874, upper bound: 466.6716479
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6675785, upper bound: 466.6675820
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6675874, upper bound: 466.6716479
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.28
Output dim: 0, lower bound: -466.6675785, upper bound: 466.6675858

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -166.8329163, 257.2044067, -182.0487671, 281.9833984, -448.8162842, 439.2531128
1: -128.4387665, 237.0249329, -140.1936951, 259.6987915, -388.1375732, 377.2185974
2: -112.7712250, 245.9585419, -123.1656647, 269.2485962, -382.0197449, 369.1241150
3: -175.2800598, 243.2385101, -191.1080322, 266.1956787, -441.4757385, 434.3465271
4: -136.3607941, 261.6822510, -148.8809204, 286.0896301, -422.4504395, 410.5631104

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744023, upper bound: 466.6744023
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744023, upper bound: 466.6745173
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -330.9225159, 504.3214417, -179.3473053, 277.9125671, -607.9680786, 683.6687622
1: -253.9430389, 462.9837646, -138.1242218, 255.9330597, -509.4100647, 601.1078491
2: -223.7483826, 481.8748169, -121.3382111, 265.3351135, -488.4559631, 602.9080200
3: -348.0227661, 476.2661743, -188.2722015, 262.3227844, -609.7725830, 664.4827271
4: -271.2294922, 515.4243774, -146.6701202, 281.8937988, -552.3688965, 661.4875488

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6748063
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6748012
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -166.8329163, 257.2044067, -497.9891357, 751.4299316, -917.1582642, 749.6268311
1: -128.4387665, 237.0249329, -378.8989868, 688.4506226, -815.9945068, 612.8386230
2: -112.7712250, 245.9585419, -333.2116394, 716.9809570, -827.8884888, 576.2048340
3: -175.2800598, 243.2385101, -518.3778687, 711.1323853, -884.9600830, 757.5725708
4: -136.3607941, 261.6822510, -404.3879395, 766.2008667, -899.6974487, 663.5725098

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6718507, upper bound: 466.6675055
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6673884, upper bound: 466.6674864
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -332.3796387, 506.7579956, -507.2649536, 769.2489624, -1098.5897217, 1007.8558350
1: -255.1003723, 465.1946716, -387.0245056, 704.2310181, -957.3145142, 848.7061768
2: -224.7740326, 484.1318054, -340.6900330, 733.1356201, -954.8554688, 821.0797729
3: -349.5700684, 478.4971619, -529.1633301, 726.9841919, -1073.5570068, 1002.5888062
4: -272.4687195, 517.7890015, -413.4290466, 783.0762939, -1051.1717529, 927.3840942

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6716844, upper bound: 466.6682722
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675394, upper bound: 466.6682551
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.08 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6744023, upper bound: 466.6744023
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6744023, upper bound: 466.6745173
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6748063
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6748012
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6718507, upper bound: 466.6675055
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6673884, upper bound: 466.6674864
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6716844, upper bound: 466.6682722
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -466.6675394, upper bound: 466.6682551

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -166.8329163, 257.2044067, -166.8329163, 257.2044067, -424.0372925, 424.0372925
1: -128.4387665, 237.0249329, -128.4387665, 237.0249329, -365.4636841, 365.4636841
2: -112.7712250, 245.9585419, -112.7712250, 245.9585419, -358.7296448, 358.7296448
3: -175.2800598, 243.2385101, -175.2800598, 243.2385101, -418.5185547, 418.5185547
4: -136.3607941, 261.6822510, -136.3607941, 261.6822510, -398.0430298, 398.0430298

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743919, upper bound: 466.6706536
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743988, upper bound: 466.6743988
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -166.8329163, 257.2044067, -330.2861938, 503.3094177, -670.1423340, 586.6295776
1: -128.4387665, 237.0249329, -253.4480591, 462.0596008, -590.4981079, 489.9877319
2: -112.7712250, 245.9585419, -223.3118744, 480.9268799, -593.3874512, 468.6359863
3: -175.2800598, 243.2385101, -347.3569946, 475.3212280, -650.5440674, 590.0402222
4: -136.3607941, 261.6822510, -270.6995850, 514.4227905, -650.1651611, 531.6194458

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743919, upper bound: 466.6707642
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6743988, upper bound: 466.6745094
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -325.6264954, 495.7525024, -157.1725006, 242.5126801, -567.3248901, 652.9249268
1: -249.7842865, 455.0653687, -120.8491440, 223.2326355, -472.5379333, 575.9143677
2: -220.1055145, 473.6967468, -106.1992035, 231.4929810, -450.9636536, 579.5744629
3: -342.3801270, 468.1998901, -164.6565704, 228.9463043, -570.6659546, 632.7210693
4: -266.8367920, 506.8009033, -128.3836670, 246.1010590, -512.1408691, 634.5290527

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6707642
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6748012
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -324.6787720, 494.8265686, -224.5028381, 344.6268616, -668.3245850, 719.3292236
1: -249.1033173, 454.2629700, -172.9687805, 317.8596802, -566.3663330, 627.2033081
2: -219.4868927, 472.8288269, -152.0457916, 329.6991577, -548.4648438, 624.4863892
3: -341.4014282, 467.2665710, -236.2754517, 326.5783691, -667.1920776, 703.3142700
4: -266.0867310, 505.7122498, -183.9519806, 351.4676208, -616.6423950, 688.9204712

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6707642
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6748012
time: 0.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.74 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6743919, upper bound: 466.6706536
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6743988, upper bound: 466.6743988
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6743919, upper bound: 466.6707642
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6743988, upper bound: 466.6745094
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6707642
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6748012
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6707642
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 0, lower bound: -466.6748012, upper bound: 466.6748012

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -144.9818115, 222.3675842, -161.7420959, 249.0589294, -394.0407410, 384.1096802
1: -111.4387207, 204.8170624, -124.4538956, 229.4652405, -340.9039612, 329.2708435
2: -97.8762436, 212.6536255, -109.2800751, 238.1630249, -336.0392761, 321.9336853
3: -152.0532379, 210.3865967, -169.8401947, 235.5360107, -387.5892029, 380.2268066
4: -118.3619308, 226.4538116, -132.1526642, 253.4290771, -371.7910156, 358.6064758

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6704728, upper bound: 466.6694518
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6726218, upper bound: 466.6696054
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -213.3869781, 326.4544373, -162.1923523, 250.2345886, -463.6215820, 488.6467896
1: -164.3702087, 301.2420044, -124.8820496, 230.6037140, -394.9738770, 426.1240540
2: -144.4453888, 312.6638184, -109.6304245, 239.2702942, -383.7156067, 422.2942505
3: -224.7991943, 309.7706604, -170.4068756, 236.5925140, -461.3917236, 480.1774597
4: -174.8038025, 333.6871033, -132.5446777, 254.5211945, -429.3249207, 466.2317810

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6704728, upper bound: 466.6724673
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6726218, upper bound: 466.6726218
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -144.9818115, 222.3675842, -324.9366760, 494.6555176, -639.6370850, 546.4751587
1: -111.4387207, 204.8170624, -249.2475739, 454.0634155, -565.5020752, 453.5646973
2: -97.8762436, 212.6536255, -219.6322632, 472.6686401, -570.2188721, 431.6265869
3: -152.0532379, 210.3865967, -341.6585388, 467.1749878, -619.0860596, 551.3772583
4: -118.3619308, 226.4538116, -266.2622986, 505.7149048, -623.4121094, 491.9090271

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6706461, upper bound: 466.6707642
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6706461, upper bound: 466.6707642
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -213.3869781, 326.4544373, -324.0670776, 493.8548279, -707.2418213, 649.4817505
1: -164.3702087, 301.2420044, -248.6266785, 453.3746948, -617.7039185, 549.2570801
2: -144.4453888, 312.6638184, -219.0668335, 471.9179382, -615.9617920, 530.9923706
3: -224.7991943, 309.7706604, -340.7607117, 466.3587341, -690.9070435, 649.7512817
4: -174.8038025, 333.6871033, -265.5767517, 504.7498169, -678.7987671, 598.3325195

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6706536, upper bound: 466.6745093
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6706536, upper bound: 466.6745094
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -350.0743103, 535.4023438, -157.1725006, 242.5126801, -591.8405762, 692.5748291
1: -268.7593689, 491.4406738, -120.8491440, 223.2326355, -491.5816345, 612.2897949
2: -236.8876190, 511.3438416, -106.1992035, 231.4929810, -467.7720947, 617.2875977
3: -368.0234070, 505.1104736, -164.6565704, 228.9463043, -596.4931641, 669.7670288
4: -287.1268311, 546.4237671, -128.3836670, 246.1010590, -532.4868774, 674.2764893

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746819
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706550
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -306.7213745, 465.1373291, -224.5028381, 344.6268616, -650.2800293, 689.6401367
1: -234.9263611, 426.8524780, -172.9687805, 317.8596802, -552.0957031, 599.7620239
2: -207.0968933, 444.5511780, -152.0457916, 329.6991577, -536.0320435, 596.1534424
3: -322.1262207, 439.4514160, -236.2754517, 326.5783691, -647.7136230, 675.3790894
4: -251.1522980, 476.0157166, -183.9519806, 351.4676208, -601.6226807, 659.0774536

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706536
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706536
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -350.0931091, 535.4331055, -224.5028381, 344.6268616, -693.7863159, 759.9359131
1: -268.7741089, 491.4685364, -172.9687805, 317.8596802, -586.0666504, 664.4372559
2: -236.9006500, 511.3722534, -152.0457916, 329.6991577, -565.8881226, 663.0823364
3: -368.0430908, 505.1387939, -236.2754517, 326.5783691, -693.9511108, 741.2708740
4: -287.1424866, 546.4535522, -183.9519806, 351.4676208, -637.7206421, 729.7247925

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746902
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6747425
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.26 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6704728, upper bound: 466.6694518
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6726218, upper bound: 466.6696054
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6704728, upper bound: 466.6724673
NS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6726218, upper bound: 466.6726218
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6706461, upper bound: 466.6707642
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6706461, upper bound: 466.6707642
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6706536, upper bound: 466.6745093
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6706536, upper bound: 466.6745094
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746819
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706550
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706536
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6706536
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746902
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6747425

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -213.3869781, 326.4544373, -306.0772095, 464.1163330, -677.5032349, 631.4044800
1: -164.3702087, 301.2420044, -234.4256439, 425.9199219, -590.2176514, 534.9624023
2: -144.4453888, 312.6638184, -206.6550446, 443.5938110, -587.5822144, 518.5377197
3: -224.7991943, 309.7706604, -321.4535828, 438.4967651, -662.9249268, 630.2404785
4: -174.8038025, 333.6871033, -250.6154633, 475.0041809, -648.9064331, 583.2858276

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6697531, upper bound: 466.6706442
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6699067, upper bound: 466.6728263
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -213.3869781, 326.4544373, -349.4963379, 534.4771729, -747.8641357, 674.9594116
1: -164.3702087, 301.2420044, -268.3096619, 490.5979919, -654.9681396, 568.9700928
2: -144.4453888, 312.6638184, -236.4920654, 510.4835815, -654.5804443, 548.4273071
3: -224.7991943, 309.7706604, -367.4207458, 504.2543030, -728.8872070, 676.5292358
4: -174.8038025, 333.6871033, -286.6494751, 545.5153198, -719.6281738, 619.4281006

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6697531, upper bound: 466.6707009
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6699067, upper bound: 466.6728266
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -349.4615479, 534.4210205, -144.9818115, 222.3675842, -571.0689697, 679.4026489
1: -268.2825928, 490.5468750, -111.4387207, 204.8170624, -472.6690979, 601.9855957
2: -236.4681854, 510.4314270, -97.8762436, 212.6536255, -448.4891663, 608.0484009
3: -367.3844910, 504.2023621, -152.0532379, 210.3865967, -577.2878418, 656.2548218
4: -286.6206970, 545.4604492, -118.3619308, 226.4538116, -512.3237915, 663.2827759

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6707329, upper bound: 466.6682701
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746819
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -349.4963379, 534.4771729, -213.3869781, 326.4544373, -674.9594116, 747.8641357
1: -268.3096619, 490.5979919, -164.3702087, 301.2420044, -568.9700928, 654.9681396
2: -236.4920654, 510.4835815, -144.4453888, 312.6638184, -548.4273071, 654.5804443
3: -367.4207458, 504.2543030, -224.7991943, 309.7706604, -676.5292358, 728.8872070
4: -286.6494751, 545.5153198, -174.8038025, 333.6871033, -619.4281006, 719.6281738

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744002, upper bound: 466.6682733
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744913, upper bound: 466.6746902
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -351.4529114, 537.6587524, -369.9630737, 563.2228394, -913.3548584, 906.1937256
1: -269.8427429, 493.4951172, -284.3480530, 517.5446167, -786.5289307, 776.9282837
2: -237.8432617, 513.4338379, -250.6109924, 538.3909912, -774.7983398, 762.6603394
3: -369.4709167, 507.1825867, -389.8164673, 532.4390869, -900.5410156, 895.5061035
4: -288.2787476, 548.6220703, -303.6853943, 576.3713989, -862.5482788, 850.3273315

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744002, upper bound: 466.6682980
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6744913, upper bound: 466.6747425
time: 0.83 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.04 seconds
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6697531, upper bound: 466.6706442
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6699067, upper bound: 466.6728263
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6697531, upper bound: 466.6707009
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6699067, upper bound: 466.6728266
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6707329, upper bound: 466.6682701
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6707642, upper bound: 466.6746819
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6744002, upper bound: 466.6682733
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6744913, upper bound: 466.6746902
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6744002, upper bound: 466.6682980
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.04
Output dim: 0, lower bound: -466.6744913, upper bound: 466.6747425

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -344.7519531, 527.1301880, -143.7982635, 220.5568542, -564.6190796, 670.9284668
1: -264.6510315, 483.8384705, -110.5295410, 203.1451874, -467.3950195, 594.3679810
2: -233.2481232, 503.4582825, -97.0744781, 210.9189606, -443.5711365, 600.2977905
3: -362.4176331, 497.3184204, -150.8090057, 208.6671600, -570.6284180, 648.1274414
4: -282.6692200, 538.0133057, -117.3892670, 224.6018219, -506.5539856, 654.8863525

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6704741, upper bound: 466.6734414
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -340.0753174, 520.0343628, -213.3869781, 326.4544373, -665.4833374, 733.4213257
1: -261.1205750, 477.2576904, -164.3702087, 301.2420044, -561.7695312, 641.6279297
2: -230.1963501, 496.6166382, -144.4453888, 312.6638184, -542.1125488, 640.6967773
3: -357.5661926, 490.5380554, -224.7991943, 309.7706604, -666.6577759, 715.1605225
4: -279.0313416, 530.7719727, -174.8038025, 333.6871033, -611.7843628, 704.8488159

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6670797, upper bound: 466.6682235
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6670797, upper bound: 466.6682733
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -344.7842407, 527.1821899, -211.8739471, 324.1532288, -668.0224609, 739.0561523
1: -264.6760864, 483.8857117, -163.2185211, 299.1281738, -563.2512817, 647.1041260
2: -233.2701569, 503.5066528, -143.4308929, 310.4543457, -543.0342407, 646.6132202
3: -362.4510803, 497.3665771, -223.2204590, 307.5971680, -669.4118652, 720.4492188
4: -282.6958618, 538.0641479, -173.5727692, 331.3265991, -613.1481323, 710.9677734

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6671110, upper bound: 466.6746403
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6671110, upper bound: 466.6746902
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -342.0317688, 523.2263184, -369.9630737, 563.2228394, -903.8785400, 891.7593384
1: -262.6574097, 480.1643066, -284.3480530, 517.5446167, -779.3319092, 763.5875854
2: -231.5503998, 499.5770874, -250.6109924, 538.3909912, -768.4860229, 748.7871704
3: -359.6212463, 493.4732361, -389.8164673, 532.4390869, -890.6744995, 881.7866211
4: -280.6623230, 533.8886719, -303.6853943, 576.3713989, -854.9063110, 835.5578613

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6683489, upper bound: 466.6682568
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6683489, upper bound: 466.6682980
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -346.8822327, 530.5860596, -368.5664978, 561.0624390, -906.6981201, 897.7522583
1: -266.3182373, 486.9890747, -283.2705994, 515.5494385, -781.0431519, 769.3635864
2: -234.7166290, 506.6669312, -249.6584167, 536.3137207, -769.6368408, 754.9687500
3: -364.6489258, 500.5048523, -388.3397827, 530.3961792, -893.7064209, 887.3860474
4: -284.4408264, 541.3932495, -302.5276184, 574.1552124, -856.5352173, 841.9707031

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683823, upper bound: 466.6747080
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683823, upper bound: 466.6747425
time: 0.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.49 seconds
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6704741, upper bound: 466.6734414
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
NS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6670797, upper bound: 466.6682235
NS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6670797, upper bound: 466.6682733
NS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6671110, upper bound: 466.6746403
NS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6671110, upper bound: 466.6746902
NS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6683489, upper bound: 466.6682568
NS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6683489, upper bound: 466.6682980
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6683823, upper bound: 466.6747080
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.49
Output dim: 0, lower bound: -466.6683823, upper bound: 466.6747425

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -344.7155762, 527.0712891, -136.1027527, 208.4523773, -552.4470215, 663.1740723
1: -264.6226501, 483.7850037, -104.5953522, 191.9609070, -456.1664124, 588.3802490
2: -233.2231140, 503.4038696, -91.8740540, 199.3588257, -431.9730530, 595.0374146
3: -362.3797302, 497.2642517, -142.7608337, 197.2406921, -559.1513062, 640.0250854
4: -282.6391602, 537.9559937, -111.0905228, 212.3972015, -494.3086853, 648.5297852

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -342.8877258, 524.2366943, -140.2535553, 214.4961548, -556.6366577, 664.4901733
1: -263.2009583, 481.1735840, -107.6581421, 197.4458313, -460.2279053, 588.8317261
2: -231.9734802, 500.6941833, -94.5824661, 205.1319885, -436.5028381, 595.0288086
3: -360.4389038, 494.5816650, -146.9893341, 202.9700317, -562.9443359, 641.5709839
4: -281.1170959, 535.0688477, -114.3802948, 218.6256866, -499.0252380, 648.9205322

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -344.7495117, 527.1263428, -205.4151764, 314.1426086, -657.9220581, 732.5413818
1: -264.6490479, 483.8348999, -158.2662048, 289.9013977, -553.9727783, 642.1010742
2: -233.2463989, 503.4548035, -139.0943909, 300.9070129, -533.4382324, 642.2111816
3: -362.4149170, 497.3150330, -216.4837799, 298.1319275, -659.8833618, 713.6301270
4: -282.6672363, 538.0095825, -168.3168182, 321.1996765, -602.9633789, 705.6412354

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664876, upper bound: 466.6739732
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735293
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -344.7894592, 527.1916504, -207.2825470, 316.7658081, -660.6184692, 734.4741821
1: -264.6802063, 483.8938293, -159.6204071, 292.3443604, -556.4539185, 643.5141602
2: -233.2738953, 503.5147705, -140.2551117, 303.4243774, -535.9932861, 643.4215698
3: -362.4564514, 497.3747864, -218.3669281, 300.7195129, -662.4993286, 715.5505981
4: -282.7001648, 538.0725708, -169.7302246, 323.8925476, -605.6800537, 707.0957642

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664876, upper bound: 466.6739945
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735511
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -346.8822327, 530.5860596, -361.2658691, 549.9843140, -895.5809937, 890.3836670
1: -266.3182373, 486.9890747, -277.7124023, 505.3504639, -770.8126221, 763.7859497
2: -234.7166290, 506.6669312, -244.8046417, 525.7219849, -759.0077515, 750.0910645
3: -364.6489258, 500.5048523, -380.7572632, 519.8751831, -883.1513672, 879.7617188
4: -284.4408264, 541.3932495, -296.6480408, 562.8761597, -845.2124023, 836.0590820

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665132, upper bound: 466.6740425
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665092, upper bound: 466.6735935
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -346.8822327, 530.5860596, -365.0284729, 555.4978638, -901.1582642, 894.2682495
1: -266.3182373, 486.9890747, -280.5027466, 510.3914490, -775.9241943, 766.5996704
2: -234.7166290, 506.6669312, -247.1987305, 530.9581909, -764.3175049, 752.5260010
3: -364.6489258, 500.5048523, -384.5589905, 525.1333008, -888.4664307, 883.6292725
4: -284.4408264, 541.3932495, -299.5244446, 568.4353027, -850.8468628, 838.9865723

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665132, upper bound: 466.6740610
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665092, upper bound: 466.6736119
time: 0.90 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.82 seconds
NS_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
NS_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
NS_A1_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
NS_A1_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
NS_A1_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6664876, upper bound: 466.6739732
NS_A1_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735293
NS_A1_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6664876, upper bound: 466.6739945
NS_A1_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735511
NS_A1_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6665132, upper bound: 466.6740425
NS_A1_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6665092, upper bound: 466.6735935
NS_A1_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6665132, upper bound: 466.6740610
NS_A1_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.82
Output dim: 0, lower bound: -466.6665092, upper bound: 466.6736119

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -336.7710571, 514.5537720, -136.1027527, 208.4523773, -544.4436035, 650.6564941
1: -258.4161682, 472.2030029, -104.5953522, 191.9609070, -449.9325867, 576.7983398
2: -227.7807159, 491.4343262, -91.8740540, 199.3588257, -426.5023804, 583.0408325
3: -353.9511414, 485.4043884, -142.7608337, 197.2406921, -550.6928711, 628.1627197
4: -276.0661621, 525.2747803, -111.0905228, 212.3972015, -487.6966248, 635.8220215

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734032
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734414
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -338.4031982, 517.3226929, -136.1027527, 208.4523773, -546.1428223, 653.4254150
1: -259.6837158, 474.7307434, -104.5953522, 191.9609070, -451.2398376, 579.3261108
2: -228.8910675, 494.0166016, -91.8740540, 199.3588257, -427.6531067, 585.6660156
3: -355.6271667, 487.9710999, -142.7608337, 197.2406921, -552.4343872, 630.7319336
4: -277.3522644, 527.9554443, -111.0905228, 212.3972015, -489.0523987, 638.5571289

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734032
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734414
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -336.1995544, 513.7211914, -140.2535553, 214.4961548, -549.8941650, 653.9746094
1: -257.9753723, 471.4348450, -107.6581421, 197.4458313, -454.9773865, 579.0929565
2: -227.3899384, 490.6270752, -94.5824661, 205.1319885, -431.8938599, 584.9375000
3: -353.3404236, 484.6092529, -146.9893341, 202.9700317, -555.8204346, 631.5985718
4: -275.5850525, 524.3966064, -114.3802948, 218.6256866, -493.4556580, 638.2251587

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6666971, upper bound: 466.6729784
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -338.4600830, 517.4150391, -140.2535553, 214.4961548, -552.2156982, 657.6685791
1: -259.7281494, 474.8146362, -107.6581421, 197.4458313, -456.7652588, 582.4725952
2: -228.9301453, 494.1018982, -94.5824661, 205.1319885, -433.4695740, 588.4508057
3: -355.6866150, 488.0559998, -146.9893341, 202.9700317, -558.2239990, 635.0453491
4: -277.3992920, 528.0452271, -114.3802948, 218.6256866, -495.3315430, 641.9219360

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6666971, upper bound: 466.6729788
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -336.8041687, 514.6074219, -205.4151764, 314.1426086, -649.9177856, 720.0224609
1: -258.4419250, 472.2517700, -158.2662048, 289.9013977, -547.7379761, 630.5179443
2: -227.8033752, 491.4841309, -139.0943909, 300.9070129, -527.9668579, 630.2127686
3: -353.9855347, 485.4540710, -216.4837799, 298.1319275, -651.4235840, 701.7504272
4: -276.0935669, 525.3270874, -168.3168182, 321.1996765, -596.3502808, 692.9316406

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663827, upper bound: 466.6738865
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664877, upper bound: 466.6739732
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -338.4467163, 517.3933716, -203.1613312, 310.6652832, -648.1511230, 720.5546875
1: -259.7177429, 474.7949219, -156.5086670, 286.6879883, -545.8445435, 631.3035889
2: -228.9209747, 494.0822449, -137.5496826, 297.5650635, -525.7866821, 631.3092651
3: -355.6726685, 488.0364685, -214.0752563, 294.8289490, -649.8770142, 701.9751587
4: -277.3884583, 528.0244751, -166.4375305, 317.6322021, -594.1502075, 693.8041382

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6663735, upper bound: 466.6734556
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735293
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -336.8419800, 514.6697388, -207.2825470, 316.7658081, -652.6120605, 721.9522705
1: -258.4714966, 472.3078308, -159.6204071, 292.3443604, -550.2177124, 631.9282227
2: -227.8294983, 491.5411987, -140.2551117, 303.4243774, -530.5203857, 631.4202881
3: -354.0248108, 485.5108032, -218.3669281, 300.7195129, -654.0373535, 703.6679688
4: -276.1247559, 525.3870239, -169.7302246, 323.8925476, -599.0652466, 694.3831787

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6735511
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6735511
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -338.4869995, 517.4595337, -204.9296417, 313.1307373, -650.6899414, 722.3890991
1: -259.7491760, 474.8545227, -157.7855988, 288.9863892, -548.1799927, 632.6401367
2: -228.9487152, 494.1428223, -138.6420135, 299.9355774, -528.1925659, 632.4512939
3: -355.7147217, 488.0968323, -215.8556213, 297.2700195, -652.3454590, 703.7919922
4: -277.4216309, 528.0880127, -167.7687531, 320.1684265, -596.7083740, 695.1758423

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733409, upper bound: 466.6735511
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733409, upper bound: 466.6735511
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -361.2658691, 549.9843140, -887.6093750, 877.9182739
1: -260.1408691, 475.4662170, -277.7124023, 505.3504639, -764.6077881, 752.2499390
2: -229.3004150, 494.7570801, -244.8046417, 525.7219849, -753.5632935, 738.1542969
3: -356.2588806, 488.7039185, -380.7572632, 519.8751831, -874.7314453, 867.9421387
4: -277.9010620, 528.7738647, -296.6480408, 562.8761597, -838.6327515, 823.4128418

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664961, upper bound: 466.6735929
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6664961, upper bound: 466.6735932
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -358.9694824, 546.4422607, -885.8711548, 878.5734863
1: -261.4806213, 478.1202393, -275.9182129, 502.0677185, -762.7163696, 753.1577759
2: -230.4738770, 497.4707336, -243.2262421, 522.3103027, -751.3748779, 739.3363037
3: -358.0322571, 491.3998413, -378.2916870, 516.5034180, -873.2081909, 868.2352295
4: -279.2641296, 531.5950317, -294.7286987, 559.2315063, -836.4336548, 824.3776855

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6650961, upper bound: 466.6735930
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6650869, upper bound: 466.6735932
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -365.0284729, 555.4978638, -893.1865845, 881.8027344
1: -260.1408691, 475.4662170, -280.5027466, 510.3914490, -769.7193604, 755.0636597
2: -229.3004150, 494.7570801, -247.1987305, 530.9581909, -758.8730469, 740.5891724
3: -356.2588806, 488.7039185, -384.5589905, 525.1333008, -880.0465088, 871.8096313
4: -277.9010620, 528.7738647, -299.5244446, 568.4353027, -844.2672119, 826.3402710

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6736119
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6736119
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -362.7229004, 551.9584351, -891.4429932, 882.4475708
1: -261.4806213, 478.1202393, -278.7070312, 507.1005859, -767.8185425, 755.9691162
2: -230.4738770, 497.4707336, -245.6179199, 527.5468140, -756.6836548, 741.7683716
3: -358.0322571, 491.3998413, -382.1005554, 521.7515259, -878.5124512, 872.1079102
4: -279.2641296, 531.5950317, -297.5983582, 564.7927856, -842.0692139, 827.2985229

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668708, upper bound: 466.6736119
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6668708, upper bound: 466.6736119
time: 0.89 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.41 seconds
NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734032
NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734414
NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734032
NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6683177, upper bound: 466.6734414
NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6666971, upper bound: 466.6729784
NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730351
NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6666971, upper bound: 466.6729788
NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6668852, upper bound: 466.6730358
NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6663827, upper bound: 466.6738865
NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6664877, upper bound: 466.6739732
NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6663735, upper bound: 466.6734556
NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6664795, upper bound: 466.6735293
NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6735511
NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6735511
NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6733409, upper bound: 466.6735511
NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6733409, upper bound: 466.6735511
NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6664961, upper bound: 466.6735929
NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6664961, upper bound: 466.6735932
NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6650961, upper bound: 466.6735930
NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6650869, upper bound: 466.6735932
NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6736119
NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6736119
NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6668708, upper bound: 466.6736119
NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.41
Output dim: 0, lower bound: -466.6668708, upper bound: 466.6736119

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -336.7359619, 514.4969482, -129.6620178, 198.6811981, -534.6275024, 644.1588135
1: -258.3887939, 472.1514587, -99.7394867, 182.9526520, -440.8898621, 571.8909302
2: -227.7565460, 491.3817139, -87.6284943, 189.9633331, -417.0744324, 578.7441406
3: -353.9144897, 485.3520813, -136.1121063, 188.0139771, -541.4313354, 621.4641724
4: -276.0370178, 525.2193604, -105.9635010, 202.4084320, -477.6795044, 630.6487427

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675557, upper bound: 466.6684210
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6696366, upper bound: 466.6739982
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -336.7258606, 514.4827881, -133.8627472, 204.6491852, -540.5761108, 648.3455200
1: -258.3809204, 472.1380920, -102.8275452, 188.4015656, -446.3457642, 574.9656372
2: -227.7496338, 491.3676453, -90.3179016, 195.7451019, -422.8709717, 581.4249878
3: -353.9035339, 485.3381653, -140.4121552, 193.6235352, -547.0382080, 625.7487793
4: -276.0284119, 525.2038574, -109.1828995, 208.6992950, -483.9660645, 633.8481445

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6675557, upper bound: 466.6684335
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6696366, upper bound: 466.6740360
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -338.3663330, 517.2631226, -129.6620178, 198.6811981, -536.3250732, 646.9251099
1: -259.6549377, 474.6766052, -99.7394867, 182.9526520, -442.1957397, 574.4160767
2: -228.8657532, 493.9614258, -87.6284943, 189.9633331, -418.2240295, 581.3667603
3: -355.5886230, 487.9161072, -136.1121063, 188.0139771, -543.1708374, 624.0281982
4: -277.3216858, 527.8972168, -105.9635010, 202.4084320, -479.0338440, 633.3812256

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -338.3679199, 517.2664795, -133.8627472, 204.6491852, -542.2852173, 651.1292114
1: -259.6562500, 474.6794434, -102.8275452, 188.4015656, -447.6606445, 577.5069580
2: -228.8669434, 493.9642029, -90.3179016, 195.7451019, -424.0285950, 584.0643311
3: -355.5903320, 487.9190063, -140.4121552, 193.6235352, -548.7902832, 628.3311157
4: -277.3230286, 527.9000244, -109.1828995, 208.6992950, -485.3302612, 636.5988770

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -335.6144104, 512.8583984, -133.4892426, 203.7693939, -538.5121460, 646.3475952
1: -257.5238953, 470.6403198, -102.4228973, 187.5667114, -444.6209717, 573.0632324
2: -226.9898834, 489.7948303, -89.9963074, 194.9285889, -421.2653198, 579.5029907
3: -352.7155151, 483.7887268, -139.8974304, 192.8932648, -545.1062012, 623.6741943
4: -275.0937500, 523.4939575, -108.8360214, 207.8688660, -482.1836243, 631.7717896

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6665260, upper bound: 466.6683714
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665260, upper bound: 466.6729913
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -330.9890442, 505.7389526, -139.9247589, 213.8137512, -543.9768066, 645.6636963
1: -253.9420013, 464.0575562, -107.4039612, 196.8073578, -450.3079834, 571.4614258
2: -223.8800659, 482.9802246, -94.4265900, 204.3915710, -427.6899719, 577.1481934
3: -347.8469543, 477.0122986, -146.6233521, 202.3680725, -549.7349243, 623.6356201
4: -271.3430481, 516.2678223, -114.1773911, 217.9888306, -488.5874634, 629.8574219

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6667137, upper bound: 466.6683985
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6667137, upper bound: 466.6730480
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -338.4289856, 517.3648071, -133.4892426, 203.7693939, -541.3818970, 650.8540649
1: -259.7038574, 474.7688904, -102.4228973, 187.5667114, -446.8318176, 577.1917725
2: -228.9088440, 494.0554504, -89.9963074, 194.9285889, -423.2153320, 583.7980957
3: -355.6541138, 488.0097046, -139.8974304, 192.8932648, -548.0944824, 627.9071045
4: -277.3735962, 527.9963989, -108.8360214, 207.8688660, -484.5178528, 636.3154297

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -332.6208191, 508.5025330, -139.9247589, 213.8137512, -545.6753540, 648.4273071
1: -255.2019348, 466.5816650, -107.4039612, 196.8073578, -451.6080017, 573.9855957
2: -224.9842682, 485.5570984, -94.4265900, 204.3915710, -428.8345642, 579.7676392
3: -349.5196838, 479.5792847, -146.6233521, 202.3680725, -551.4729614, 626.2025757
4: -272.6220398, 518.9417114, -114.1773911, 217.9888306, -489.9353943, 632.5912476

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -336.7737732, 514.5582886, -198.7956543, 303.7096252, -639.3915405, 713.3539429
1: -258.4182129, 472.2070923, -153.1651306, 280.2986755, -538.0991211, 625.3721924
2: -227.7824860, 491.4385681, -134.6109009, 291.0139465, -518.0322266, 625.6723633
3: -353.9538574, 485.4086914, -209.5938721, 288.2973633, -641.5535889, 694.7933960
4: -276.0683594, 525.2791138, -162.8830566, 310.7561340, -585.8641357, 687.4422607

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6660317, upper bound: 466.6683491
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6660317, upper bound: 466.6738865
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -330.9866943, 505.7351379, -209.6241760, 320.5453186, -650.5118408, 715.3592529
1: -253.9401398, 464.0539856, -161.6215363, 295.7377930, -549.0812988, 625.6702881
2: -223.8783875, 482.9768066, -142.1245575, 307.0101013, -530.1535034, 624.7075195
3: -347.8444214, 477.0089111, -221.1098480, 304.1482849, -651.2998657, 697.9160767
4: -271.3410950, 516.2641602, -171.9756622, 327.9118042, -598.2930298, 687.4753418

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6660471, upper bound: 466.6684016
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6660471, upper bound: 466.6739732
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -338.4151917, 517.3425293, -196.5056458, 300.1711121, -637.5615234, 713.8481445
1: -259.6932068, 474.7486877, -151.3773346, 277.0303650, -536.1488647, 626.1259766
2: -228.8993988, 494.0350342, -133.0401001, 287.6149597, -515.7933350, 626.7402344
3: -355.6398621, 487.9895630, -207.1446991, 284.9385986, -639.9493408, 694.9741211
4: -277.3623352, 527.9747314, -160.9717712, 307.1287231, -583.6018066, 688.2802734

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -332.6070251, 508.4796448, -207.3542023, 317.0279236, -648.6823730, 715.8338013
1: -255.1910553, 466.5612183, -159.8469543, 292.4850464, -547.1243286, 626.4080200
2: -224.9747162, 485.5364990, -140.5640869, 303.6273804, -527.9124146, 625.7502441
3: -349.5052490, 479.5588074, -218.6780548, 300.8092346, -649.6878662, 698.0848999
4: -272.6107178, 518.9199829, -170.0777283, 324.3037720, -596.0276489, 688.2953491

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -336.8070068, 514.6128540, -200.1395874, 305.4555969, -641.2296753, 714.7523804
1: -258.4441528, 472.2563171, -154.0663910, 281.9138184, -539.7282104, 626.3150024
2: -227.8054047, 491.4886780, -135.3819885, 292.6980591, -519.7409668, 626.4802246
3: -353.9884033, 485.4585266, -210.8938599, 290.0405273, -643.2990112, 696.1040649
4: -276.0957336, 525.3317871, -163.8255463, 312.5639954, -587.6749878, 688.4054565

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674961, upper bound: 466.6684114
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6739945
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -336.9133911, 514.7857666, -199.1943817, 304.0708923, -639.9442749, 713.9801025
1: -258.5271912, 472.4129944, -153.2037354, 280.5128174, -538.4376221, 625.6087036
2: -227.8786163, 491.6483459, -134.6353912, 291.2032776, -518.3374634, 625.8896484
3: -354.0992737, 485.6174927, -209.6293640, 288.6221008, -642.0131226, 695.0205688
4: -276.1837769, 525.4998169, -162.9030151, 310.9204407, -586.1296387, 687.6443481

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674961, upper bound: 466.6684114
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6733366, upper bound: 466.6739945
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -338.4496765, 517.3988647, -200.1395874, 305.4555969, -642.9395142, 717.5384521
1: -259.7200317, 474.7995605, -154.0663910, 281.9138184, -541.0438843, 628.8659668
2: -228.9230652, 494.0868530, -135.3819885, 292.6980591, -520.8990479, 629.1212769
3: -355.6757812, 488.0411377, -210.8938599, 290.0405273, -645.0517578, 698.7348022
4: -277.3907471, 528.0291748, -163.8255463, 312.5639954, -589.0395508, 691.1573486

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -338.5803833, 517.6115112, -199.1943817, 304.0708923, -641.6784058, 716.8059082
1: -259.8222351, 474.9922791, -153.2037354, 280.5128174, -539.7724609, 628.1960449
2: -229.0129395, 494.2831116, -134.6353912, 291.2032776, -519.5122070, 628.5674438
3: -355.8123474, 488.2364807, -209.6293640, 288.6221008, -643.7919312, 697.6878052
4: -277.4989014, 528.2358398, -162.9030151, 310.9204407, -587.5140991, 690.4349976

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -354.0242920, 538.5703125, -876.1503296, 870.6074219
1: -260.1408691, 475.4662170, -272.0485535, 494.8294373, -754.0432739, 746.5524902
2: -229.3004150, 494.7570801, -239.8364716, 514.8790894, -742.6708984, 733.1528931
3: -356.2588806, 488.7039185, -373.1106567, 509.0793762, -863.8840942, 860.2254639
4: -277.9010620, 528.7738647, -290.6431580, 551.3871460, -827.0867310, 817.3623047

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6660471, upper bound: 466.6684016
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665039, upper bound: 466.6740425
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -353.2916565, 537.9334717, -875.5874023, 869.9606934
1: -260.1408691, 475.4662170, -271.4610596, 494.0386047, -753.3605957, 746.0241699
2: -229.3004150, 494.7570801, -239.3138580, 513.9673462, -741.8629150, 732.6846313
3: -356.2588806, 488.7039185, -372.1318970, 508.2168884, -863.1229248, 859.4009399
4: -277.9010620, 528.7738647, -289.9334717, 550.2749634, -826.1082764, 816.7426758

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6660471, upper bound: 466.6684016
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6665039, upper bound: 466.6740422
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -354.0242920, 538.5703125, -877.9448242, 873.5549927
1: -261.4806213, 478.1202393, -272.0485535, 494.8294373, -755.4232788, 749.2502441
2: -230.4738770, 497.4707336, -239.8364716, 514.8790894, -743.8847656, 735.9093628
3: -358.0322571, 491.3998413, -373.1106567, 509.0793762, -865.7225952, 862.9694214
4: -279.2641296, 531.5950317, -290.6431580, 551.3871460, -828.5191040, 820.2375488

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -353.2916565, 537.9334717, -877.3818970, 872.9082642
1: -261.4806213, 478.1202393, -271.4610596, 494.0386047, -754.7406006, 748.7219238
2: -230.4738770, 497.4707336, -239.3138580, 513.9673462, -743.0768433, 735.4411011
3: -358.0322571, 491.3998413, -372.1318970, 508.2168884, -864.9613647, 862.1448975
4: -279.2641296, 531.5950317, -289.9334717, 550.2749634, -827.5406494, 819.6179199

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -357.6283875, 543.8073120, -881.4742432, 874.3369141
1: -260.1408691, 475.4662170, -274.7325439, 499.6350403, -758.9201660, 749.2600098
2: -229.3004150, 494.7570801, -242.1376190, 519.8555298, -747.7215576, 735.4949341
3: -356.2588806, 488.7039185, -376.7447510, 514.1080322, -868.9697266, 863.9258423
4: -277.9010620, 528.7738647, -293.4135132, 556.6730347, -832.4491577, 820.1829834

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674961, upper bound: 466.6684114
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6740610
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -338.9692078, 518.1342773, -356.3973694, 542.4803467, -880.1759033, 873.1836548
1: -260.1408691, 475.4662170, -273.7489319, 498.1575012, -757.5435181, 748.3328247
2: -229.3004150, 494.7570801, -241.2592163, 518.2778931, -746.2401733, 734.6713867
3: -356.2588806, 488.7039185, -375.2716980, 512.5333862, -867.4942017, 862.6003418
4: -277.9010620, 528.7738647, -292.2649536, 554.8469849, -830.7552490, 819.1295166

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -466.6674961, upper bound: 466.6684114
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -466.6736492, upper bound: 466.6740610
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -357.6283875, 543.8073120, -883.2687988, 877.2844849
1: -261.4806213, 478.1202393, -274.7325439, 499.6350403, -760.3001709, 751.9577637
2: -230.4738770, 497.4707336, -242.1376190, 519.8555298, -748.9354248, 738.2514038
3: -358.0322571, 491.3998413, -376.7447510, 514.1080322, -870.8081665, 866.6697998
4: -279.2641296, 531.5950317, -293.4135132, 556.6730347, -833.8815308, 823.0581665

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -340.6958923, 521.0419922, -356.3973694, 542.4803467, -881.9703979, 876.1311035
1: -261.4806213, 478.1202393, -273.7489319, 498.1575012, -758.9235840, 751.0305176
2: -230.4738770, 497.4707336, -241.2592163, 518.2778931, -747.4541016, 737.4278564
3: -358.0322571, 491.3998413, -375.2716980, 512.5333862, -869.3326416, 865.3442993
4: -279.2641296, 531.5950317, -292.2649536, 554.8469849, -832.1875610, 822.0048218

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.49 + 265.42 = 268.91 seconds
