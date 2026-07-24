## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0302838


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042060, 0.0066431, -0.0042060, 0.0066431, -0.0107326, 0.0107326)
1: (0.0007764, 0.0081940, 0.0007764, 0.0081940, -0.0074176, 0.0074176)
2: (0.0056167, 0.0151711, 0.0056167, 0.0151711, -0.0095544, 0.0095544)
3: (-0.0036217, 0.0059786, -0.0036217, 0.0059786, -0.0096004, 0.0096004)
4: (-0.0167712, -0.0001116, -0.0167712, -0.0001116, -0.0166596, 0.0166596)
5: (0.0012416, 0.0087179, 0.0012416, 0.0087179, -0.0074762, 0.0074762)
6: (-0.0055534, 0.0482379, -0.0055534, 0.0482379, -0.0537913, 0.0537913)
7: (-0.0173300, 0.0058338, -0.0173300, 0.0058338, -0.0231638, 0.0231638)
8: (0.9498128, 0.9927323, 0.9498128, 0.9927323, -0.0429195, 0.0429195)
9: (-0.0167859, 0.0052511, -0.0167859, 0.0052511, -0.0220371, 0.0220371)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 2.86 = 4.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0356280, upper bound: 0.0356280

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0337967, upper bound: 0.0342015
time: 2.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0343184, upper bound: 0.0343184
time: 2.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.24 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.24
Output dim: 8, lower bound: -0.0337967, upper bound: 0.0342015
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.24
Output dim: 8, lower bound: -0.0343184, upper bound: 0.0343184

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0001082, 0.0065945, -0.0013616, 0.0066181, -0.0063704, 0.0076178
1: 0.0011559, 0.0022750, 0.0009344, 0.0036890, -0.0025331, 0.0013406
2: 0.0107139, 0.0146090, 0.0098678, 0.0149406, -0.0042267, 0.0047412
3: -0.0035996, 0.0003584, -0.0036129, 0.0020388, -0.0056384, 0.0039714
4: -0.0042892, -0.0001402, -0.0078531, -0.0001248, -0.0041645, 0.0077129
5: 0.0042460, 0.0082959, 0.0037320, 0.0085449, -0.0042989, 0.0045640
6: -0.0054535, 0.0101254, -0.0055089, 0.0209459, -0.0263994, 0.0156343
7: -0.0164091, 0.0048705, -0.0169494, 0.0051979, -0.0216069, 0.0218199
8: 0.9767771, 0.9926447, 0.9684792, 0.9926965, -0.0159194, 0.0241656
9: -0.0092107, 0.0044784, -0.0105981, 0.0049334, -0.0141441, 0.0150765

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0327535, upper bound: 0.0326452
time: 1.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0330018, upper bound: 0.0333690
time: 1.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0000700, 0.0067379, -0.0018050, 0.0066146, -0.0063589, 0.0083103
1: 0.0012143, 0.0022957, 0.0010275, 0.0045713, -0.0033570, 0.0012682
2: 0.0106347, 0.0145214, 0.0092207, 0.0148012, -0.0041665, 0.0053007
3: -0.0036816, 0.0002926, -0.0036109, 0.0026529, -0.0063345, 0.0039035
4: -0.0042656, -0.0000514, -0.0095484, -0.0001268, -0.0041388, 0.0094969
5: 0.0041620, 0.0082302, 0.0033920, 0.0084402, -0.0042782, 0.0048383
6: -0.0057867, 0.0100368, -0.0055010, 0.0260570, -0.0318437, 0.0155377
7: -0.0162664, 0.0053243, -0.0167223, 0.0052996, -0.0215660, 0.0220466
8: 0.9771857, 0.9929643, 0.9657608, 0.9926889, -0.0155032, 0.0272036
9: -0.0095008, 0.0043583, -0.0114130, 0.0047421, -0.0142430, 0.0157712

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0332734, upper bound: 0.0327568
time: 2.02 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0334635, upper bound: 0.0334635
time: 1.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.48 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 8, lower bound: -0.0327535, upper bound: 0.0326452
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 8, lower bound: -0.0330018, upper bound: 0.0333690
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 8, lower bound: -0.0332734, upper bound: 0.0327568
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.48
Output dim: 8, lower bound: -0.0334635, upper bound: 0.0334635

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0000007, 0.0065802, 0.0000083, 0.0065847, -0.0060839, 0.0061005
1: 0.0013224, 0.0022730, 0.0013235, 0.0022736, -0.0008790, 0.0008813
2: 0.0107218, 0.0143595, 0.0107194, 0.0143553, -0.0033728, 0.0033636
3: -0.0035914, 0.0001708, -0.0035940, 0.0001665, -0.0034883, 0.0034788
4: -0.0042219, -0.0001490, -0.0042171, -0.0001462, -0.0037661, 0.0037763
5: 0.0042544, 0.0081086, 0.0042517, 0.0081042, -0.0035737, 0.0035640
6: -0.0054203, 0.0098724, -0.0054307, 0.0098547, -0.0141793, 0.0141407
7: -0.0160021, 0.0048252, -0.0159780, 0.0048394, -0.0192584, 0.0193109
8: 0.9779416, 0.9926128, 0.9779586, 0.9926228, -0.0135660, 0.0136030
9: -0.0091817, 0.0041358, -0.0091908, 0.0041204, -0.0123479, 0.0123144

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0307483, upper bound: 0.0296997
time: 1.87 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319985, upper bound: 0.0318800
time: 2.28 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0000130, 0.0065811, -0.0000591, 0.0067015, -0.0063646, 0.0062746
1: 0.0013018, 0.0022731, 0.0012310, 0.0022905, -0.0009887, 0.0010421
2: 0.0107213, 0.0143905, 0.0106547, 0.0144964, -0.0037751, 0.0037357
3: -0.0035920, 0.0001941, -0.0036608, 0.0002738, -0.0038658, 0.0038549
4: -0.0042303, -0.0001484, -0.0042589, -0.0000739, -0.0041564, 0.0041104
5: 0.0042538, 0.0081319, 0.0041833, 0.0082115, -0.0039577, 0.0039487
6: -0.0054224, 0.0099041, -0.0057023, 0.0100115, -0.0154338, 0.0156064
7: -0.0160530, 0.0048281, -0.0162257, 0.0052093, -0.0212623, 0.0210538
8: 0.9777971, 0.9926149, 0.9773024, 0.9928834, -0.0150863, 0.0153125
9: -0.0091836, 0.0041786, -0.0094273, 0.0043240, -0.0135076, 0.0136059

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 124

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0299115, upper bound: 0.0313699
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
time: 1.86 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0000392, 0.0067231, 0.0000684, 0.0065804, -0.0060954, 0.0063796
1: 0.0013280, 0.0022936, 0.0013322, 0.0022730, -0.0008806, 0.0009217
2: 0.0106429, 0.0143382, 0.0107218, 0.0143220, -0.0035271, 0.0033700
3: -0.0036731, 0.0001488, -0.0035915, 0.0001321, -0.0036479, 0.0034854
4: -0.0041980, -0.0000606, -0.0041799, -0.0001489, -0.0037731, 0.0039491
5: 0.0041707, 0.0080861, 0.0042543, 0.0080690, -0.0037371, 0.0035706
6: -0.0057523, 0.0097829, -0.0054206, 0.0097150, -0.0148279, 0.0141673
7: -0.0158801, 0.0052774, -0.0157877, 0.0048256, -0.0192946, 0.0201943
8: 0.9780275, 0.9929313, 0.9780927, 0.9926131, -0.0135915, 0.0142253
9: -0.0094708, 0.0040578, -0.0091820, 0.0039987, -0.0129128, 0.0123375

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312735, upper bound: 0.0298093
time: 1.68 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325017, upper bound: 0.0319725
time: 1.92 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0000246, 0.0067251, -0.0000027, 0.0066983, -0.0063490, 0.0064336
1: 0.0013259, 0.0022939, 0.0013175, 0.0022900, -0.0009057, 0.0009764
2: 0.0106417, 0.0143463, 0.0106565, 0.0143669, -0.0037252, 0.0034662
3: -0.0036743, 0.0001571, -0.0036590, 0.0001764, -0.0038507, 0.0035849
4: -0.0042070, -0.0000593, -0.0042240, -0.0000759, -0.0041311, 0.0039825
5: 0.0041695, 0.0080946, 0.0041852, 0.0081143, -0.0039448, 0.0036726
6: -0.0057570, 0.0098168, -0.0056948, 0.0098803, -0.0149534, 0.0155116
7: -0.0159263, 0.0052838, -0.0160146, 0.0051991, -0.0198454, 0.0212985
8: 0.9779951, 0.9929358, 0.9779071, 0.9928762, -0.0139795, 0.0150288
9: -0.0094750, 0.0040874, -0.0094208, 0.0041463, -0.0136212, 0.0126897

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 124

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0314845, upper bound: 0.0303557
time: 2.02 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0326953, upper bound: 0.0326953
time: 1.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.14 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0307483, upper bound: 0.0296997
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0319985, upper bound: 0.0318800
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0299115, upper bound: 0.0313699
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0322231, upper bound: 0.0326010
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0312735, upper bound: 0.0298093
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0325017, upper bound: 0.0319725
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0314845, upper bound: 0.0303557
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.14
Output dim: 8, lower bound: -0.0326953, upper bound: 0.0326953

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.0001332, 0.0065630, 0.0003087, 0.0065220, -0.0058335, 0.0057403
1: 0.0013415, 0.0022705, 0.0013669, 0.0022645, -0.0008428, 0.0008293
2: 0.0107313, 0.0142862, 0.0107540, 0.0141892, -0.0031737, 0.0032252
3: -0.0035816, 0.0000950, -0.0035581, -0.0000053, -0.0032823, 0.0033356
4: -0.0041398, -0.0001596, -0.0040312, -0.0001851, -0.0036110, 0.0035533
5: 0.0042644, 0.0080310, 0.0042885, 0.0079282, -0.0033626, 0.0034172
6: -0.0053803, 0.0095644, -0.0052849, 0.0091565, -0.0133420, 0.0135586
7: -0.0155826, 0.0047708, -0.0150271, 0.0046408, -0.0184656, 0.0181707
8: 0.9782372, 0.9925745, 0.9786285, 0.9924829, -0.0130075, 0.0127998
9: -0.0091470, 0.0038676, -0.0090638, 0.0035124, -0.0116188, 0.0118074

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305937, upper bound: 0.0295656
time: 1.67 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305937, upper bound: 0.0295540
time: 2.17 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: 0.0000599, 0.0065735, 0.0001564, 0.0065682, -0.0059666, 0.0058552
1: 0.0013310, 0.0022720, 0.0013449, 0.0022712, -0.0008620, 0.0008459
2: 0.0107255, 0.0143267, 0.0107285, 0.0142734, -0.0032372, 0.0032988
3: -0.0035876, 0.0001369, -0.0035846, 0.0000817, -0.0033480, 0.0034118
4: -0.0041852, -0.0001532, -0.0041254, -0.0001564, -0.0036934, 0.0036245
5: 0.0042583, 0.0080740, 0.0042614, 0.0080174, -0.0034300, 0.0034952
6: -0.0054046, 0.0097347, -0.0053924, 0.0095104, -0.0136091, 0.0138681
7: -0.0158146, 0.0048040, -0.0155090, 0.0047872, -0.0188871, 0.0185344
8: 0.9780738, 0.9925979, 0.9782889, 0.9925861, -0.0133045, 0.0130560
9: -0.0091681, 0.0040159, -0.0091574, 0.0038205, -0.0118514, 0.0120769

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318474, upper bound: 0.0317280
time: 1.92 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0318474, upper bound: 0.0317172
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0002826, 0.0065191, 0.0000744, 0.0066842, -0.0059145, 0.0059024
1: 0.0013631, 0.0022641, 0.0013331, 0.0022880, -0.0008545, 0.0008527
2: 0.0107557, 0.0142036, 0.0106643, 0.0143187, -0.0032633, 0.0032700
3: -0.0035565, 0.0000096, -0.0036509, 0.0001286, -0.0033751, 0.0033820
4: -0.0040474, -0.0001869, -0.0041762, -0.0000846, -0.0036612, 0.0036537
5: 0.0042902, 0.0079435, 0.0041934, 0.0080655, -0.0034576, 0.0034647
6: -0.0052781, 0.0092172, -0.0056620, 0.0097010, -0.0137189, 0.0137469
7: -0.0151098, 0.0046316, -0.0157687, 0.0051545, -0.0187222, 0.0186839
8: 0.9785702, 0.9924765, 0.9781061, 0.9928448, -0.0131883, 0.0131614
9: -0.0090579, 0.0035653, -0.0093922, 0.0039866, -0.0119470, 0.0119715

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297705, upper bound: 0.0312144
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0297584, upper bound: 0.0312144
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0001346, 0.0065648, -0.0000008, 0.0066948, -0.0061822, 0.0060375
1: 0.0013417, 0.0022707, 0.0013204, 0.0022895, -0.0008694, 0.0009503
2: 0.0107304, 0.0142854, 0.0106585, 0.0143625, -0.0036321, 0.0033270
3: -0.0035826, 0.0000942, -0.0036570, 0.0001731, -0.0037557, 0.0034409
4: -0.0041390, -0.0001586, -0.0042228, -0.0000781, -0.0040609, 0.0037373
5: 0.0042634, 0.0080302, 0.0041872, 0.0081109, -0.0038475, 0.0035251
6: -0.0053843, 0.0095611, -0.0056866, 0.0098758, -0.0140327, 0.0152478
7: -0.0155782, 0.0047762, -0.0160074, 0.0051880, -0.0190486, 0.0207837
8: 0.9782403, 0.9925783, 0.9779277, 0.9928683, -0.0134183, 0.0146506
9: -0.0091504, 0.0038648, -0.0094137, 0.0041402, -0.0132906, 0.0121802

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320868, upper bound: 0.0324462
time: 1.93 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324462
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0001732, 0.0067060, 0.0003691, 0.0065174, -0.0058430, 0.0060289
1: 0.0013473, 0.0022911, 0.0013756, 0.0022639, -0.0008441, 0.0008710
2: 0.0106523, 0.0142641, 0.0107566, 0.0141558, -0.0033332, 0.0032304
3: -0.0036633, 0.0000721, -0.0035555, -0.0000399, -0.0034474, 0.0033411
4: -0.0041151, -0.0000712, -0.0039938, -0.0001879, -0.0036169, 0.0037320
5: 0.0041807, 0.0080076, 0.0042912, 0.0078928, -0.0035317, 0.0034228
6: -0.0057125, 0.0094714, -0.0052743, 0.0090161, -0.0140129, 0.0135806
7: -0.0154559, 0.0052233, -0.0148359, 0.0046264, -0.0184956, 0.0190844
8: 0.9783264, 0.9928933, 0.9787631, 0.9924728, -0.0130287, 0.0134434
9: -0.0094362, 0.0037866, -0.0090546, 0.0033901, -0.0122031, 0.0118266

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311093, upper bound: 0.0296807
time: 1.96 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311093, upper bound: 0.0296627
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0000961, 0.0067162, 0.0002125, 0.0065637, -0.0059780, 0.0061257
1: 0.0013362, 0.0022926, 0.0013530, 0.0022706, -0.0008637, 0.0008850
2: 0.0106466, 0.0143067, 0.0107309, 0.0142424, -0.0033867, 0.0033051
3: -0.0036692, 0.0001162, -0.0035820, 0.0000497, -0.0035027, 0.0034183
4: -0.0041628, -0.0000648, -0.0040907, -0.0001592, -0.0037005, 0.0037919
5: 0.0041747, 0.0080527, 0.0042640, 0.0079846, -0.0035884, 0.0035019
6: -0.0057364, 0.0096506, -0.0053819, 0.0093801, -0.0142378, 0.0138946
7: -0.0157000, 0.0052558, -0.0153315, 0.0047730, -0.0189232, 0.0193907
8: 0.9781545, 0.9929162, 0.9784140, 0.9925761, -0.0133299, 0.0136592
9: -0.0094570, 0.0039426, -0.0091483, 0.0037071, -0.0123989, 0.0121000

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323401, upper bound: 0.0318163
time: 1.87 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323401, upper bound: 0.0317973
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0001562, 0.0067081, 0.0002932, 0.0066412, -0.0060237, 0.0060837
1: 0.0013449, 0.0022914, 0.0013647, 0.0022818, -0.0008702, 0.0008789
2: 0.0106511, 0.0142735, 0.0106881, 0.0141978, -0.0033635, 0.0033303
3: -0.0036646, 0.0000819, -0.0036263, 0.0000035, -0.0034787, 0.0034444
4: -0.0041256, -0.0000699, -0.0040408, -0.0001113, -0.0037288, 0.0037659
5: 0.0041795, 0.0080175, 0.0042186, 0.0079373, -0.0035638, 0.0035287
6: -0.0057175, 0.0095109, -0.0055620, 0.0091925, -0.0141401, 0.0140007
7: -0.0155098, 0.0052300, -0.0150762, 0.0050183, -0.0190677, 0.0192576
8: 0.9782885, 0.9928980, 0.9785939, 0.9927489, -0.0134317, 0.0135654
9: -0.0094406, 0.0038210, -0.0093052, 0.0035438, -0.0123138, 0.0121924

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0302017
time: 1.74 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0301871
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0000823, 0.0067184, 0.0001421, 0.0066818, -0.0061959, 0.0061799
1: 0.0013342, 0.0022929, 0.0013428, 0.0022876, -0.0008951, 0.0008928
2: 0.0106455, 0.0143144, 0.0106657, 0.0142813, -0.0034167, 0.0034256
3: -0.0036704, 0.0001241, -0.0036495, 0.0000899, -0.0035337, 0.0035429
4: -0.0041713, -0.0000635, -0.0041343, -0.0000861, -0.0038354, 0.0038255
5: 0.0041734, 0.0080609, 0.0041948, 0.0080258, -0.0036202, 0.0036296
6: -0.0057413, 0.0096828, -0.0056564, 0.0095436, -0.0143638, 0.0144011
7: -0.0157438, 0.0052624, -0.0155543, 0.0051468, -0.0196130, 0.0195623
8: 0.9781236, 0.9929209, 0.9782571, 0.9928395, -0.0138158, 0.0137801
9: -0.0094613, 0.0039707, -0.0093874, 0.0038495, -0.0125087, 0.0125411

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325566
time: 2.15 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325398
time: 1.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.51 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0305937, upper bound: 0.0295656
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0305937, upper bound: 0.0295540
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0318474, upper bound: 0.0317280
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0318474, upper bound: 0.0317172
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0297705, upper bound: 0.0312144
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0297584, upper bound: 0.0312144
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0320868, upper bound: 0.0324462
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0320672, upper bound: 0.0324462
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0311093, upper bound: 0.0296807
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0311093, upper bound: 0.0296627
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0323401, upper bound: 0.0318163
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0323401, upper bound: 0.0317973
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0302017
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0313102, upper bound: 0.0301871
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325566
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.51
Output dim: 8, lower bound: -0.0325398, upper bound: 0.0325398

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0001716, 0.0064213, 0.0003236, 0.0064624, -0.0057296, 0.0054118
1: 0.0013471, 0.0022500, 0.0013691, 0.0022559, -0.0008278, 0.0007818
2: 0.0108097, 0.0142650, 0.0107869, 0.0141809, -0.0029920, 0.0031678
3: -0.0035006, 0.0000731, -0.0035241, -0.0000139, -0.0030945, 0.0032762
4: -0.0041161, -0.0002474, -0.0040220, -0.0002219, -0.0035467, 0.0033500
5: 0.0043475, 0.0080085, 0.0043234, 0.0079195, -0.0031702, 0.0033564
6: -0.0050509, 0.0094752, -0.0051465, 0.0091218, -0.0125784, 0.0133172
7: -0.0154611, 0.0043222, -0.0149799, 0.0044524, -0.0181369, 0.0171307
8: 0.9783227, 0.9922585, 0.9786617, 0.9923502, -0.0127760, 0.0120672
9: -0.0088601, 0.0037899, -0.0089433, 0.0034822, -0.0109538, 0.0115972

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299722, upper bound: 0.0290362
time: 1.89 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304171, upper bound: 0.0294130
time: 2.18 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0001008, 0.0063890, 0.0003271, 0.0064348, -0.0057664, 0.0054275
1: 0.0013369, 0.0022453, 0.0013696, 0.0022519, -0.0008331, 0.0007841
2: 0.0108276, 0.0143042, 0.0108022, 0.0141790, -0.0030007, 0.0031881
3: -0.0034821, 0.0001136, -0.0035083, -0.0000159, -0.0031035, 0.0032973
4: -0.0041599, -0.0002674, -0.0040198, -0.0002390, -0.0035695, 0.0033597
5: 0.0043664, 0.0080500, 0.0043395, 0.0079174, -0.0031794, 0.0033779
6: -0.0049758, 0.0096398, -0.0050824, 0.0091137, -0.0126151, 0.0134026
7: -0.0156853, 0.0042199, -0.0149687, 0.0043650, -0.0182532, 0.0171806
8: 0.9781647, 0.9921864, 0.9786696, 0.9922886, -0.0128579, 0.0121024
9: -0.0087947, 0.0039333, -0.0088875, 0.0034751, -0.0109858, 0.0116716

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0299722, upper bound: 0.0290173
time: 1.98 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304171, upper bound: 0.0293975
time: 2.34 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0000981, 0.0064312, 0.0001711, 0.0065115, -0.0058641, 0.0055841
1: 0.0013365, 0.0022514, 0.0013470, 0.0022630, -0.0008472, 0.0008067
2: 0.0108042, 0.0143056, 0.0107598, 0.0142652, -0.0030873, 0.0032421
3: -0.0035063, 0.0001151, -0.0035522, 0.0000733, -0.0031930, 0.0033531
4: -0.0041616, -0.0002412, -0.0041163, -0.0001915, -0.0036300, 0.0034566
5: 0.0043416, 0.0080516, 0.0042946, 0.0080088, -0.0032711, 0.0034352
6: -0.0050740, 0.0096461, -0.0052606, 0.0094762, -0.0129790, 0.0136298
7: -0.0156938, 0.0043536, -0.0154625, 0.0046078, -0.0185626, 0.0176762
8: 0.9781588, 0.9922807, 0.9783218, 0.9924597, -0.0130759, 0.0124515
9: -0.0088802, 0.0039387, -0.0090427, 0.0037908, -0.0113027, 0.0118694

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309755, upper bound: 0.0308888
time: 1.95 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0316807, upper bound: 0.0315587
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0000293, 0.0063991, 0.0001770, 0.0064757, -0.0058856, 0.0055956
1: 0.0013265, 0.0022468, 0.0013479, 0.0022578, -0.0008503, 0.0008084
2: 0.0108220, 0.0143436, 0.0107796, 0.0142620, -0.0030937, 0.0032540
3: -0.0034879, 0.0001544, -0.0035317, 0.0000700, -0.0031996, 0.0033654
4: -0.0042041, -0.0002611, -0.0041127, -0.0002137, -0.0036433, 0.0034638
5: 0.0043605, 0.0080919, 0.0043156, 0.0080054, -0.0032779, 0.0034478
6: -0.0049992, 0.0098058, -0.0051773, 0.0094626, -0.0130057, 0.0136798
7: -0.0159113, 0.0042518, -0.0154440, 0.0044943, -0.0186306, 0.0177126
8: 0.9780056, 0.9922090, 0.9783348, 0.9923797, -0.0131238, 0.0124772
9: -0.0088151, 0.0040778, -0.0089701, 0.0037789, -0.0113259, 0.0119129

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310201, upper bound: 0.0307907
time: 2.20 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0316807, upper bound: 0.0315455
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0002972, 0.0064598, 0.0001136, 0.0065395, -0.0056590, 0.0057982
1: 0.0013652, 0.0022556, 0.0013387, 0.0022671, -0.0008176, 0.0008377
2: 0.0107884, 0.0141955, 0.0107443, 0.0142971, -0.0032057, 0.0031287
3: -0.0035226, 0.0000012, -0.0035681, 0.0001062, -0.0033155, 0.0032359
4: -0.0040383, -0.0002236, -0.0041520, -0.0001742, -0.0035030, 0.0035892
5: 0.0043249, 0.0079349, 0.0042782, 0.0080425, -0.0033966, 0.0033150
6: -0.0051404, 0.0091831, -0.0053256, 0.0096100, -0.0134766, 0.0131531
7: -0.0150633, 0.0044440, -0.0156447, 0.0046963, -0.0179134, 0.0183540
8: 0.9786029, 0.9923444, 0.9781935, 0.9925221, -0.0126186, 0.0129289
9: -0.0089380, 0.0035356, -0.0090993, 0.0039073, -0.0117361, 0.0114543

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0292242, upper bound: 0.0304728
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0296173, upper bound: 0.0310214
time: 1.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0003008, 0.0064319, 0.0000397, 0.0065104, -0.0056792, 0.0059127
1: 0.0013658, 0.0022515, 0.0013280, 0.0022629, -0.0008205, 0.0008542
2: 0.0108039, 0.0141936, 0.0107604, 0.0143379, -0.0032690, 0.0031399
3: -0.0035066, -0.0000008, -0.0035515, 0.0001485, -0.0033809, 0.0032474
4: -0.0040361, -0.0002408, -0.0041977, -0.0001922, -0.0035155, 0.0036600
5: 0.0043413, 0.0079329, 0.0042953, 0.0080858, -0.0034636, 0.0033269
6: -0.0050754, 0.0091749, -0.0052579, 0.0097817, -0.0137427, 0.0132000
7: -0.0150521, 0.0043556, -0.0158786, 0.0046041, -0.0179773, 0.0187163
8: 0.9786109, 0.9922820, 0.9780286, 0.9924571, -0.0126636, 0.0131842
9: -0.0088814, 0.0035284, -0.0090403, 0.0040569, -0.0119677, 0.0114951

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0292075, upper bound: 0.0304728
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0296022, upper bound: 0.0310214
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0001492, 0.0065082, 0.0000380, 0.0065496, -0.0058408, 0.0059348
1: 0.0013439, 0.0022625, 0.0013278, 0.0022685, -0.0008438, 0.0008574
2: 0.0107617, 0.0142774, 0.0107388, 0.0143388, -0.0032812, 0.0032292
3: -0.0035502, 0.0000859, -0.0035739, 0.0001494, -0.0033936, 0.0033398
4: -0.0041299, -0.0001936, -0.0041987, -0.0001680, -0.0036156, 0.0036737
5: 0.0042966, 0.0080217, 0.0042723, 0.0080868, -0.0034766, 0.0034215
6: -0.0052528, 0.0095273, -0.0053490, 0.0097856, -0.0137941, 0.0135757
7: -0.0155321, 0.0045971, -0.0158838, 0.0047282, -0.0184889, 0.0187863
8: 0.9782727, 0.9924521, 0.9780250, 0.9925445, -0.0130239, 0.0132335
9: -0.0090359, 0.0038353, -0.0091197, 0.0040602, -0.0120125, 0.0118223

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311166, upper bound: 0.0315156
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319225, upper bound: 0.0322676
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0001550, 0.0064726, -0.0000355, 0.0065210, -0.0060011, 0.0060485
1: 0.0013447, 0.0022574, 0.0012673, 0.0022644, -0.0008458, 0.0009901
2: 0.0107813, 0.0142742, 0.0107546, 0.0144421, -0.0036608, 0.0032368
3: -0.0035299, 0.0000826, -0.0035576, 0.0002330, -0.0037629, 0.0033477
4: -0.0041263, -0.0002156, -0.0042442, -0.0001857, -0.0039406, 0.0037441
5: 0.0043174, 0.0080183, 0.0042891, 0.0081707, -0.0038533, 0.0034296
6: -0.0051701, 0.0095138, -0.0052825, 0.0099565, -0.0140584, 0.0147963
7: -0.0155137, 0.0044845, -0.0161372, 0.0046376, -0.0185324, 0.0206217
8: 0.9782858, 0.9923729, 0.9775559, 0.9924807, -0.0130546, 0.0148170
9: -0.0089639, 0.0038235, -0.0090618, 0.0042495, -0.0132133, 0.0118501

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0310962, upper bound: 0.0315156
time: 2.42 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0319034, upper bound: 0.0322676
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0002109, 0.0065606, 0.0003840, 0.0064578, -0.0057400, 0.0056985
1: 0.0013528, 0.0022701, 0.0013778, 0.0022553, -0.0008293, 0.0008233
2: 0.0107327, 0.0142432, 0.0107895, 0.0141475, -0.0031505, 0.0031735
3: -0.0035802, 0.0000506, -0.0035214, -0.0000484, -0.0032584, 0.0032822
4: -0.0040917, -0.0001612, -0.0039846, -0.0002248, -0.0035532, 0.0035275
5: 0.0042659, 0.0079855, 0.0043261, 0.0078841, -0.0033382, 0.0033625
6: -0.0053746, 0.0093837, -0.0051356, 0.0089814, -0.0132448, 0.0133414
7: -0.0153366, 0.0047630, -0.0147886, 0.0044375, -0.0181698, 0.0180383
8: 0.9784105, 0.9925690, 0.9787965, 0.9923398, -0.0127992, 0.0127066
9: -0.0091419, 0.0037103, -0.0089338, 0.0033599, -0.0115342, 0.0116182

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303267, upper bound: 0.0290529
time: 1.71 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309209, upper bound: 0.0295310
time: 1.90 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0001458, 0.0065294, 0.0003876, 0.0064303, -0.0058495, 0.0057846
1: 0.0013434, 0.0022656, 0.0013783, 0.0022513, -0.0008451, 0.0008357
2: 0.0107499, 0.0142792, 0.0108047, 0.0141456, -0.0031982, 0.0032340
3: -0.0035624, 0.0000878, -0.0035057, -0.0000504, -0.0033077, 0.0033448
4: -0.0041320, -0.0001805, -0.0039824, -0.0002418, -0.0036209, 0.0035808
5: 0.0042841, 0.0080236, 0.0043422, 0.0078820, -0.0033886, 0.0034266
6: -0.0053021, 0.0095350, -0.0050718, 0.0089732, -0.0134450, 0.0135958
7: -0.0155426, 0.0046643, -0.0147774, 0.0043506, -0.0185163, 0.0183110
8: 0.9782653, 0.9924995, 0.9788043, 0.9922785, -0.0130433, 0.0128986
9: -0.0090788, 0.0038420, -0.0088782, 0.0033527, -0.0117085, 0.0118398

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304225, upper bound: 0.0290937
time: 2.17 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0309209, upper bound: 0.0295077
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0001337, 0.0065707, 0.0002272, 0.0065070, -0.0058765, 0.0058631
1: 0.0013416, 0.0022716, 0.0013551, 0.0022624, -0.0008490, 0.0008470
2: 0.0107271, 0.0142860, 0.0107623, 0.0142343, -0.0032415, 0.0032490
3: -0.0035860, 0.0000947, -0.0035496, 0.0000413, -0.0033526, 0.0033602
4: -0.0041395, -0.0001549, -0.0040816, -0.0001943, -0.0036377, 0.0036293
5: 0.0042600, 0.0080307, 0.0042972, 0.0079760, -0.0034346, 0.0034425
6: -0.0053980, 0.0095633, -0.0052502, 0.0093459, -0.0136274, 0.0136587
7: -0.0155811, 0.0047950, -0.0152851, 0.0045936, -0.0186019, 0.0185594
8: 0.9782382, 0.9925916, 0.9784467, 0.9924497, -0.0131036, 0.0130736
9: -0.0091624, 0.0038666, -0.0090336, 0.0036774, -0.0118674, 0.0118946

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312630, upper bound: 0.0309491
time: 1.83 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0321582, upper bound: 0.0316496
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0000694, 0.0065397, 0.0002330, 0.0064711, -0.0059861, 0.0059616
1: 0.0013323, 0.0022671, 0.0013560, 0.0022572, -0.0008648, 0.0008613
2: 0.0107443, 0.0143215, 0.0107821, 0.0142310, -0.0032960, 0.0033095
3: -0.0035682, 0.0001315, -0.0035291, 0.0000379, -0.0034089, 0.0034229
4: -0.0041793, -0.0001741, -0.0040780, -0.0002165, -0.0037055, 0.0036904
5: 0.0042781, 0.0080684, 0.0043183, 0.0079725, -0.0034923, 0.0035066
6: -0.0053260, 0.0097128, -0.0051667, 0.0093324, -0.0138565, 0.0139132
7: -0.0157847, 0.0046968, -0.0152666, 0.0044799, -0.0189486, 0.0188713
8: 0.9780948, 0.9925224, 0.9784598, 0.9923697, -0.0133478, 0.0132934
9: -0.0090996, 0.0039968, -0.0089609, 0.0036655, -0.0120668, 0.0121163

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0312630, upper bound: 0.0309460
time: 1.82 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0321582, upper bound: 0.0316270
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0001939, 0.0065628, 0.0003084, 0.0065823, -0.0059230, 0.0058310
1: 0.0013503, 0.0022704, 0.0013668, 0.0022732, -0.0008557, 0.0008424
2: 0.0107315, 0.0142527, 0.0107207, 0.0141894, -0.0032238, 0.0032747
3: -0.0035815, 0.0000603, -0.0035926, -0.0000051, -0.0033342, 0.0033868
4: -0.0041023, -0.0001598, -0.0040314, -0.0001477, -0.0036664, 0.0036095
5: 0.0042646, 0.0079955, 0.0042532, 0.0079284, -0.0034158, 0.0034697
6: -0.0053798, 0.0094233, -0.0054250, 0.0091573, -0.0135527, 0.0137666
7: -0.0153904, 0.0047701, -0.0150281, 0.0048317, -0.0187489, 0.0184576
8: 0.9783725, 0.9925740, 0.9786277, 0.9926174, -0.0132071, 0.0130020
9: -0.0091465, 0.0037447, -0.0091859, 0.0035131, -0.0118023, 0.0119886

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304456, upper bound: 0.0295851
time: 2.22 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311216, upper bound: 0.0300444
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0001272, 0.0065315, 0.0003127, 0.0065518, -0.0060320, 0.0058410
1: 0.0013407, 0.0022659, 0.0013675, 0.0022689, -0.0008714, 0.0008439
2: 0.0107488, 0.0142896, 0.0107375, 0.0141870, -0.0032293, 0.0033349
3: -0.0035636, 0.0000985, -0.0035752, -0.0000076, -0.0033399, 0.0034491
4: -0.0041436, -0.0001792, -0.0040287, -0.0001666, -0.0037339, 0.0036157
5: 0.0042829, 0.0080346, 0.0042710, 0.0079259, -0.0034216, 0.0035335
6: -0.0053070, 0.0095784, -0.0053543, 0.0091471, -0.0135760, 0.0140200
7: -0.0156017, 0.0046709, -0.0150143, 0.0047353, -0.0190940, 0.0184894
8: 0.9782237, 0.9925041, 0.9786375, 0.9925495, -0.0134502, 0.0130243
9: -0.0090830, 0.0038798, -0.0091242, 0.0035042, -0.0118226, 0.0122092

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0305403, upper bound: 0.0295977
time: 1.83 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0311216, upper bound: 0.0300251
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0001198, 0.0065728, 0.0001570, 0.0066242, -0.0060956, 0.0059162
1: 0.0013396, 0.0022719, 0.0013450, 0.0022793, -0.0008806, 0.0008547
2: 0.0107259, 0.0142936, 0.0106975, 0.0142731, -0.0032709, 0.0033701
3: -0.0035872, 0.0001027, -0.0036166, 0.0000814, -0.0033829, 0.0034855
4: -0.0041481, -0.0001536, -0.0041251, -0.0001218, -0.0037733, 0.0036622
5: 0.0042587, 0.0080389, 0.0042286, 0.0080171, -0.0034657, 0.0035708
6: -0.0054030, 0.0095956, -0.0055225, 0.0095091, -0.0137508, 0.0141679
7: -0.0156251, 0.0048017, -0.0155073, 0.0049644, -0.0192955, 0.0187274
8: 0.9782072, 0.9925963, 0.9782903, 0.9927109, -0.0135921, 0.0131920
9: -0.0091667, 0.0038948, -0.0092707, 0.0038194, -0.0119748, 0.0123381

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0315804
time: 1.98 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323616, upper bound: 0.0323807
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0000505, 0.0065417, 0.0001627, 0.0065914, -0.0062073, 0.0060166
1: 0.0013296, 0.0022674, 0.0013458, 0.0022746, -0.0008968, 0.0008692
2: 0.0107431, 0.0143320, 0.0107157, 0.0142699, -0.0033264, 0.0034318
3: -0.0035694, 0.0001423, -0.0035978, 0.0000781, -0.0034403, 0.0035494
4: -0.0041910, -0.0001728, -0.0041215, -0.0001421, -0.0038424, 0.0037244
5: 0.0042769, 0.0080795, 0.0042478, 0.0080137, -0.0035245, 0.0036362
6: -0.0053308, 0.0097567, -0.0054461, 0.0094957, -0.0139842, 0.0144274
7: -0.0158445, 0.0047033, -0.0154891, 0.0048604, -0.0196489, 0.0190453
8: 0.9780527, 0.9925269, 0.9783031, 0.9926376, -0.0138411, 0.0134159
9: -0.0091038, 0.0040350, -0.0092042, 0.0038078, -0.0121781, 0.0125640

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0315711
time: 1.90 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0323616, upper bound: 0.0323616
time: 1.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.19 seconds
NS_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0299722, upper bound: 0.0290362
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0304171, upper bound: 0.0294130
NS_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0299722, upper bound: 0.0290173
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0304171, upper bound: 0.0293975
NS_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0309755, upper bound: 0.0308888
NS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0316807, upper bound: 0.0315587
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0310201, upper bound: 0.0307907
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0316807, upper bound: 0.0315455
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0292242, upper bound: 0.0304728
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0296173, upper bound: 0.0310214
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0292075, upper bound: 0.0304728
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0296022, upper bound: 0.0310214
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0311166, upper bound: 0.0315156
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0319225, upper bound: 0.0322676
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0310962, upper bound: 0.0315156
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0319034, upper bound: 0.0322676
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0303267, upper bound: 0.0290529
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0309209, upper bound: 0.0295310
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0304225, upper bound: 0.0290937
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0309209, upper bound: 0.0295077
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0312630, upper bound: 0.0309491
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0321582, upper bound: 0.0316496
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0312630, upper bound: 0.0309460
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0321582, upper bound: 0.0316270
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0304456, upper bound: 0.0295851
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0311216, upper bound: 0.0300444
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0305403, upper bound: 0.0295977
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0311216, upper bound: 0.0300251
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0315804
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0323616, upper bound: 0.0323807
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0313953, upper bound: 0.0315711
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.19
Output dim: 8, lower bound: -0.0323616, upper bound: 0.0323616

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0001763, 0.0063990, 0.0003386, 0.0063657, -0.0056246, 0.0053795
1: 0.0013478, 0.0022468, 0.0013712, 0.0022420, -0.0008126, 0.0007772
2: 0.0108220, 0.0142624, 0.0108404, 0.0141726, -0.0029742, 0.0031097
3: -0.0034878, 0.0000704, -0.0034688, -0.0000224, -0.0030761, 0.0032162
4: -0.0041132, -0.0002612, -0.0040127, -0.0002818, -0.0034817, 0.0033300
5: 0.0043605, 0.0080058, 0.0043800, 0.0079107, -0.0031513, 0.0032949
6: -0.0049990, 0.0094643, -0.0049217, 0.0090869, -0.0125035, 0.0130731
7: -0.0154462, 0.0042515, -0.0149323, 0.0041463, -0.0178044, 0.0170287
8: 0.9783332, 0.9922087, 0.9786953, 0.9921346, -0.0125418, 0.0119954
9: -0.0088149, 0.0037804, -0.0087476, 0.0034518, -0.0108886, 0.0113846

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0282079, upper bound: 0.0275100
time: 1.91 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284344, upper bound: 0.0274853
time: 2.06 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0001044, 0.0063663, 0.0003423, 0.0063366, -0.0056699, 0.0053952
1: 0.0013374, 0.0022420, 0.0013718, 0.0022378, -0.0008191, 0.0007795
2: 0.0108401, 0.0143022, 0.0108565, 0.0141706, -0.0029829, 0.0031348
3: -0.0034691, 0.0001115, -0.0034522, -0.0000245, -0.0030850, 0.0032421
4: -0.0041577, -0.0002814, -0.0040104, -0.0002998, -0.0035098, 0.0033397
5: 0.0043797, 0.0080479, 0.0043971, 0.0079085, -0.0031605, 0.0033214
6: -0.0049230, 0.0096314, -0.0048541, 0.0090784, -0.0125400, 0.0131785
7: -0.0156739, 0.0041480, -0.0149207, 0.0040541, -0.0179479, 0.0170784
8: 0.9781728, 0.9921357, 0.9787034, 0.9920697, -0.0126429, 0.0120304
9: -0.0087487, 0.0039260, -0.0086887, 0.0034444, -0.0109204, 0.0114764

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284426, upper bound: 0.0272943
time: 2.05 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0284345, upper bound: 0.0274814
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0000381, 0.0061293, 0.0002088, 0.0063730, -0.0057462, 0.0051357
1: 0.0013168, 0.0022078, 0.0013525, 0.0022430, -0.0008302, 0.0007420
2: 0.0109711, 0.0143810, 0.0108364, 0.0142444, -0.0028394, 0.0031769
3: -0.0033336, 0.0001930, -0.0034730, 0.0000518, -0.0029367, 0.0032857
4: -0.0042459, -0.0004281, -0.0040930, -0.0002773, -0.0035570, 0.0031791
5: 0.0045185, 0.0081314, 0.0043757, 0.0079867, -0.0030085, 0.0033661
6: -0.0043722, 0.0099627, -0.0049387, 0.0093887, -0.0119368, 0.0133558
7: -0.0161250, 0.0033979, -0.0153433, 0.0041694, -0.0181895, 0.0162569
8: 0.9778551, 0.9916074, 0.9784057, 0.9921508, -0.0128130, 0.0114517
9: -0.0082690, 0.0042144, -0.0087624, 0.0037146, -0.0103951, 0.0116308

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286042, upper bound: 0.0287163
time: 1.77 seconds

## Relational analysis of NS_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286247, upper bound: 0.0286254
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0001168, 0.0063308, 0.0001759, 0.0064889, -0.0057432, 0.0054690
1: 0.0013392, 0.0022369, 0.0013477, 0.0022598, -0.0008297, 0.0007901
2: 0.0108597, 0.0142953, 0.0107723, 0.0142626, -0.0030237, 0.0031753
3: -0.0034488, 0.0001044, -0.0035392, 0.0000706, -0.0031272, 0.0032840
4: -0.0041500, -0.0003034, -0.0041134, -0.0002055, -0.0035551, 0.0033854
5: 0.0044005, 0.0080407, 0.0043079, 0.0080060, -0.0032037, 0.0033644
6: -0.0048405, 0.0096026, -0.0052080, 0.0094653, -0.0127114, 0.0133488
7: -0.0156347, 0.0040356, -0.0154476, 0.0045361, -0.0181799, 0.0173118
8: 0.9782005, 0.9920567, 0.9783323, 0.9924092, -0.0128063, 0.0121948
9: -0.0086768, 0.0039009, -0.0089968, 0.0037813, -0.0110697, 0.0116247

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295060, upper bound: 0.0292518
time: 1.86 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294670, upper bound: 0.0293546
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0000619, 0.0062630, 0.0000334, 0.0061744, -0.0055338, 0.0054667
1: 0.0013312, 0.0022271, 0.0013271, 0.0022143, -0.0007995, 0.0007898
2: 0.0108972, 0.0143257, 0.0109462, 0.0143414, -0.0030224, 0.0030595
3: -0.0034100, 0.0001358, -0.0033594, 0.0001521, -0.0031259, 0.0031643
4: -0.0041840, -0.0003454, -0.0042016, -0.0004002, -0.0034255, 0.0033840
5: 0.0044402, 0.0080728, 0.0044921, 0.0080895, -0.0032024, 0.0032417
6: -0.0046829, 0.0097302, -0.0044771, 0.0097964, -0.0127061, 0.0128621
7: -0.0158084, 0.0038210, -0.0158986, 0.0035407, -0.0175170, 0.0173046
8: 0.9780781, 0.9919055, 0.9780146, 0.9917080, -0.0123394, 0.0121897
9: -0.0085396, 0.0040120, -0.0083604, 0.0040696, -0.0110650, 0.0112009

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286067, upper bound: 0.0286041
time: 1.77 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287049, upper bound: 0.0285153
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0000330, 0.0063764, 0.0001971, 0.0063736, -0.0057815, 0.0055005
1: 0.0013271, 0.0022435, 0.0013508, 0.0022431, -0.0008353, 0.0007947
2: 0.0108345, 0.0143416, 0.0108361, 0.0142509, -0.0030411, 0.0031964
3: -0.0034749, 0.0001523, -0.0034733, 0.0000585, -0.0031452, 0.0033059
4: -0.0042018, -0.0002752, -0.0041003, -0.0002769, -0.0035788, 0.0034049
5: 0.0043738, 0.0080897, 0.0043754, 0.0079936, -0.0032222, 0.0033868
6: -0.0049466, 0.0097973, -0.0049401, 0.0094159, -0.0127846, 0.0134378
7: -0.0158997, 0.0041801, -0.0153804, 0.0041712, -0.0183011, 0.0174115
8: 0.9780138, 0.9921584, 0.9783795, 0.9921522, -0.0128917, 0.0122650
9: -0.0087692, 0.0040704, -0.0087635, 0.0037383, -0.0111334, 0.0117022

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295060, upper bound: 0.0292608
time: 1.75 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294670, upper bound: 0.0293495
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0001636, 0.0061451, 0.0001471, 0.0064012, -0.0056147, 0.0053410
1: 0.0013459, 0.0022101, 0.0013436, 0.0022471, -0.0008112, 0.0007716
2: 0.0109624, 0.0142694, 0.0108208, 0.0142785, -0.0029529, 0.0031042
3: -0.0033426, 0.0000776, -0.0034891, 0.0000871, -0.0030540, 0.0032105
4: -0.0041210, -0.0004184, -0.0041312, -0.0002598, -0.0034756, 0.0033062
5: 0.0045093, 0.0080132, 0.0043592, 0.0080229, -0.0031287, 0.0032891
6: -0.0044089, 0.0094937, -0.0050042, 0.0095321, -0.0124139, 0.0130502
7: -0.0154863, 0.0034478, -0.0155385, 0.0042585, -0.0177732, 0.0169067
8: 0.9783050, 0.9916425, 0.9782683, 0.9922137, -0.0125198, 0.0119094
9: -0.0083010, 0.0038060, -0.0088194, 0.0038394, -0.0108106, 0.0113647

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272751, upper bound: 0.0283071
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272682, upper bound: 0.0287644
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0003121, 0.0063630, 0.0001184, 0.0065169, -0.0056277, 0.0056930
1: 0.0013674, 0.0022416, 0.0013394, 0.0022638, -0.0008130, 0.0008225
2: 0.0108419, 0.0141873, 0.0107569, 0.0142944, -0.0031475, 0.0031114
3: -0.0034672, -0.0000073, -0.0035552, 0.0001035, -0.0032553, 0.0032180
4: -0.0040291, -0.0002835, -0.0041490, -0.0001882, -0.0034837, 0.0035241
5: 0.0043816, 0.0079262, 0.0042915, 0.0080397, -0.0033349, 0.0032967
6: -0.0049154, 0.0091485, -0.0052730, 0.0095989, -0.0132321, 0.0130804
7: -0.0150162, 0.0041376, -0.0156296, 0.0046246, -0.0178143, 0.0180209
8: 0.9786361, 0.9921286, 0.9782040, 0.9924715, -0.0125488, 0.0126943
9: -0.0087421, 0.0035054, -0.0090535, 0.0038976, -0.0115231, 0.0113910

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0277895, upper bound: 0.0293535
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0278122, upper bound: 0.0293400
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0001673, 0.0061172, 0.0000713, 0.0063727, -0.0056360, 0.0054518
1: 0.0013465, 0.0022061, 0.0013326, 0.0022430, -0.0008142, 0.0007876
2: 0.0109778, 0.0142674, 0.0108365, 0.0143205, -0.0030141, 0.0031160
3: -0.0033267, 0.0000755, -0.0034728, 0.0001304, -0.0031174, 0.0032227
4: -0.0041187, -0.0004356, -0.0041782, -0.0002775, -0.0034888, 0.0033747
5: 0.0045256, 0.0080110, 0.0043759, 0.0080673, -0.0031936, 0.0033016
6: -0.0043441, 0.0094851, -0.0049380, 0.0097083, -0.0126714, 0.0130997
7: -0.0154746, 0.0033596, -0.0157786, 0.0041684, -0.0178407, 0.0172574
8: 0.9783132, 0.9915804, 0.9780990, 0.9921502, -0.0125673, 0.0121564
9: -0.0082446, 0.0037985, -0.0087617, 0.0039929, -0.0110348, 0.0114078

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272558, upper bound: 0.0283071
time: 2.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272500, upper bound: 0.0287646
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0003158, 0.0063336, 0.0000437, 0.0064881, -0.0056474, 0.0057344
1: 0.0013679, 0.0022373, 0.0013286, 0.0022596, -0.0008159, 0.0008285
2: 0.0108582, 0.0141852, 0.0107727, 0.0143357, -0.0031704, 0.0031223
3: -0.0034504, -0.0000094, -0.0035388, 0.0001462, -0.0032790, 0.0032292
4: -0.0040268, -0.0003017, -0.0041952, -0.0002060, -0.0034958, 0.0035497
5: 0.0043988, 0.0079240, 0.0043083, 0.0080835, -0.0033592, 0.0033082
6: -0.0048471, 0.0091399, -0.0052062, 0.0097724, -0.0133284, 0.0131262
7: -0.0150045, 0.0040446, -0.0158659, 0.0045337, -0.0178767, 0.0181521
8: 0.9786444, 0.9920629, 0.9780376, 0.9924076, -0.0125927, 0.0127867
9: -0.0086826, 0.0034979, -0.0089953, 0.0040488, -0.0116070, 0.0114308

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0277937, upper bound: 0.0293535
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0277948, upper bound: 0.0293400
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0000050, 0.0062036, 0.0000721, 0.0064111, -0.0056903, 0.0054895
1: 0.0013230, 0.0022185, 0.0013327, 0.0022485, -0.0008221, 0.0007931
2: 0.0109301, 0.0143571, 0.0108153, 0.0143200, -0.0030350, 0.0031460
3: -0.0033761, 0.0001683, -0.0034947, 0.0001299, -0.0031389, 0.0032538
4: -0.0042192, -0.0003822, -0.0041776, -0.0002537, -0.0035224, 0.0033981
5: 0.0044750, 0.0081061, 0.0043534, 0.0080668, -0.0032157, 0.0033334
6: -0.0045448, 0.0098623, -0.0050272, 0.0097063, -0.0127591, 0.0132258
7: -0.0159883, 0.0036329, -0.0157759, 0.0042899, -0.0180124, 0.0173768
8: 0.9779513, 0.9917729, 0.9781010, 0.9922357, -0.0126883, 0.0122406
9: -0.0084193, 0.0041270, -0.0088394, 0.0039912, -0.0111112, 0.0115176

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289395, upper bound: 0.0293000
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289194, upper bound: 0.0296123
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0001701, 0.0064065, 0.0000428, 0.0065270, -0.0058124, 0.0058222
1: 0.0013469, 0.0022479, 0.0013285, 0.0022653, -0.0008397, 0.0008411
2: 0.0108179, 0.0142658, 0.0107513, 0.0143362, -0.0032189, 0.0032135
3: -0.0034921, 0.0000739, -0.0035610, 0.0001467, -0.0033292, 0.0033236
4: -0.0041170, -0.0002566, -0.0041958, -0.0001820, -0.0035980, 0.0036040
5: 0.0043561, 0.0080094, 0.0042856, 0.0080840, -0.0034106, 0.0034049
6: -0.0050165, 0.0094787, -0.0052965, 0.0097746, -0.0135323, 0.0135096
7: -0.0154659, 0.0042753, -0.0158688, 0.0046566, -0.0183990, 0.0184298
8: 0.9783193, 0.9922255, 0.9780355, 0.9924940, -0.0129606, 0.0129824
9: -0.0088301, 0.0037930, -0.0090739, 0.0040506, -0.0117845, 0.0117648

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0299145, upper bound: 0.0304469
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0298897, upper bound: 0.0304096
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0000091, 0.0061711, -0.0000036, 0.0063832, -0.0059921, 0.0056028
1: 0.0013236, 0.0022139, 0.0013161, 0.0022445, -0.0008251, 0.0008977
2: 0.0109480, 0.0143548, 0.0108307, 0.0143689, -0.0034209, 0.0031576
3: -0.0033575, 0.0001660, -0.0034788, 0.0001779, -0.0035355, 0.0032658
4: -0.0042166, -0.0004022, -0.0042245, -0.0002709, -0.0039457, 0.0034682
5: 0.0044940, 0.0081037, 0.0043698, 0.0081158, -0.0036218, 0.0033457
6: -0.0044694, 0.0098528, -0.0049624, 0.0098823, -0.0130224, 0.0148152
7: -0.0159753, 0.0035302, -0.0160179, 0.0042017, -0.0180789, 0.0195482
8: 0.9779605, 0.9917006, 0.9778977, 0.9921736, -0.0127351, 0.0138029
9: -0.0083537, 0.0041187, -0.0087830, 0.0041490, -0.0125027, 0.0115601

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289210, upper bound: 0.0293000
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296124
time: 2.04 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0001748, 0.0063706, -0.0000315, 0.0064987, -0.0059528, 0.0058509
1: 0.0013476, 0.0022427, 0.0012734, 0.0022612, -0.0008276, 0.0009693
2: 0.0108377, 0.0142632, 0.0107669, 0.0144330, -0.0035952, 0.0031670
3: -0.0034716, 0.0000712, -0.0035448, 0.0002261, -0.0036976, 0.0032755
4: -0.0041141, -0.0002788, -0.0042418, -0.0001995, -0.0039146, 0.0036218
5: 0.0043772, 0.0080066, 0.0043021, 0.0081638, -0.0037867, 0.0033556
6: -0.0049330, 0.0094677, -0.0052308, 0.0099472, -0.0135990, 0.0146984
7: -0.0154509, 0.0041616, -0.0161223, 0.0045671, -0.0181326, 0.0202838
8: 0.9783299, 0.9921454, 0.9775987, 0.9924311, -0.0127730, 0.0145467
9: -0.0087574, 0.0037834, -0.0090167, 0.0042369, -0.0129943, 0.0115945

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301420
time: 2.06 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304096
time: 1.95 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0000841, 0.0062567, 0.0004152, 0.0063122, -0.0056050, 0.0053486
1: 0.0013345, 0.0022262, 0.0013823, 0.0022342, -0.0008098, 0.0007727
2: 0.0109007, 0.0143134, 0.0108700, 0.0141303, -0.0029571, 0.0030989
3: -0.0034064, 0.0001231, -0.0034382, -0.0000662, -0.0030584, 0.0032050
4: -0.0041702, -0.0003493, -0.0039653, -0.0003149, -0.0034696, 0.0033109
5: 0.0044439, 0.0080598, 0.0044113, 0.0078658, -0.0031332, 0.0032834
6: -0.0046683, 0.0096785, -0.0047974, 0.0089090, -0.0124317, 0.0130276
7: -0.0157380, 0.0038011, -0.0146900, 0.0039769, -0.0177424, 0.0169309
8: 0.9781276, 0.9918914, 0.9788659, 0.9920152, -0.0124981, 0.0119265
9: -0.0085268, 0.0039670, -0.0086393, 0.0032968, -0.0108261, 0.0113450

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285691, upper bound: 0.0270908
time: 1.92 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285432, upper bound: 0.0272372
time: 2.09 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0002327, 0.0064551, 0.0003874, 0.0064363, -0.0057169, 0.0055877
1: 0.0013559, 0.0022549, 0.0013783, 0.0022522, -0.0008259, 0.0008073
2: 0.0107910, 0.0142312, 0.0108014, 0.0141457, -0.0030893, 0.0031607
3: -0.0035199, 0.0000381, -0.0035091, -0.0000503, -0.0031951, 0.0032690
4: -0.0040782, -0.0002265, -0.0039825, -0.0002381, -0.0035388, 0.0034589
5: 0.0043276, 0.0079727, 0.0043387, 0.0078821, -0.0032733, 0.0033489
6: -0.0051295, 0.0093330, -0.0050856, 0.0089735, -0.0129874, 0.0132876
7: -0.0152675, 0.0044292, -0.0147779, 0.0043695, -0.0180965, 0.0176877
8: 0.9784591, 0.9923339, 0.9788041, 0.9922919, -0.0127476, 0.0124596
9: -0.0089285, 0.0036661, -0.0088903, 0.0033530, -0.0113100, 0.0115714

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292621, upper bound: 0.0276150
time: 1.80 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292594, upper bound: 0.0277506
time: 1.86 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0001797, 0.0063971, 0.0002519, 0.0061161, -0.0053948, 0.0057434
1: 0.0013483, 0.0022465, 0.0013587, 0.0022059, -0.0007794, 0.0008298
2: 0.0108231, 0.0142605, 0.0109785, 0.0142206, -0.0031754, 0.0029827
3: -0.0034867, 0.0000684, -0.0033260, 0.0000272, -0.0032841, 0.0030848
4: -0.0041110, -0.0002624, -0.0040664, -0.0004363, -0.0033395, 0.0035553
5: 0.0043616, 0.0080038, 0.0045263, 0.0079615, -0.0033645, 0.0031603
6: -0.0049946, 0.0094563, -0.0043414, 0.0092886, -0.0133492, 0.0125391
7: -0.0154354, 0.0042455, -0.0152070, 0.0033559, -0.0170772, 0.0181805
8: 0.9783407, 0.9922045, 0.9785018, 0.9915779, -0.0120295, 0.0128067
9: -0.0088110, 0.0037735, -0.0082422, 0.0036274, -0.0116251, 0.0109196

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0282621, upper bound: 0.0272476
time: 2.47 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0286969, upper bound: 0.0272445
time: 1.87 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0001507, 0.0065052, 0.0004029, 0.0063321, -0.0057426, 0.0057500
1: 0.0013441, 0.0022621, 0.0013805, 0.0022371, -0.0008296, 0.0008307
2: 0.0107633, 0.0142765, 0.0108590, 0.0141371, -0.0031790, 0.0031749
3: -0.0035485, 0.0000850, -0.0034496, -0.0000592, -0.0032879, 0.0032837
4: -0.0041290, -0.0001955, -0.0039729, -0.0003026, -0.0035548, 0.0035593
5: 0.0042983, 0.0080208, 0.0043997, 0.0078730, -0.0033683, 0.0033640
6: -0.0052458, 0.0095237, -0.0048435, 0.0089375, -0.0133646, 0.0133474
7: -0.0155272, 0.0045877, -0.0147288, 0.0040397, -0.0181780, 0.0182014
8: 0.9782762, 0.9924455, 0.9788386, 0.9920595, -0.0128050, 0.0128214
9: -0.0090298, 0.0038322, -0.0086795, 0.0033216, -0.0116385, 0.0116235

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292621, upper bound: 0.0276188
time: 2.13 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0292594, upper bound: 0.0277381
time: 2.08 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0000006, 0.0062680, 0.0002642, 0.0063685, -0.0057509, 0.0054063
1: 0.0013222, 0.0022279, 0.0013605, 0.0022424, -0.0008308, 0.0007811
2: 0.0108944, 0.0143602, 0.0108389, 0.0142138, -0.0029890, 0.0031795
3: -0.0034129, 0.0001715, -0.0034704, 0.0000201, -0.0030914, 0.0032884
4: -0.0042227, -0.0003423, -0.0040588, -0.0002801, -0.0035599, 0.0033466
5: 0.0044372, 0.0081094, 0.0043784, 0.0079543, -0.0031670, 0.0033689
6: -0.0046946, 0.0098755, -0.0049281, 0.0092600, -0.0125657, 0.0133667
7: -0.0160062, 0.0038370, -0.0151681, 0.0041550, -0.0182043, 0.0171134
8: 0.9779387, 0.9919168, 0.9785292, 0.9921407, -0.0128235, 0.0120550
9: -0.0085498, 0.0041385, -0.0087532, 0.0036025, -0.0109428, 0.0116403

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293824, upper bound: 0.0288452
time: 1.96 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293527, upper bound: 0.0288772
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0001552, 0.0064654, 0.0002318, 0.0064844, -0.0058461, 0.0057406
1: 0.0013447, 0.0022564, 0.0013558, 0.0022591, -0.0008446, 0.0008294
2: 0.0107853, 0.0142740, 0.0107748, 0.0142317, -0.0031738, 0.0032322
3: -0.0035258, 0.0000824, -0.0035366, 0.0000386, -0.0032825, 0.0033429
4: -0.0041262, -0.0002201, -0.0040788, -0.0002083, -0.0036189, 0.0035535
5: 0.0043216, 0.0080181, 0.0043105, 0.0079732, -0.0033628, 0.0034247
6: -0.0051534, 0.0095132, -0.0051975, 0.0093351, -0.0133428, 0.0135881
7: -0.0155128, 0.0044618, -0.0152704, 0.0045218, -0.0185058, 0.0181717
8: 0.9782864, 0.9923568, 0.9784571, 0.9923991, -0.0130358, 0.0128005
9: -0.0089493, 0.0038230, -0.0089877, 0.0036679, -0.0116195, 0.0118331

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303739, upper bound: 0.0296252
time: 1.88 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303472, upper bound: 0.0296110
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0000763, 0.0062489, 0.0002680, 0.0063327, -0.0058555, 0.0054915
1: 0.0013113, 0.0022251, 0.0013610, 0.0022372, -0.0008459, 0.0007934
2: 0.0109050, 0.0144020, 0.0108587, 0.0142117, -0.0030361, 0.0032373
3: -0.0034020, 0.0002148, -0.0034499, 0.0000179, -0.0031401, 0.0033482
4: -0.0042695, -0.0003541, -0.0040564, -0.0003023, -0.0036246, 0.0033993
5: 0.0044485, 0.0081537, 0.0043994, 0.0079521, -0.0032169, 0.0034301
6: -0.0046501, 0.0100512, -0.0048449, 0.0092511, -0.0127637, 0.0136097
7: -0.0162456, 0.0037764, -0.0151559, 0.0040416, -0.0185353, 0.0173830
8: 0.9777701, 0.9918741, 0.9785377, 0.9920608, -0.0130566, 0.0122450
9: -0.0085111, 0.0042916, -0.0086806, 0.0035948, -0.0111152, 0.0118520

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291525, upper bound: 0.0289128
time: 1.66 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0293527, upper bound: 0.0288775
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0000880, 0.0064331, 0.0002377, 0.0064485, -0.0058615, 0.0058413
1: 0.0013350, 0.0022517, 0.0013566, 0.0022539, -0.0008468, 0.0008439
2: 0.0108031, 0.0143112, 0.0107947, 0.0142284, -0.0032295, 0.0032407
3: -0.0035073, 0.0001209, -0.0035161, 0.0000353, -0.0033401, 0.0033517
4: -0.0041678, -0.0002401, -0.0040751, -0.0002306, -0.0036284, 0.0036159
5: 0.0043405, 0.0080575, 0.0043315, 0.0079698, -0.0034218, 0.0034337
6: -0.0050784, 0.0096695, -0.0051140, 0.0093215, -0.0135768, 0.0136238
7: -0.0157257, 0.0043596, -0.0152518, 0.0044081, -0.0185544, 0.0184904
8: 0.9781363, 0.9922849, 0.9784701, 0.9923191, -0.0130701, 0.0130250
9: -0.0088840, 0.0039591, -0.0089150, 0.0036561, -0.0118233, 0.0118642

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303736, upper bound: 0.0296275
time: 1.75 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0303472, upper bound: 0.0295974
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: 0.0000640, 0.0062589, 0.0003398, 0.0064373, -0.0057924, 0.0054038
1: 0.0013316, 0.0022265, 0.0013714, 0.0022523, -0.0008368, 0.0007807
2: 0.0108995, 0.0143245, 0.0108009, 0.0141720, -0.0029876, 0.0032025
3: -0.0034077, 0.0001346, -0.0035097, -0.0000231, -0.0030900, 0.0033121
4: -0.0041826, -0.0003479, -0.0040119, -0.0002375, -0.0035856, 0.0033451
5: 0.0044426, 0.0080715, 0.0043381, 0.0079100, -0.0031656, 0.0033932
6: -0.0046735, 0.0097252, -0.0050880, 0.0090842, -0.0125600, 0.0134631
7: -0.0158015, 0.0038082, -0.0149286, 0.0043727, -0.0183356, 0.0171056
8: 0.9780829, 0.9918965, 0.9786979, 0.9922941, -0.0129159, 0.0120496
9: -0.0085314, 0.0040076, -0.0088924, 0.0034494, -0.0109378, 0.0117243

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287761, upper bound: 0.0276437
time: 2.08 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0287683, upper bound: 0.0279813
time: 1.99 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: 0.0002156, 0.0064574, 0.0003125, 0.0065601, -0.0058944, 0.0056426
1: 0.0013534, 0.0022552, 0.0013674, 0.0022701, -0.0008516, 0.0008152
2: 0.0107898, 0.0142407, 0.0107329, 0.0141871, -0.0031196, 0.0032589
3: -0.0035212, 0.0000479, -0.0035800, -0.0000075, -0.0032265, 0.0033705
4: -0.0040888, -0.0002251, -0.0040289, -0.0001614, -0.0036488, 0.0034928
5: 0.0043263, 0.0079828, 0.0042661, 0.0079260, -0.0033054, 0.0034530
6: -0.0051347, 0.0093729, -0.0053736, 0.0091477, -0.0131148, 0.0137003
7: -0.0153219, 0.0044363, -0.0150151, 0.0047616, -0.0186586, 0.0178613
8: 0.9784209, 0.9923389, 0.9786369, 0.9925681, -0.0131435, 0.0125819
9: -0.0089330, 0.0037009, -0.0091411, 0.0035047, -0.0114210, 0.0119308

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0291673, upper bound: 0.0285461
time: 1.80 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0285461
time: 1.92 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0001607, 0.0063992, 0.0001688, 0.0062372, -0.0055778, 0.0058068
1: 0.0013455, 0.0022468, 0.0013467, 0.0022234, -0.0008058, 0.0008389
2: 0.0108219, 0.0142710, 0.0109115, 0.0142665, -0.0032104, 0.0030838
3: -0.0034879, 0.0000793, -0.0033953, 0.0000747, -0.0033204, 0.0031894
4: -0.0041228, -0.0002611, -0.0041178, -0.0003613, -0.0034527, 0.0035945
5: 0.0043604, 0.0080149, 0.0044553, 0.0080102, -0.0034016, 0.0032675
6: -0.0049994, 0.0095005, -0.0046230, 0.0094816, -0.0134967, 0.0129643
7: -0.0154956, 0.0042520, -0.0154699, 0.0037395, -0.0176563, 0.0183813
8: 0.9782984, 0.9922091, 0.9783166, 0.9918480, -0.0124375, 0.0129482
9: -0.0088152, 0.0038120, -0.0084875, 0.0037955, -0.0117535, 0.0112899

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0285343, upper bound: 0.0279465
time: 1.78 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0289302, upper bound: 0.0279465
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0001320, 0.0065073, 0.0003282, 0.0064542, -0.0059196, 0.0058057
1: 0.0013414, 0.0022624, 0.0013697, 0.0022547, -0.0008552, 0.0008388
2: 0.0107622, 0.0142869, 0.0107915, 0.0141784, -0.0032098, 0.0032728
3: -0.0035497, 0.0000957, -0.0035194, -0.0000165, -0.0033198, 0.0033849
4: -0.0041406, -0.0001942, -0.0040191, -0.0002270, -0.0036643, 0.0035938
5: 0.0042971, 0.0080317, 0.0043282, 0.0079168, -0.0034010, 0.0034677
6: -0.0052506, 0.0095672, -0.0051273, 0.0091111, -0.0134941, 0.0137587
7: -0.0155864, 0.0045942, -0.0149653, 0.0044262, -0.0187382, 0.0183777
8: 0.9782345, 0.9924501, 0.9786721, 0.9923318, -0.0131996, 0.0129457
9: -0.0090340, 0.0038700, -0.0089266, 0.0034728, -0.0117512, 0.0119817

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0281807
time: 1.97 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0285237
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0000223, 0.0062702, 0.0001938, 0.0064853, -0.0059669, 0.0055269
1: 0.0013191, 0.0022282, 0.0013503, 0.0022592, -0.0008620, 0.0007985
2: 0.0108932, 0.0143722, 0.0107743, 0.0142527, -0.0030557, 0.0032989
3: -0.0034142, 0.0001840, -0.0035372, 0.0000603, -0.0031604, 0.0034119
4: -0.0042361, -0.0003409, -0.0041023, -0.0002077, -0.0036936, 0.0034213
5: 0.0044360, 0.0081221, 0.0043099, 0.0079955, -0.0032377, 0.0034954
6: -0.0046996, 0.0099259, -0.0051997, 0.0094235, -0.0128461, 0.0138686
7: -0.0160750, 0.0038438, -0.0153906, 0.0045249, -0.0188879, 0.0174953
8: 0.9778903, 0.9919215, 0.9783724, 0.9924012, -0.0133050, 0.0123241
9: -0.0085542, 0.0041824, -0.0089897, 0.0037448, -0.0111870, 0.0120774

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294146, upper bound: 0.0298201
time: 1.71 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295424, upper bound: 0.0298073
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0001411, 0.0064676, 0.0001617, 0.0066014, -0.0060657, 0.0057939
1: 0.0013427, 0.0022567, 0.0013457, 0.0022760, -0.0008763, 0.0008370
2: 0.0107841, 0.0142819, 0.0107101, 0.0142705, -0.0032033, 0.0033536
3: -0.0035270, 0.0000905, -0.0036035, 0.0000787, -0.0033130, 0.0034684
4: -0.0041349, -0.0002188, -0.0041222, -0.0001359, -0.0037548, 0.0035865
5: 0.0043204, 0.0080264, 0.0042420, 0.0080143, -0.0033940, 0.0035533
6: -0.0051584, 0.0095461, -0.0054694, 0.0094983, -0.0134665, 0.0140983
7: -0.0155576, 0.0044685, -0.0154925, 0.0048921, -0.0192007, 0.0183402
8: 0.9782547, 0.9923616, 0.9783006, 0.9926599, -0.0135254, 0.0129192
9: -0.0089537, 0.0038516, -0.0092245, 0.0038100, -0.0117272, 0.0122774

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0304494, upper bound: 0.0306888
time: 1.99 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306659, upper bound: 0.0306856
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0000984, 0.0062509, 0.0001980, 0.0064519, -0.0060774, 0.0055426
1: 0.0013081, 0.0022254, 0.0013509, 0.0022544, -0.0008780, 0.0008008
2: 0.0109039, 0.0144143, 0.0107927, 0.0142504, -0.0030644, 0.0033601
3: -0.0034031, 0.0002275, -0.0035181, 0.0000580, -0.0031693, 0.0034751
4: -0.0042832, -0.0003529, -0.0040997, -0.0002284, -0.0037620, 0.0034310
5: 0.0044473, 0.0081667, 0.0043295, 0.0079930, -0.0032469, 0.0035601
6: -0.0046548, 0.0101028, -0.0051221, 0.0094137, -0.0128826, 0.0141256
7: -0.0163159, 0.0037827, -0.0153774, 0.0044191, -0.0192379, 0.0175450
8: 0.9777207, 0.9918786, 0.9783816, 0.9923269, -0.0135515, 0.0123591
9: -0.0085151, 0.0043365, -0.0089221, 0.0037364, -0.0112188, 0.0123012

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0294146, upper bound: 0.0298138
time: 1.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0295424, upper bound: 0.0298044
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0000693, 0.0064352, 0.0001675, 0.0065688, -0.0060808, 0.0058963
1: 0.0013323, 0.0022520, 0.0013465, 0.0022713, -0.0008785, 0.0008518
2: 0.0108020, 0.0143216, 0.0107281, 0.0142673, -0.0032599, 0.0033619
3: -0.0035085, 0.0001316, -0.0035849, 0.0000754, -0.0033715, 0.0034771
4: -0.0041794, -0.0002388, -0.0041186, -0.0001561, -0.0037641, 0.0036499
5: 0.0043393, 0.0080685, 0.0042610, 0.0080109, -0.0034540, 0.0035621
6: -0.0050832, 0.0097130, -0.0053937, 0.0094848, -0.0137046, 0.0141335
7: -0.0157850, 0.0043662, -0.0154741, 0.0047891, -0.0192486, 0.0186645
8: 0.9780946, 0.9922895, 0.9783136, 0.9925874, -0.0135591, 0.0131476
9: -0.0088882, 0.0039970, -0.0091586, 0.0037982, -0.0119346, 0.0123081

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306687, upper bound: 0.0304483
time: 1.84 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0306653, upper bound: 0.0306653
time: 1.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.32 seconds
NS_A1_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0282079, upper bound: 0.0275100
NS_A1_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0284344, upper bound: 0.0274853
NS_A1_B1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0284426, upper bound: 0.0272943
NS_A1_B1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0284345, upper bound: 0.0274814
NS_A1_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0286042, upper bound: 0.0287163
NS_A1_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0286247, upper bound: 0.0286254
NS_A1_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0295060, upper bound: 0.0292518
NS_A1_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0294670, upper bound: 0.0293546
NS_A1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0286067, upper bound: 0.0286041
NS_A1_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0287049, upper bound: 0.0285153
NS_A1_B1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0295060, upper bound: 0.0292608
NS_A1_B1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0294670, upper bound: 0.0293495
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0272751, upper bound: 0.0283071
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0272682, upper bound: 0.0287644
NS_A1_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0277895, upper bound: 0.0293535
NS_A1_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0278122, upper bound: 0.0293400
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0272558, upper bound: 0.0283071
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0272500, upper bound: 0.0287646
NS_A1_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0277937, upper bound: 0.0293535
NS_A1_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0277948, upper bound: 0.0293400
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0289395, upper bound: 0.0293000
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0289194, upper bound: 0.0296123
NS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0299145, upper bound: 0.0304469
NS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0298897, upper bound: 0.0304096
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0289210, upper bound: 0.0293000
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0289031, upper bound: 0.0296124
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0298821, upper bound: 0.0301420
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0298699, upper bound: 0.0304096
NS_A2_B1_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0285691, upper bound: 0.0270908
NS_A2_B1_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0285432, upper bound: 0.0272372
NS_A2_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0292621, upper bound: 0.0276150
NS_A2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0292594, upper bound: 0.0277506
NS_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0282621, upper bound: 0.0272476
NS_A2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0286969, upper bound: 0.0272445
NS_A2_B1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0292621, upper bound: 0.0276188
NS_A2_B1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0292594, upper bound: 0.0277381
NS_A2_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0293824, upper bound: 0.0288452
NS_A2_B1_B2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0293527, upper bound: 0.0288772
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0303739, upper bound: 0.0296252
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0303472, upper bound: 0.0296110
NS_A2_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0291525, upper bound: 0.0289128
NS_A2_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0293527, upper bound: 0.0288775
NS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0303736, upper bound: 0.0296275
NS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0303472, upper bound: 0.0295974
NS_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0287761, upper bound: 0.0276437
NS_A2_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0287683, upper bound: 0.0279813
NS_A2_B2_B1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0291673, upper bound: 0.0285461
NS_A2_B2_B1_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0285461
NS_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0285343, upper bound: 0.0279465
NS_A2_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0289302, upper bound: 0.0279465
NS_A2_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0281807
NS_A2_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0296005, upper bound: 0.0285237
NS_A2_B2_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0294146, upper bound: 0.0298201
NS_A2_B2_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0295424, upper bound: 0.0298073
NS_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0304494, upper bound: 0.0306888
NS_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0306659, upper bound: 0.0306856
NS_A2_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0294146, upper bound: 0.0298138
NS_A2_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0295424, upper bound: 0.0298044
NS_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0306687, upper bound: 0.0304483
NS_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 8, lower bound: -0.0306653, upper bound: 0.0306653

## BFS NS instance: NS_A1_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0002928, 0.0063941, 0.0000541, 0.0065257, -0.0056816, 0.0057134
1: 0.0013646, 0.0022461, 0.0013301, 0.0022651, -0.0008208, 0.0008254
2: 0.0108247, 0.0141980, 0.0107520, 0.0143300, -0.0031588, 0.0031412
3: -0.0034850, 0.0000038, -0.0035603, 0.0001403, -0.0032670, 0.0032488
4: -0.0040410, -0.0002642, -0.0041888, -0.0001827, -0.0035170, 0.0035367
5: 0.0043634, 0.0079375, 0.0042863, 0.0080774, -0.0033469, 0.0033283
6: -0.0049877, 0.0091935, -0.0052936, 0.0097483, -0.0132795, 0.0132056
7: -0.0150774, 0.0042361, -0.0158331, 0.0046527, -0.0179849, 0.0180855
8: 0.9785931, 0.9921979, 0.9780607, 0.9924914, -0.0126689, 0.0127398
9: -0.0088051, 0.0035446, -0.0090714, 0.0040277, -0.0115644, 0.0115000

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0269430, upper bound: 0.0278297
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0267762, upper bound: 0.0271136
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0004161, 0.0066603, 0.0001775, 0.0065125, -0.0056204, 0.0058970
1: 0.0013824, 0.0022845, 0.0013479, 0.0022632, -0.0008120, 0.0008519
2: 0.0106775, 0.0141298, 0.0107592, 0.0142617, -0.0032603, 0.0031074
3: -0.0036372, -0.0000667, -0.0035527, 0.0000697, -0.0033720, 0.0032138
4: -0.0039647, -0.0000994, -0.0041124, -0.0001909, -0.0034791, 0.0036503
5: 0.0042074, 0.0078653, 0.0042940, 0.0080050, -0.0034545, 0.0032924
6: -0.0056064, 0.0089069, -0.0052629, 0.0094613, -0.0137063, 0.0130633
7: -0.0146871, 0.0050788, -0.0154423, 0.0046110, -0.0177911, 0.0186667
8: 0.9788680, 0.9927915, 0.9783360, 0.9924619, -0.0125324, 0.0131492
9: -0.0093439, 0.0032950, -0.0090447, 0.0037778, -0.0119360, 0.0113761

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272552, upper bound: 0.0269981
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0263138, upper bound: 0.0268005
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0003135, 0.0063573, 0.0002197, 0.0067783, -0.0058370, 0.0057079
1: 0.0013676, 0.0022407, 0.0013540, 0.0023016, -0.0008433, 0.0008246
2: 0.0108451, 0.0141866, 0.0106123, 0.0142384, -0.0031558, 0.0032272
3: -0.0034640, -0.0000081, -0.0037047, 0.0000456, -0.0032639, 0.0033377
4: -0.0040282, -0.0002870, -0.0040863, -0.0000264, -0.0036132, 0.0035333
5: 0.0043850, 0.0079254, 0.0041384, 0.0079804, -0.0033437, 0.0034193
6: -0.0049021, 0.0091454, -0.0058805, 0.0093635, -0.0132668, 0.0135669
7: -0.0150120, 0.0041195, -0.0153089, 0.0054521, -0.0184769, 0.0180683
8: 0.9786391, 0.9921157, 0.9784300, 0.9930544, -0.0130155, 0.0127277
9: -0.0087305, 0.0035027, -0.0095825, 0.0036926, -0.0115533, 0.0118147

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0263818, upper bound: 0.0274696
time: 1.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0262914, upper bound: 0.0266864
time: 2.00 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0001670, 0.0064644, 0.0003586, 0.0064704, -0.0058194, 0.0056031
1: 0.0013464, 0.0022562, 0.0013741, 0.0022571, -0.0008407, 0.0008095
2: 0.0107859, 0.0142675, 0.0107825, 0.0141616, -0.0030978, 0.0032174
3: -0.0035252, 0.0000757, -0.0035287, -0.0000339, -0.0032039, 0.0033276
4: -0.0041189, -0.0002207, -0.0040003, -0.0002170, -0.0036023, 0.0034684
5: 0.0043222, 0.0080112, 0.0043187, 0.0078990, -0.0032823, 0.0034090
6: -0.0051510, 0.0094858, -0.0051650, 0.0090405, -0.0130231, 0.0135258
7: -0.0154755, 0.0044585, -0.0148691, 0.0044776, -0.0184209, 0.0177363
8: 0.9783126, 0.9923546, 0.9787397, 0.9923680, -0.0129761, 0.0124938
9: -0.0089472, 0.0037991, -0.0089595, 0.0034114, -0.0113411, 0.0117788

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0270793, upper bound: 0.0270244
time: 1.74 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0269290, upper bound: 0.0263466
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0002794, 0.0064531, 0.0004604, 0.0067391, -0.0061732, 0.0056046
1: 0.0013627, 0.0022546, 0.0013888, 0.0022959, -0.0008918, 0.0008097
2: 0.0107921, 0.0142054, 0.0106340, 0.0141053, -0.0030986, 0.0034130
3: -0.0035188, 0.0000114, -0.0036823, -0.0000921, -0.0032048, 0.0035299
4: -0.0040493, -0.0002277, -0.0039373, -0.0000507, -0.0038213, 0.0034693
5: 0.0043288, 0.0079454, 0.0041613, 0.0078393, -0.0032832, 0.0036162
6: -0.0051248, 0.0092246, -0.0057895, 0.0088038, -0.0130266, 0.0143482
7: -0.0151198, 0.0044229, -0.0145468, 0.0053281, -0.0195410, 0.0177411
8: 0.9785631, 0.9923294, 0.9789668, 0.9929671, -0.0137651, 0.0124972
9: -0.0089244, 0.0035716, -0.0095033, 0.0032052, -0.0113442, 0.0124950

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268618, upper bound: 0.0269387
time: 1.77 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0265885, upper bound: 0.0258970
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0000998, 0.0064319, 0.0003645, 0.0064344, -0.0058352, 0.0057041
1: 0.0013367, 0.0022515, 0.0013750, 0.0022519, -0.0008430, 0.0008241
2: 0.0108038, 0.0143047, 0.0108024, 0.0141584, -0.0031537, 0.0032261
3: -0.0035066, 0.0001141, -0.0035081, -0.0000372, -0.0032617, 0.0033366
4: -0.0041605, -0.0002408, -0.0039967, -0.0002393, -0.0036121, 0.0035309
5: 0.0043412, 0.0080506, 0.0043398, 0.0078955, -0.0033415, 0.0034183
6: -0.0050755, 0.0096420, -0.0050814, 0.0090269, -0.0132580, 0.0135627
7: -0.0156883, 0.0043557, -0.0148505, 0.0043637, -0.0184712, 0.0180562
8: 0.9781627, 0.9922821, 0.9787529, 0.9922878, -0.0130115, 0.0127192
9: -0.0088815, 0.0039352, -0.0088866, 0.0033995, -0.0115456, 0.0118110

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0269872, upper bound: 0.0269761
time: 1.85 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0268649, upper bound: 0.0263369
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0002112, 0.0064199, 0.0004662, 0.0067010, -0.0061592, 0.0056099
1: 0.0013528, 0.0022498, 0.0013897, 0.0022904, -0.0008898, 0.0008105
2: 0.0108105, 0.0142431, 0.0106550, 0.0141021, -0.0031015, 0.0034053
3: -0.0034998, 0.0000504, -0.0036605, -0.0000954, -0.0032078, 0.0035219
4: -0.0040916, -0.0002482, -0.0039337, -0.0000742, -0.0038126, 0.0034726
5: 0.0043483, 0.0079853, 0.0041836, 0.0078359, -0.0032862, 0.0036080
6: -0.0050477, 0.0093832, -0.0057010, 0.0087904, -0.0130389, 0.0143156
7: -0.0153358, 0.0043178, -0.0145284, 0.0052076, -0.0194967, 0.0177578
8: 0.9784110, 0.9922554, 0.9789798, 0.9928823, -0.0137339, 0.0125090
9: -0.0088572, 0.0037098, -0.0094262, 0.0031935, -0.0113548, 0.0124667

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0267301, upper bound: 0.0268752
time: 1.78 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0265128, upper bound: 0.0258828
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: 0.0002762, 0.0064555, 0.0001731, 0.0066001, -0.0059254, 0.0057395
1: 0.0013622, 0.0022549, 0.0013473, 0.0022758, -0.0008560, 0.0008292
2: 0.0107908, 0.0142071, 0.0107108, 0.0142641, -0.0031732, 0.0032760
3: -0.0035201, 0.0000132, -0.0036028, 0.0000722, -0.0032819, 0.0033882
4: -0.0040513, -0.0002262, -0.0041151, -0.0001367, -0.0036679, 0.0035528
5: 0.0043274, 0.0079472, 0.0042427, 0.0080076, -0.0033622, 0.0034711
6: -0.0051303, 0.0092320, -0.0054664, 0.0094716, -0.0133401, 0.0137723
7: -0.0151299, 0.0044303, -0.0154562, 0.0048881, -0.0187566, 0.0181681
8: 0.9785561, 0.9923347, 0.9783261, 0.9926572, -0.0132126, 0.0127980
9: -0.0089292, 0.0035781, -0.0092219, 0.0037868, -0.0116172, 0.0119935

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0275852, upper bound: 0.0282056
time: 2.31 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0273455, upper bound: 0.0271936
time: 1.63 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: 0.0003542, 0.0067992, 0.0002874, 0.0065879, -0.0059321, 0.0059985
1: 0.0013735, 0.0023046, 0.0013638, 0.0022741, -0.0008570, 0.0008666
2: 0.0106007, 0.0141640, 0.0107176, 0.0142010, -0.0033164, 0.0032797
3: -0.0037167, -0.0000314, -0.0035958, 0.0000068, -0.0034300, 0.0033920
4: -0.0040030, -0.0000134, -0.0040444, -0.0001443, -0.0036721, 0.0037131
5: 0.0041261, 0.0079015, 0.0042499, 0.0079407, -0.0035139, 0.0034750
6: -0.0059293, 0.0090507, -0.0054381, 0.0092059, -0.0139421, 0.0137878
7: -0.0148829, 0.0055185, -0.0150944, 0.0048495, -0.0187778, 0.0189879
8: 0.9787301, 0.9931012, 0.9785811, 0.9926299, -0.0132275, 0.0133755
9: -0.0096250, 0.0034202, -0.0091973, 0.0035554, -0.0121414, 0.0120070

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0279377, upper bound: 0.0271131
time: 1.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271191, upper bound: 0.0270176
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0000813, 0.0064340, 0.0002978, 0.0065546, -0.0060193, 0.0057531
1: 0.0013341, 0.0022518, 0.0013653, 0.0022692, -0.0008696, 0.0008312
2: 0.0108027, 0.0143149, 0.0107360, 0.0141952, -0.0031807, 0.0033279
3: -0.0035078, 0.0001247, -0.0035768, 0.0000009, -0.0032897, 0.0034419
4: -0.0041719, -0.0002396, -0.0040379, -0.0001649, -0.0037260, 0.0035613
5: 0.0043400, 0.0080614, 0.0042694, 0.0079346, -0.0033702, 0.0035261
6: -0.0050803, 0.0096849, -0.0053607, 0.0091817, -0.0133718, 0.0139904
7: -0.0157467, 0.0043622, -0.0150614, 0.0047441, -0.0190538, 0.0182113
8: 0.9781215, 0.9922866, 0.9786043, 0.9925556, -0.0134219, 0.0128284
9: -0.0088857, 0.0039725, -0.0091298, 0.0035343, -0.0116448, 0.0121835

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0272329, upper bound: 0.0278586
time: 1.96 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271488, upper bound: 0.0272488
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0001910, 0.0064225, 0.0003834, 0.0068506, -0.0062730, 0.0056745
1: 0.0013499, 0.0022502, 0.0013777, 0.0023120, -0.0009063, 0.0008198
2: 0.0108091, 0.0142543, 0.0105723, 0.0141479, -0.0031373, 0.0034681
3: -0.0035012, 0.0000620, -0.0037461, -0.0000481, -0.0032447, 0.0035869
4: -0.0041041, -0.0002467, -0.0039849, 0.0000184, -0.0038831, 0.0035126
5: 0.0043468, 0.0079972, 0.0040960, 0.0078844, -0.0033241, 0.0036747
6: -0.0050536, 0.0094302, -0.0060488, 0.0089828, -0.0131892, 0.0145801
7: -0.0153998, 0.0043258, -0.0147905, 0.0056812, -0.0198568, 0.0179625
8: 0.9783660, 0.9922611, 0.9787952, 0.9932159, -0.0139875, 0.0126532
9: -0.0088624, 0.0037507, -0.0097291, 0.0033611, -0.0114857, 0.0126970

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0271058, upper bound: 0.0279364
time: 1.68 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0269978, upper bound: 0.0269978
time: 1.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.87 seconds
NS_A1_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0269430, upper bound: 0.0278297
NS_A1_B2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0267762, upper bound: 0.0271136
NS_A1_B2_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0272552, upper bound: 0.0269981
NS_A1_B2_A2_B1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0263138, upper bound: 0.0268005
NS_A1_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0263818, upper bound: 0.0274696
NS_A1_B2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0262914, upper bound: 0.0266864
NS_A2_B1_B2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0270793, upper bound: 0.0270244
NS_A2_B1_B2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0269290, upper bound: 0.0263466
NS_A2_B1_B2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0268618, upper bound: 0.0269387
NS_A2_B1_B2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0265885, upper bound: 0.0258970
NS_A2_B1_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0269872, upper bound: 0.0269761
NS_A2_B1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0268649, upper bound: 0.0263369
NS_A2_B1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0267301, upper bound: 0.0268752
NS_A2_B1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0265128, upper bound: 0.0258828
NS_A2_B2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0275852, upper bound: 0.0282056
NS_A2_B2_B2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0273455, upper bound: 0.0271936
NS_A2_B2_B2_A1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0279377, upper bound: 0.0271131
NS_A2_B2_B2_A1_A2_A2_A2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0271191, upper bound: 0.0270176
NS_A2_B2_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0272329, upper bound: 0.0278586
NS_A2_B2_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0271488, upper bound: 0.0272488
NS_A2_B2_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0271058, upper bound: 0.0279364
NS_A2_B2_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 4.87
Output dim: 8, lower bound: -0.0269978, upper bound: 0.0269978

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.42 + 387.18 = 391.60 seconds
