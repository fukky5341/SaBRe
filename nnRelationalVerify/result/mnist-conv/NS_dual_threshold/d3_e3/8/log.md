## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.716269655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.6121473, 1.6121478)
1: (-6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7759919, 1.7759929)
2: (-8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6092749, 1.6092749)
3: (-4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4462447, 1.4462445)
4: (-7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2081947, 1.2081950)
5: (-5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3884630, 1.3884630)
6: (-13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5876408, 1.5876408)
7: (2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2234151, 1.2234151)
8: (-0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.3073392, 1.3073397)
9: (-8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4662395, 1.4662395)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.25 + 35.14 = 58.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.7198690, upper bound: 0.7198687

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6192
type: B, layer: 1, pos: 6192
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6192

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7139732, upper bound: 0.7060259
time: 6.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7198665
time: 3.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.94 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 9.94
Output dim: 7, lower bound: -0.7139732, upper bound: 0.7060259
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.94
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7198665

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -15.3441124, -12.2636013, -15.3441200, -12.2636013, -1.5790257, 1.6121478
1: -6.8066788, -4.8410778, -6.8066797, -4.8410749, -1.7954149, 1.7759895
2: -8.3706245, -6.5761147, -8.3706236, -6.5761118, -1.6105318, 1.6092720
3: -4.6016631, -2.8533654, -4.6016669, -2.8533633, -1.4462419, 1.4584467
4: -7.5307083, -5.6608071, -7.5307088, -5.6608067, -1.2146811, 1.2081935
5: -5.9167213, -4.1329212, -5.9167237, -4.1329203, -1.3884611, 1.3954849
6: -13.9713306, -11.5294218, -13.9713326, -11.5294180, -1.5873899, 1.5712531
7: 2.7536249, 4.5407906, 2.7536235, 4.5407939, -1.2234147, 1.2108374
8: -0.9690351, 0.6157970, -0.9690433, 0.6157956, -1.2933931, 1.3073382
9: -8.3658695, -6.1971216, -8.3658676, -6.1971173, -1.4662380, 1.4428995

Time for backsubstitution: 20.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6140
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 6140

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178370, upper bound: 0.7197014
time: 4.00 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198631, upper bound: 0.7198639
time: 3.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.35 seconds
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 7, lower bound: -0.7178370, upper bound: 0.7197014
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.35
Output dim: 7, lower bound: -0.7198631, upper bound: 0.7198639

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -15.3439302, -12.2638750, -15.3435421, -12.2644444, -1.5774665, 1.6106584
1: -6.8061442, -4.8435931, -6.8049994, -4.8489552, -1.7873373, 1.7724462
2: -8.3701267, -6.5763493, -8.3690681, -6.5768661, -1.6054211, 1.6037064
3: -4.5989199, -2.8537016, -4.5930700, -2.8544338, -1.4422398, 1.4493709
4: -7.5305362, -5.6641378, -7.5301619, -5.6712494, -1.2031505, 1.2037401
5: -5.9159517, -4.1332560, -5.9143105, -4.1339850, -1.3861942, 1.3921430
6: -13.9684763, -11.5297985, -13.9623804, -11.5306273, -1.5834899, 1.5618224
7: 2.7540579, 4.5394478, 2.7549934, 4.5365868, -1.2187912, 1.2085035
8: -0.9681997, 0.6156740, -0.9663873, 0.6154137, -1.2889409, 1.3017378
9: -8.3631239, -6.1973786, -8.3572588, -6.1979313, -1.4627295, 1.4342251

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196951
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7196954
time: 3.91 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -15.3441105, -12.2636023, -15.3475046, -12.2619343, -1.5835047, 1.6156976
1: -6.8066750, -4.8410883, -6.8196139, -4.8365116, -1.8007860, 1.7934942
2: -8.3706226, -6.5761166, -8.3747339, -6.5661340, -1.6185446, 1.6195521
3: -4.6016459, -2.8533671, -4.6060247, -2.8321295, -1.4670641, 1.4604545
4: -7.5307059, -5.6608262, -7.5664654, -5.6587439, -1.2158251, 1.2320068
5: -5.9167180, -4.1329231, -5.9206352, -4.1107988, -1.4084187, 1.3989754
6: -13.9713240, -11.5294237, -13.9755354, -11.4980650, -1.6040568, 1.5729337
7: 2.7536263, 4.5407887, 2.7416630, 4.5431023, -1.2254453, 1.2259848
8: -0.9690318, 0.6157951, -0.9801741, 0.6182160, -1.2935867, 1.3198097
9: -8.3658571, -6.1971231, -8.3709106, -6.1738162, -1.4732871, 1.4503047

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: B, layer: 1, pos: 4612
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184429, upper bound: 0.7198556
time: 3.90 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198569, upper bound: 0.7198554
time: 3.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.85 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196951
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7196954
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 7, lower bound: -0.7184429, upper bound: 0.7198556
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 7, lower bound: -0.7198569, upper bound: 0.7198554

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -15.3424158, -12.2680149, -15.3428192, -12.2664776, -1.5724225, 1.6023424
1: -6.7889385, -4.8441677, -6.7967138, -4.8492274, -1.7697668, 1.7635226
2: -8.3695574, -6.5814209, -8.3688011, -6.5793099, -1.6016855, 1.5969667
3: -4.5975628, -2.8573480, -4.5924311, -2.8561919, -1.4390917, 1.4451995
4: -7.5287752, -5.6680717, -7.5293379, -5.6731520, -1.1970603, 1.1951411
5: -5.9151616, -4.1339369, -5.9139409, -4.1343079, -1.3849840, 1.3907747
6: -13.9624310, -11.5299759, -13.9594641, -11.5307102, -1.5746651, 1.5573750
7: 2.7578225, 4.5382862, 2.7568083, 4.5360394, -1.2144251, 1.2056212
8: -0.9675474, 0.6150379, -0.9660773, 0.6151161, -1.2867670, 1.2984719
9: -8.3501511, -6.1976094, -8.3510084, -6.1980386, -1.4495149, 1.4276538

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178295
time: 3.73 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196951
time: 3.70 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.3558578, -12.2551823, -15.3435402, -12.2644491, -1.5912590, 1.6173539
1: -6.8102570, -4.8182430, -6.8049688, -4.8489561, -1.7875490, 1.7968621
2: -8.3802795, -6.5734200, -8.3690681, -6.5768690, -1.6158061, 1.6056237
3: -4.6156349, -2.8520093, -4.5930681, -2.8544412, -1.4588981, 1.4511275
4: -7.5444670, -5.6635580, -7.5301595, -5.6712532, -1.2168765, 1.2041445
5: -5.9296131, -4.1306424, -5.9143109, -4.1339860, -1.4018536, 1.3964026
6: -13.9756155, -11.5191450, -13.9623718, -11.5306282, -1.5912108, 1.5723751
7: 2.7518935, 4.5512486, 2.7549992, 4.5365844, -1.2197194, 1.2213359
8: -0.9721460, 0.6229062, -0.9663858, 0.6154141, -1.2955317, 1.3103266
9: -8.3670864, -6.1789989, -8.3572350, -6.1979318, -1.4591351, 1.4483991

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7178296
time: 4.02 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7196953
time: 3.91 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -15.3425980, -12.2677450, -15.3467846, -12.2639713, -1.5784588, 1.6073787
1: -6.7894735, -4.8416619, -6.8113298, -4.8367839, -1.7832193, 1.7845736
2: -8.3700504, -6.5811863, -8.3744678, -6.5685792, -1.6148024, 1.6128035
3: -4.6002846, -2.8570113, -4.6053891, -2.8338871, -1.4635522, 1.4562805
4: -7.5289469, -5.6647596, -7.5656261, -5.6606460, -1.2097359, 1.2233058
5: -5.9159279, -4.1336079, -5.9202633, -4.1111255, -1.4072018, 1.3976011
6: -13.9652834, -11.5296040, -13.9726219, -11.4981537, -1.5952091, 1.5684834
7: 2.7573886, 4.5396242, 2.7434816, 4.5425568, -1.2210813, 1.2231021
8: -0.9683752, 0.6151586, -0.9798594, 0.6179161, -1.2914176, 1.3165178
9: -8.3528852, -6.1973553, -8.3646641, -6.1739302, -1.4600410, 1.4437318

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178294
time: 6.46 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178317
time: 3.46 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.3560371, -12.2549124, -15.3475037, -12.2619419, -1.5972996, 1.6223927
1: -6.8107772, -4.8157349, -6.8195820, -4.8365135, -1.8010001, 1.8063488
2: -8.3807898, -6.5731878, -8.3747320, -6.5661373, -1.6289339, 1.6214719
3: -4.6183748, -2.8516788, -4.6060219, -2.8321369, -1.4752493, 1.4622064
4: -7.5446401, -5.6602468, -7.5664630, -5.6587462, -1.2295525, 1.2280771
5: -5.9303780, -4.1303115, -5.9206347, -4.1107988, -1.4240780, 1.4032354
6: -13.9784603, -11.5187702, -13.9755287, -11.4980650, -1.6075845, 1.5834863
7: 2.7514682, 4.5525932, 2.7416706, 4.5431004, -1.2263689, 1.2388210
8: -0.9729805, 0.6230278, -0.9801731, 0.6182160, -1.3001790, 1.3238485
9: -8.3698215, -6.1787434, -8.3708897, -6.1738162, -1.4693298, 1.4647207

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6140
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6140

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7178295
time: 4.33 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7198561
time: 6.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.62 seconds
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178295
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196951
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7178296
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7196953
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178294
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178317
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7178295
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.62
Output dim: 7, lower bound: -0.7178307, upper bound: 0.7198561

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3428192, -12.2664776, -1.5716696, 1.6015203
1: -6.7877836, -4.8495293, -6.7967138, -4.8492274, -1.7690344, 1.7582607
2: -8.3685093, -6.5819416, -8.3688011, -6.5793099, -1.5987153, 1.5944386
3: -4.5917115, -2.8580873, -4.5924311, -2.8561919, -1.4331861, 1.4443502
4: -7.5284009, -5.6751847, -7.5293379, -5.6731520, -1.1964681, 1.1874709
5: -5.9135199, -4.1346598, -5.9139409, -4.1343079, -1.3829913, 1.3898654
6: -13.9563313, -11.5308027, -13.9594641, -11.5307102, -1.5685105, 1.5566778
7: 2.7587647, 4.5354218, 2.7568083, 4.5360394, -1.2137680, 1.2026796
8: -0.9657359, 0.6147766, -0.9660773, 0.6151161, -1.2835712, 1.2964497
9: -8.3442841, -6.1981640, -8.3510084, -6.1980386, -1.4437900, 1.4270973

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163956
time: 3.76 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178295
time: 4.09 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3428192, -12.2664776, -1.5763249, 1.6050286
1: -6.8023996, -4.8370848, -6.7967138, -4.8492274, -1.7876205, 1.7715178
2: -8.3741760, -6.5712104, -8.3688011, -6.5793099, -1.6037526, 1.6061769
3: -4.6046753, -2.8357792, -4.5924311, -2.8561919, -1.4460082, 1.4610846
4: -7.5646691, -5.6626768, -7.5293379, -5.6731520, -1.2175696, 1.2001171
5: -5.9198356, -4.1114826, -5.9139409, -4.1343079, -1.3899450, 1.4111490
6: -13.9694901, -11.4982491, -13.9594641, -11.5307102, -1.5809879, 1.5758543
7: 2.7454405, 4.5419421, 2.7568083, 4.5360394, -1.2298746, 1.2091002
8: -0.9795127, 0.6175756, -0.9660773, 0.6151161, -1.2968717, 1.2996368
9: -8.3579407, -6.1740561, -8.3510084, -6.1980386, -1.4622059, 1.4406450

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7182737
time: 3.68 seconds

## Relational analysis of NS_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196950
time: 3.87 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -15.3554678, -12.2557526, -15.3435402, -12.2644491, -1.5904965, 1.6165302
1: -6.8091316, -4.8236084, -6.8049688, -4.8489561, -1.7868176, 1.7919703
2: -8.3791914, -6.5739355, -8.3690681, -6.5768690, -1.6128078, 1.6030879
3: -4.6097465, -2.8527339, -4.5930681, -2.8544412, -1.4528279, 1.4502730
4: -7.5440922, -5.6706729, -7.5301595, -5.6712532, -1.2162809, 1.1964748
5: -5.9279718, -4.1313696, -5.9143109, -4.1339860, -1.3998637, 1.3954825
6: -13.9695120, -11.5199776, -13.9623718, -11.5306282, -1.5850334, 1.5716836
7: 2.7528129, 4.5483732, 2.7549992, 4.5365844, -1.2190747, 1.2183852
8: -0.9703269, 0.6226459, -0.9663858, 0.6154141, -1.2923245, 1.3083043
9: -8.3612194, -6.1795549, -8.3572350, -6.1979318, -1.4534116, 1.4478517

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7163957
time: 4.25 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7163958
time: 4.54 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -15.3594093, -12.2532406, -15.3435402, -12.2644491, -1.5951252, 1.6200428
1: -6.8236847, -4.8111382, -6.8049688, -4.8489561, -1.8053846, 1.8032374
2: -8.3849325, -6.5632133, -8.3690681, -6.5768690, -1.6179252, 1.6148033
3: -4.6227713, -2.8304629, -4.5930681, -2.8544412, -1.4656458, 1.4670398
4: -7.5803432, -5.6581745, -7.5301595, -5.6712532, -1.2289784, 1.2091205
5: -5.9342909, -4.1082211, -5.9143109, -4.1339860, -1.4068074, 1.4158933
6: -13.9826975, -11.4874382, -13.9623718, -11.5306282, -1.5971541, 1.5825899
7: 2.7395205, 4.5549111, 2.7549992, 4.5365844, -1.2351522, 1.2248182
8: -0.9840770, 0.6254587, -0.9663858, 0.6154141, -1.3055911, 1.3114910
9: -8.3748693, -6.1554675, -8.3572350, -6.1979318, -1.4718380, 1.4499512

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7182739
time: 4.15 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7182738
time: 3.94 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3467846, -12.2639713, -1.5751772, 1.6061776
1: -6.7877836, -4.8495293, -6.8113298, -4.8367839, -1.7822895, 1.7768459
2: -8.3685093, -6.5819416, -8.3744678, -6.5685792, -1.6104579, 1.5994759
3: -4.5917115, -2.8580873, -4.6053891, -2.8338871, -1.4549410, 1.4571803
4: -7.5284009, -5.6751847, -7.5656261, -5.6606460, -1.2091155, 1.2120469
5: -5.9135199, -4.1346598, -5.9202633, -4.1111255, -1.4042821, 1.3968227
6: -13.9563313, -11.5308027, -13.9726219, -11.4981537, -1.5861568, 1.5691535
7: 2.7587647, 4.5354218, 2.7434816, 4.5425568, -1.2201874, 1.2187865
8: -0.9657359, 0.6147766, -0.9798594, 0.6179161, -1.2867470, 1.3097467
9: -8.3442841, -6.1981640, -8.3646641, -6.1739302, -1.4516759, 1.4455142

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163954
time: 5.97 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178292
time: 6.14 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3467846, -12.2639713, -1.5807381, 1.6105890
1: -6.8023996, -4.8370848, -6.8113298, -4.8367839, -1.7997894, 1.7890162
2: -8.3741760, -6.5712104, -8.3744678, -6.5685792, -1.6249228, 1.6206532
3: -4.6046753, -2.8357792, -4.6053891, -2.8338871, -1.4518557, 1.4630253
4: -7.5646691, -5.6626768, -7.5656261, -5.6606460, -1.2171502, 1.2081544
5: -5.9198356, -4.1114826, -5.9202633, -4.1111255, -1.4070096, 1.4138815
6: -13.9694901, -11.4982491, -13.9726219, -11.4981537, -1.5969582, 1.5883379
7: 2.7454405, 4.5419421, 2.7434816, 4.5425568, -1.2358017, 1.2247155
8: -0.9795127, 0.6175756, -0.9798594, 0.6179161, -1.3109059, 1.3202927
9: -8.3579407, -6.1740561, -8.3646641, -6.1739302, -1.4659629, 1.4492681

Time for backsubstitution: 21.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163980
time: 3.54 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178317
time: 3.59 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -15.3554678, -12.2557526, -15.3475037, -12.2619419, -1.5940037, 1.6211896
1: -6.8091316, -4.8236084, -6.8195820, -4.8365135, -1.8000712, 1.7985506
2: -8.3791914, -6.5739355, -8.3747320, -6.5661373, -1.6245508, 1.6081257
3: -4.6097465, -2.8527339, -4.6060219, -2.8321369, -1.4664488, 1.4631071
4: -7.5440922, -5.6706729, -7.5664630, -5.6587462, -1.2289298, 1.2168176
5: -5.9279718, -4.1313696, -5.9206347, -4.1107988, -1.4211612, 1.4024434
6: -13.9695120, -11.5199776, -13.9755287, -11.4980650, -1.5985174, 1.5812130
7: 2.7528129, 4.5483732, 2.7416706, 4.5431004, -1.2254932, 1.2344933
8: -0.9703269, 0.6226459, -0.9801731, 0.6182160, -1.2954917, 1.3193190
9: -8.3612194, -6.1795549, -8.3708897, -6.1738162, -1.4609671, 1.4638500

Time for backsubstitution: 21.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7163957
time: 4.08 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7163957
time: 4.78 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -15.3594093, -12.2532406, -15.3475037, -12.2619419, -1.5995774, 1.6255941
1: -6.8236847, -4.8111382, -6.8195820, -4.8365135, -1.8175511, 1.8117704
2: -8.3849325, -6.5632133, -8.3747320, -6.5661373, -1.6389894, 1.6293101
3: -4.6227713, -2.8304629, -4.6060219, -2.8321369, -1.4714961, 1.4688773
4: -7.5803432, -5.6581745, -7.5664630, -5.6587462, -1.2369978, 1.2171566
5: -5.9342909, -4.1082211, -5.9206347, -4.1107988, -1.4238796, 1.4194489
6: -13.9826975, -11.4874382, -13.9755287, -11.4980650, -1.6093564, 1.5950727
7: 2.7395205, 4.5549111, 2.7416706, 4.5431004, -1.2410784, 1.2404339
8: -0.9840770, 0.6254587, -0.9801731, 0.6182160, -1.3196154, 1.3276157
9: -8.3748693, -6.1554675, -8.3708897, -6.1738162, -1.4755979, 1.4686193

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7184415
time: 5.49 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7184415
time: 5.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 32.60 seconds
NS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163956
NS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178295
NS_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7182737
NS_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7196950
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7163957
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7163958
NS_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7182739
NS_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7182738
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163954
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178292
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7163980
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7163977, upper bound: 0.7178317
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7163957
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7163957
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178306, upper bound: 0.7184415
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.60
Output dim: 7, lower bound: -0.7178308, upper bound: 0.7184415

## BFS NS instance: NS_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3420286, -12.2685862, -1.5676699, 1.6007917
1: -6.7877836, -4.8495293, -6.7877846, -4.8495274, -1.7687197, 1.7492924
2: -8.3685093, -6.5819416, -8.3685112, -6.5819407, -1.5953665, 1.5941081
3: -4.5917115, -2.8580873, -4.5917158, -2.8580861, -1.4313774, 1.4435816
4: -7.5284009, -5.6751847, -7.5284004, -5.6751823, -1.1926539, 1.1861651
5: -5.9135199, -4.1346598, -5.9135218, -4.1346583, -1.3824558, 1.3894794
6: -13.9563313, -11.5308027, -13.9563313, -11.5308018, -1.5683537, 1.5521891
7: 2.7587647, 4.5354218, 2.7587628, 4.5354261, -1.2132366, 1.2006593
8: -0.9657359, 0.6147766, -0.9657435, 0.6147785, -1.2820730, 1.2960186
9: -8.3442841, -6.1981640, -8.3442841, -6.1981578, -1.4436541, 1.4203138

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of NS_A2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7123516, upper bound: 0.7150248
time: 6.81 seconds

## Relational analysis of NS_A2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163981, upper bound: 0.7163987
time: 3.58 seconds

## BFS NS instance: NS_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -15.3420238, -12.2685862, -15.3554735, -12.2557507, -1.5771208, 1.6150668
1: -6.7877836, -4.8495293, -6.8091331, -4.8236070, -1.7917266, 1.7700734
2: -8.3685093, -6.5819416, -8.3791924, -6.5739331, -1.6033573, 1.6051188
3: -4.5917115, -2.8580873, -4.6097503, -2.8527324, -1.4374428, 1.4615450
4: -7.5284009, -5.6751847, -7.5440936, -5.6706700, -1.1971402, 1.2024412
5: -5.9135199, -4.1346598, -5.9279737, -4.1313696, -1.3874822, 1.4058709
6: -13.9563313, -11.5308027, -13.9695148, -11.5199738, -1.5766339, 1.5657263
7: 2.7587647, 4.5354218, 2.7528114, 4.5483780, -1.2270708, 1.2061300
8: -0.9657359, 0.6147766, -0.9703321, 0.6226463, -1.2906113, 1.3034182
9: -8.3442841, -6.1981640, -8.3612194, -6.1795516, -1.4506812, 1.4365413

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 451
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 451
type: B, layer: 1, pos: 6192

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 451

## Relational analysis of NS_A2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7123516, upper bound: 0.7027569
time: 6.20 seconds

## Relational analysis of NS_A2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7163981, upper bound: 0.7178324
time: 3.69 seconds

## BFS NS instance: NS_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -15.3459835, -12.2660780, -15.3420286, -12.2685862, -1.5723252, 1.6043000
1: -6.8023996, -4.8370848, -6.7877846, -4.8495274, -1.7873049, 1.7625494
2: -8.3741760, -6.5712104, -8.3685112, -6.5819407, -1.6004038, 1.6058464
3: -4.6046753, -2.8357792, -4.5917158, -2.8580861, -1.4441996, 1.4603200
4: -7.5646691, -5.6626768, -7.5284004, -5.6751823, -1.2147863, 1.1988113
5: -5.9198356, -4.1114826, -5.9135218, -4.1346583, -1.3894095, 1.4107635
6: -13.9694901, -11.4982491, -13.9563313, -11.5308018, -1.5808311, 1.5723925
7: 2.7454405, 4.5419421, 2.7587628, 4.5354261, -1.2293427, 1.2070799
8: -0.9795127, 0.6175756, -0.9657435, 0.6147785, -1.2953734, 1.2992058
9: -8.3579407, -6.1740561, -8.3442841, -6.1981578, -1.4620695, 1.4355869

Time for backsubstitution: 21.02 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.39 + 544.58 = 602.96 seconds
