## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00372996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0055609, 0.0100454, 0.0055609, 0.0100454, -0.0042223, 0.0042223)
1: (0.0014625, 0.0057346, 0.0014625, 0.0057346, -0.0041844, 0.0041844)
2: (-0.0215600, -0.0108682, -0.0215600, -0.0108682, -0.0071858, 0.0071858)
3: (-0.0049446, 0.0043334, -0.0049446, 0.0043334, -0.0081654, 0.0081655)
4: (0.0146063, 0.0160720, 0.0146063, 0.0160720, -0.0014656, 0.0014656)
5: (-0.0082123, 0.0048530, -0.0082123, 0.0048530, -0.0119883, 0.0119883)
6: (0.9919193, 1.0007224, 0.9919193, 1.0007224, -0.0076607, 0.0076607)
7: (0.0130571, 0.0174099, 0.0130571, 0.0174099, -0.0024329, 0.0024329)
8: (0.0033857, 0.0073466, 0.0033857, 0.0073466, -0.0039609, 0.0039609)
9: (-0.0240699, -0.0150880, -0.0240699, -0.0150880, -0.0068805, 0.0068805)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.68 = 3.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0038336, upper bound: 0.0038336

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038004, upper bound: 0.0037624
time: 0.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0038005, upper bound: 0.0038006
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 6, lower bound: -0.0038004, upper bound: 0.0037624
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 6, lower bound: -0.0038005, upper bound: 0.0038006

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056834, 0.0100090, 0.0056101, 0.0100312, -0.0040653, 0.0041020
1: 0.0015792, 0.0057000, 0.0015094, 0.0057211, -0.0040315, 0.0040687
2: -0.0212680, -0.0109549, -0.0214427, -0.0109021, -0.0068572, 0.0069107
3: -0.0048693, 0.0040800, -0.0049151, 0.0042316, -0.0079208, 0.0078535
4: 0.0146290, 0.0160656, 0.0146152, 0.0160695, -0.0014404, 0.0014503
5: -0.0081064, 0.0044963, -0.0081709, 0.0047098, -0.0116400, 0.0115378
6: 0.9919907, 1.0004822, 0.9919473, 1.0006261, -0.0074291, 0.0073665
7: 0.0130982, 0.0173044, 0.0130732, 0.0173675, -0.0023359, 0.0023194
8: 0.0034939, 0.0073145, 0.0034291, 0.0073340, -0.0038401, 0.0038853
9: -0.0238246, -0.0151609, -0.0239714, -0.0151165, -0.0065989, 0.0066510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037297, upper bound: 0.0037153
time: 0.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037629, upper bound: 0.0037219
time: 0.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0056316, 0.0100699, 0.0055900, 0.0100322, -0.0041688, 0.0041070
1: 0.0015299, 0.0057579, 0.0014902, 0.0057220, -0.0041361, 0.0040732
2: -0.0213914, -0.0108098, -0.0214906, -0.0108997, -0.0070104, 0.0069523
3: -0.0049952, 0.0041871, -0.0049172, 0.0042732, -0.0079321, 0.0080485
4: 0.0145911, 0.0160763, 0.0146146, 0.0160696, -0.0014786, 0.0014617
5: -0.0082837, 0.0046470, -0.0081738, 0.0047683, -0.0116548, 0.0118276
6: 0.9918712, 1.0005836, 0.9919453, 1.0006653, -0.0074400, 0.0075491
7: 0.0130295, 0.0173490, 0.0130721, 0.0173848, -0.0023765, 0.0023500
8: 0.0034481, 0.0073682, 0.0034114, 0.0073349, -0.0038868, 0.0039568
9: -0.0239283, -0.0150390, -0.0240117, -0.0151145, -0.0067555, 0.0066703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037555, upper bound: 0.0037297
time: 0.68 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0037630, upper bound: 0.0037630
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.99 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.99
Output dim: 6, lower bound: -0.0037297, upper bound: 0.0037153
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 6, lower bound: -0.0037629, upper bound: 0.0037219
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 6, lower bound: -0.0037555, upper bound: 0.0037297
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.99
Output dim: 6, lower bound: -0.0037630, upper bound: 0.0037630

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056907, 0.0100078, 0.0056478, 0.0100249, -0.0040473, 0.0040150
1: 0.0015862, 0.0056987, 0.0015453, 0.0057151, -0.0040141, 0.0039863
2: -0.0212504, -0.0109579, -0.0213529, -0.0109170, -0.0068221, 0.0066680
3: -0.0048667, 0.0040647, -0.0049022, 0.0041537, -0.0077393, 0.0078168
4: 0.0146298, 0.0160653, 0.0146191, 0.0160684, -0.0014385, 0.0014462
5: -0.0081026, 0.0044747, -0.0081526, 0.0046000, -0.0113857, 0.0114858
6: 0.9919932, 1.0004675, 0.9919595, 1.0005519, -0.0072566, 0.0073318
7: 0.0130997, 0.0172980, 0.0130803, 0.0173350, -0.0022144, 0.0023072
8: 0.0035004, 0.0073134, 0.0034625, 0.0073285, -0.0038281, 0.0038509
9: -0.0238098, -0.0151634, -0.0238959, -0.0151291, -0.0065659, 0.0064664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 92

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036813, upper bound: 0.0036675
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036878, upper bound: 0.0036564
time: 0.79 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0057785, 0.0100904, 0.0056487, 0.0100221, -0.0040824, 0.0040631
1: 0.0016698, 0.0057775, 0.0015461, 0.0057124, -0.0040425, 0.0040316
2: -0.0210411, -0.0107609, -0.0213506, -0.0109239, -0.0067117, 0.0068404
3: -0.0050377, 0.0038831, -0.0048962, 0.0041517, -0.0078424, 0.0078543
4: 0.0145783, 0.0160799, 0.0146209, 0.0160678, -0.0014896, 0.0014590
5: -0.0083435, 0.0042190, -0.0081443, 0.0045972, -0.0115273, 0.0115685
6: 0.9918309, 1.0002952, 0.9919652, 1.0005502, -0.0073555, 0.0073621
7: 0.0130064, 0.0172224, 0.0130835, 0.0173342, -0.0023402, 0.0021962
8: 0.0035780, 0.0073864, 0.0034633, 0.0073260, -0.0037480, 0.0039231
9: -0.0236340, -0.0149979, -0.0238941, -0.0151348, -0.0065372, 0.0065821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037065, upper bound: 0.0036607
time: 0.70 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036812, upper bound: 0.0036621
time: 0.72 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0056701, 0.0100636, 0.0055976, 0.0100309, -0.0040844, 0.0040882
1: 0.0015665, 0.0057519, 0.0014975, 0.0057208, -0.0040569, 0.0040552
2: -0.0212996, -0.0108248, -0.0214725, -0.0109027, -0.0067599, 0.0069165
3: -0.0049822, 0.0041075, -0.0049146, 0.0042575, -0.0078937, 0.0078664
4: 0.0145950, 0.0160752, 0.0146154, 0.0160694, -0.0014744, 0.0014598
5: -0.0082654, 0.0045349, -0.0081702, 0.0047462, -0.0116002, 0.0115789
6: 0.9918836, 1.0005082, 0.9919478, 1.0006505, -0.0074041, 0.0073747
7: 0.0130366, 0.0173158, 0.0130735, 0.0173783, -0.0023643, 0.0022300
8: 0.0034821, 0.0073627, 0.0034182, 0.0073338, -0.0038517, 0.0039445
9: -0.0238512, -0.0150516, -0.0239964, -0.0151170, -0.0065602, 0.0066363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 92

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0037128, upper bound: 0.0036814
time: 0.69 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0036878, upper bound: 0.0036879
time: 0.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.18 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0036813, upper bound: 0.0036675
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0036878, upper bound: 0.0036564
IS_A2_A1_A1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0037065, upper bound: 0.0036607
IS_A2_A1_A2, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0036812, upper bound: 0.0036621
IS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0037128, upper bound: 0.0036814
IS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 6, lower bound: -0.0036878, upper bound: 0.0036879

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.26 + 16.76 = 20.02 seconds
