## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00071487


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137279, -0.0096066, -0.0137279, -0.0096066, -0.0028545, 0.0028545)
1: (-0.0068091, -0.0056471, -0.0068091, -0.0056471, -0.0008048, 0.0008048)
2: (-0.0116790, -0.0031057, -0.0116790, -0.0031057, -0.0059379, 0.0059379)
3: (0.0000818, 0.0012163, 0.0000818, 0.0012163, -0.0007858, 0.0007858)
4: (0.0084129, 0.0148200, 0.0084129, 0.0148200, -0.0044376, 0.0044376)
5: (0.9978436, 0.9996237, 0.9978436, 0.9996237, -0.0012329, 0.0012329)
6: (0.0059263, 0.0075421, 0.0059263, 0.0075421, -0.0011191, 0.0011191)
7: (-0.0012657, 0.0047642, -0.0012657, 0.0047642, -0.0041763, 0.0041763)
8: (-0.0129008, -0.0082078, -0.0129008, -0.0082078, -0.0032504, 0.0032504)
9: (-0.0033016, -0.0028967, -0.0033016, -0.0028967, -0.0002804, 0.0002804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.68 = 3.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0007731, upper bound: 0.0007733

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007310, upper bound: 0.0007117
time: 0.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007310, upper bound: 0.0007310
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 5, lower bound: -0.0007310, upper bound: 0.0007117
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.84
Output dim: 5, lower bound: -0.0007310, upper bound: 0.0007310

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0134696, -0.0096818, -0.0136626, -0.0096087, -0.0024992, 0.0024861
1: -0.0067362, -0.0056683, -0.0067906, -0.0056477, -0.0007046, 0.0007009
2: -0.0111416, -0.0032624, -0.0115431, -0.0031103, -0.0051988, 0.0051716
3: 0.0001529, 0.0011956, 0.0000998, 0.0012157, -0.0006880, 0.0006844
4: 0.0085299, 0.0144184, 0.0084163, 0.0147184, -0.0038649, 0.0038853
5: 0.9978761, 0.9995121, 0.9978446, 0.9995955, -0.0010738, 0.0010794
6: 0.0059558, 0.0074408, 0.0059271, 0.0075164, -0.0009747, 0.0009798
7: -0.0011555, 0.0043862, -0.0012625, 0.0046686, -0.0036373, 0.0036565
8: -0.0126066, -0.0082935, -0.0128264, -0.0082103, -0.0028458, 0.0028309
9: -0.0032942, -0.0029221, -0.0033014, -0.0029031, -0.0002442, 0.0002455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007118
time: 0.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007118
time: 1.14 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0135499, -0.0096125, -0.0136831, -0.0096080, -0.0023045, 0.0028372
1: -0.0067589, -0.0056488, -0.0067964, -0.0056475, -0.0006497, 0.0007999
2: -0.0113086, -0.0031180, -0.0115858, -0.0031088, -0.0047939, 0.0059020
3: 0.0001308, 0.0012147, 0.0000941, 0.0012159, -0.0006344, 0.0007810
4: 0.0084221, 0.0145432, 0.0084152, 0.0147504, -0.0044108, 0.0035826
5: 0.9978461, 0.9995468, 0.9978443, 0.9996043, -0.0012254, 0.0009954
6: 0.0059286, 0.0074723, 0.0059269, 0.0075245, -0.0011123, 0.0009035
7: -0.0012570, 0.0045036, -0.0012635, 0.0046986, -0.0041510, 0.0033717
8: -0.0126981, -0.0082145, -0.0128498, -0.0082095, -0.0026242, 0.0032308
9: -0.0033010, -0.0029142, -0.0033015, -0.0029011, -0.0002787, 0.0002264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007310
time: 0.84 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007310
time: 1.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.44 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.44
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007118
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.44
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007118
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007310
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 5, lower bound: -0.0007118, upper bound: 0.0007310

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0135499, -0.0096125, -0.0134696, -0.0096818, -0.0025374, 0.0024967
1: -0.0067589, -0.0056488, -0.0067362, -0.0056683, -0.0007154, 0.0007039
2: -0.0113086, -0.0031180, -0.0111416, -0.0032624, -0.0052783, 0.0051936
3: 0.0001308, 0.0012147, 0.0001529, 0.0011956, -0.0006985, 0.0006873
4: 0.0084221, 0.0145432, 0.0085299, 0.0144184, -0.0038813, 0.0039446
5: 0.9978461, 0.9995468, 0.9978761, 0.9995121, -0.0010784, 0.0010959
6: 0.0059286, 0.0074723, 0.0059558, 0.0074408, -0.0009788, 0.0009948
7: -0.0012570, 0.0045036, -0.0011555, 0.0043862, -0.0036528, 0.0037124
8: -0.0126981, -0.0082145, -0.0126066, -0.0082935, -0.0028893, 0.0028430
9: -0.0033010, -0.0029142, -0.0032942, -0.0029221, -0.0002453, 0.0002493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006994, upper bound: 0.0007042
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007175
time: 0.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0135499, -0.0096125, -0.0135499, -0.0096125, -0.0023012, 0.0023012
1: -0.0067589, -0.0056488, -0.0067589, -0.0056488, -0.0006488, 0.0006488
2: -0.0113086, -0.0031180, -0.0113086, -0.0031180, -0.0047871, 0.0047871
3: 0.0001308, 0.0012147, 0.0001308, 0.0012147, -0.0006335, 0.0006335
4: 0.0084221, 0.0145432, 0.0084221, 0.0145432, -0.0035775, 0.0035775
5: 0.9978461, 0.9995468, 0.9978461, 0.9995468, -0.0009939, 0.0009939
6: 0.0059286, 0.0074723, 0.0059286, 0.0074723, -0.0009022, 0.0009022
7: -0.0012570, 0.0045036, -0.0012570, 0.0045036, -0.0033669, 0.0033669
8: -0.0126981, -0.0082145, -0.0126981, -0.0082145, -0.0026204, 0.0026204
9: -0.0033010, -0.0029142, -0.0033010, -0.0029142, -0.0002261, 0.0002261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006994, upper bound: 0.0007041
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007175
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.20 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 5, lower bound: -0.0006994, upper bound: 0.0007042
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007175
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.20
Output dim: 5, lower bound: -0.0006994, upper bound: 0.0007041
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 5, lower bound: -0.0006995, upper bound: 0.0007175

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0134611, -0.0096829, -0.0023897, 0.0024824
1: -0.0067453, -0.0056504, -0.0067338, -0.0056686, -0.0006738, 0.0006999
2: -0.0112086, -0.0031298, -0.0111239, -0.0032646, -0.0049712, 0.0051638
3: 0.0001440, 0.0012131, 0.0001552, 0.0011953, -0.0006579, 0.0006833
4: 0.0084308, 0.0144685, 0.0085316, 0.0144052, -0.0038591, 0.0037151
5: 0.9978486, 0.9995260, 0.9978766, 0.9995084, -0.0010722, 0.0010322
6: 0.0059308, 0.0074534, 0.0059562, 0.0074374, -0.0009732, 0.0009369
7: -0.0012488, 0.0044333, -0.0011539, 0.0043737, -0.0036319, 0.0034964
8: -0.0126433, -0.0082209, -0.0125970, -0.0082948, -0.0027212, 0.0028267
9: -0.0033005, -0.0029189, -0.0032941, -0.0029229, -0.0002439, 0.0002348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0007174
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0007175
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0135401, -0.0096136, -0.0021516, 0.0022868
1: -0.0067453, -0.0056504, -0.0067561, -0.0056491, -0.0006066, 0.0006447
2: -0.0112086, -0.0031298, -0.0112882, -0.0031204, -0.0044758, 0.0047570
3: 0.0001440, 0.0012131, 0.0001335, 0.0012144, -0.0005923, 0.0006295
4: 0.0084308, 0.0144685, 0.0084239, 0.0145280, -0.0035551, 0.0033450
5: 0.9978486, 0.9995260, 0.9978467, 0.9995425, -0.0009877, 0.0009293
6: 0.0059308, 0.0074534, 0.0059291, 0.0074684, -0.0008965, 0.0008435
7: -0.0012488, 0.0044333, -0.0012553, 0.0044893, -0.0033457, 0.0031480
8: -0.0126433, -0.0082209, -0.0126869, -0.0082158, -0.0024501, 0.0026040
9: -0.0033005, -0.0029189, -0.0033009, -0.0029152, -0.0002247, 0.0002114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 62

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006816, upper bound: 0.0007175
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006816, upper bound: 0.0007175
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.33 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0007174
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 5, lower bound: -0.0006815, upper bound: 0.0007175
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 5, lower bound: -0.0006816, upper bound: 0.0007175
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 5, lower bound: -0.0006816, upper bound: 0.0007175

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0133825, -0.0097345, -0.0023850, 0.0023888
1: -0.0067453, -0.0056504, -0.0067117, -0.0056832, -0.0006724, 0.0006735
2: -0.0112086, -0.0031298, -0.0109604, -0.0033719, -0.0049612, 0.0049691
3: 0.0001440, 0.0012131, 0.0001769, 0.0011811, -0.0006565, 0.0006576
4: 0.0084308, 0.0144685, 0.0086118, 0.0142830, -0.0037136, 0.0037077
5: 0.9978486, 0.9995260, 0.9978989, 0.9994744, -0.0010318, 0.0010301
6: 0.0059308, 0.0074534, 0.0059764, 0.0074066, -0.0009365, 0.0009350
7: -0.0012488, 0.0044333, -0.0010784, 0.0042587, -0.0034949, 0.0034893
8: -0.0126433, -0.0082209, -0.0125075, -0.0083535, -0.0027158, 0.0027201
9: -0.0033005, -0.0029189, -0.0032890, -0.0029307, -0.0002347, 0.0002343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006709, upper bound: 0.0006976
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0134278, -0.0096871, -0.0023863, 0.0023853
1: -0.0067453, -0.0056504, -0.0067245, -0.0056698, -0.0006728, 0.0006725
2: -0.0112086, -0.0031298, -0.0110548, -0.0032732, -0.0049640, 0.0049620
3: 0.0001440, 0.0012131, 0.0001644, 0.0011941, -0.0006569, 0.0006566
4: 0.0084308, 0.0144685, 0.0085381, 0.0143535, -0.0037083, 0.0037098
5: 0.9978486, 0.9995260, 0.9978784, 0.9994941, -0.0010303, 0.0010307
6: 0.0059308, 0.0074534, 0.0059578, 0.0074244, -0.0009352, 0.0009356
7: -0.0012488, 0.0044333, -0.0011479, 0.0043251, -0.0034899, 0.0034913
8: -0.0126433, -0.0082209, -0.0125591, -0.0082995, -0.0027173, 0.0027162
9: -0.0033005, -0.0029189, -0.0032937, -0.0029262, -0.0002343, 0.0002344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006709, upper bound: 0.0006976
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0134585, -0.0096534, -0.0021652, 0.0021887
1: -0.0067453, -0.0056504, -0.0067331, -0.0056603, -0.0006105, 0.0006171
2: -0.0112086, -0.0031298, -0.0111187, -0.0032033, -0.0045041, 0.0045529
3: 0.0001440, 0.0012131, 0.0001559, 0.0012034, -0.0005960, 0.0006025
4: 0.0084308, 0.0144685, 0.0084858, 0.0144013, -0.0034026, 0.0033661
5: 0.9978486, 0.9995260, 0.9978639, 0.9995074, -0.0009453, 0.0009352
6: 0.0059308, 0.0074534, 0.0059447, 0.0074365, -0.0008581, 0.0008489
7: -0.0012488, 0.0044333, -0.0011971, 0.0043701, -0.0032022, 0.0031679
8: -0.0126433, -0.0082209, -0.0125941, -0.0082612, -0.0024656, 0.0024923
9: -0.0033005, -0.0029189, -0.0032970, -0.0029232, -0.0002150, 0.0002127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0006976
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0135018, -0.0096181, -0.0135018, -0.0096181, -0.0021477, 0.0021477
1: -0.0067453, -0.0056504, -0.0067453, -0.0056504, -0.0006055, 0.0006055
2: -0.0112086, -0.0031298, -0.0112086, -0.0031298, -0.0044677, 0.0044677
3: 0.0001440, 0.0012131, 0.0001440, 0.0012131, -0.0005912, 0.0005912
4: 0.0084308, 0.0144685, 0.0084308, 0.0144685, -0.0033389, 0.0033389
5: 0.9978486, 0.9995260, 0.9978486, 0.9995260, -0.0009276, 0.0009276
6: 0.0059308, 0.0074534, 0.0059308, 0.0074534, -0.0008420, 0.0008420
7: -0.0012488, 0.0044333, -0.0012488, 0.0044333, -0.0031423, 0.0031423
8: -0.0126433, -0.0082209, -0.0126433, -0.0082209, -0.0024456, 0.0024456
9: -0.0033005, -0.0029189, -0.0033005, -0.0029189, -0.0002110, 0.0002110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0006976
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007075
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.58 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006709, upper bound: 0.0006976
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006709, upper bound: 0.0006976
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0006976
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007076
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0006976
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.58
Output dim: 5, lower bound: -0.0006708, upper bound: 0.0007075

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.34 + 36.21 = 39.55 seconds
