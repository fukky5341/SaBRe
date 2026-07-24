## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.46374671450000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7471986, 0.7471986)
1: (-8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9235189, 0.9235187)
2: (-2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7697604, 0.7697604)
3: (-10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8960600, 0.8960595)
4: (-8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8202255, 0.8202257)
5: (-5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6728578, 0.6728578)
6: (-1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8079462, 0.8079464)
7: (-8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.9045877, 0.9045880)
8: (-1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7171779, 0.7171779)
9: (-6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8136141, 0.8136144)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.42 + 33.15 = 57.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4660768, upper bound: 0.4660770

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4653043, upper bound: 0.4594494
time: 3.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660691, upper bound: 0.4660686
time: 3.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.58 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 0, lower bound: -0.4653043, upper bound: 0.4594494
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 0, lower bound: -0.4660691, upper bound: 0.4660686

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 6.1331582, 7.3987856, 6.1249514, 7.4114141, -0.7310331, 0.7258617
1: -8.7575912, -7.1591492, -8.7968254, -7.1384668, -0.8631380, 0.8799698
2: -2.9724138, -1.7035923, -2.9776804, -1.6981771, -0.7617683, 0.7606225
3: -10.3559055, -9.1091175, -10.3793268, -9.0709419, -0.8529415, 0.8373950
4: -8.2852497, -6.9487348, -8.3242874, -6.9299822, -0.7597914, 0.7809100
5: -5.8633595, -4.9384718, -5.8676500, -4.9298859, -0.6593430, 0.6601105
6: -1.5791979, -0.3234253, -1.5966129, -0.3184485, -0.7829056, 0.7916591
7: -8.5059156, -6.7682753, -8.5081940, -6.7652550, -0.9008017, 0.9004929
8: -1.6907382, -0.7279558, -1.6962919, -0.7254047, -0.7093906, 0.7097278
9: -6.3741193, -4.9508686, -6.3968396, -4.9088178, -0.7621522, 0.7498763

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594491
time: 2.91 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594499
time: 2.90 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 6.1240330, 7.4177566, 6.1240315, 7.4177637, -0.7471974, 0.7355554
1: -8.8166838, -7.1378679, -8.8166971, -7.1378689, -0.8755596, 0.9195175
2: -2.9785860, -1.6955370, -2.9785862, -1.6955311, -0.7697594, 0.7687893
3: -10.3806286, -9.0514965, -10.3806295, -9.0514898, -0.8936236, 0.8451240
4: -8.3440285, -6.9297085, -8.3440437, -6.9297085, -0.7713540, 0.8202250
5: -5.8682680, -4.9259892, -5.8682694, -4.9259853, -0.6706889, 0.6735122
6: -1.6049767, -0.3183832, -1.6049852, -0.3183823, -0.7984633, 0.8057237
7: -8.5092392, -6.7643952, -8.5092402, -6.7643933, -0.9045835, 0.9048123
8: -1.6987910, -0.7250733, -1.6987944, -0.7250729, -0.7187605, 0.7157953
9: -6.3969994, -4.8874631, -6.3970003, -4.8874464, -0.8041394, 0.7546283

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4653046
time: 3.50 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4660701
time: 2.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.18 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 29.18
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594491
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 29.18
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594499
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4653046
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.18
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4660701

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 6.1240330, 7.4177566, 6.1331582, 7.3987856, -0.7267725, 0.7378643
1: -8.8166838, -7.1378679, -8.7575912, -7.1591492, -0.8816195, 0.8595216
2: -2.9785860, -1.6955370, -2.9724138, -1.7035923, -0.7612302, 0.7645543
3: -10.3806286, -9.0514965, -10.3559055, -9.1091175, -0.8358026, 0.8533566
4: -8.3440285, -6.9297085, -8.2852497, -6.9487348, -0.7914507, 0.7600677
5: -5.8682680, -4.9259892, -5.8633595, -4.9384718, -0.6588454, 0.6651185
6: -1.6049767, -0.3183832, -1.5791979, -0.3234253, -0.8006549, 0.7807622
7: -8.5092392, -6.7643952, -8.5059156, -6.7682753, -0.9015760, 0.9012170
8: -1.6987910, -0.7250733, -1.6907382, -0.7279558, -0.7128067, 0.7083752
9: -6.3969994, -4.8874631, -6.3741193, -4.9508686, -0.7404075, 0.7630820

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4650425
time: 3.22 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4650426
time: 3.32 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 6.1240330, 7.4177566, 6.1240330, 7.4177566, -0.7355547, 0.7355549
1: -8.8166838, -7.1378679, -8.8166838, -7.1378679, -0.8755593, 0.8755591
2: -2.9785860, -1.6955370, -2.9785860, -1.6955370, -0.7687886, 0.7687883
3: -10.3806286, -9.0514965, -10.3806286, -9.0514965, -0.8451223, 0.8451223
4: -8.3440285, -6.9297085, -8.3440285, -6.9297085, -0.7713537, 0.7713537
5: -5.8682680, -4.9259892, -5.8682680, -4.9259892, -0.6735098, 0.6735101
6: -1.6049767, -0.3183832, -1.6049767, -0.3183832, -0.7984619, 0.7984617
7: -8.5092392, -6.7643952, -8.5092392, -6.7643952, -0.9048114, 0.9048116
8: -1.6987910, -0.7250733, -1.6987910, -0.7250733, -0.7187581, 0.7187583
9: -6.3969994, -4.8874631, -6.3969994, -4.8874631, -0.7546287, 0.7546287

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4658080
time: 3.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4658080
time: 3.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.02 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4650425
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4650426
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4658080
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.02
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4658080

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142933, 6.1353321, 7.3987346, -0.7205870, 0.7321799
1: -8.8136292, -7.1427507, -8.7575207, -7.1608477, -0.8756649, 0.8544785
2: -2.9745994, -1.7009498, -2.9722590, -1.7054963, -0.7554443, 0.7589755
3: -10.3794327, -9.0533953, -10.3558884, -9.1097651, -0.8333814, 0.8513682
4: -8.3419619, -6.9315615, -8.2851419, -6.9493785, -0.7882459, 0.7580717
5: -5.8642211, -4.9291921, -5.8618908, -4.9386153, -0.6546795, 0.6604774
6: -1.6036339, -0.3190961, -1.5788980, -0.3234959, -0.7990551, 0.7801313
7: -8.5061531, -6.7696247, -8.5058384, -6.7700887, -0.8966289, 0.8959074
8: -1.6978998, -0.7261963, -1.6906557, -0.7283354, -0.7115879, 0.7071548
9: -6.3964672, -4.8880863, -6.3740702, -4.9510679, -0.7395860, 0.7624085

Time for backsubstitution: 22.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4582975
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4650425
time: 3.29 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 6.1240406, 7.4177561, 6.1331606, 7.3987856, -0.7202067, 0.7378623
1: -8.8166847, -7.1378784, -8.7575922, -7.1591511, -0.8789582, 0.8543243
2: -2.9785833, -1.6955433, -2.9724140, -1.7035955, -0.7612290, 0.7588680
3: -10.3806286, -9.0515003, -10.3559055, -9.1091166, -0.8345714, 0.8515649
4: -8.3440266, -6.9297123, -8.2852488, -6.9487348, -0.7902944, 0.7584047
5: -5.8682647, -4.9259901, -5.8633585, -4.9384718, -0.6542273, 0.6651175
6: -1.6049752, -0.3183823, -1.5791969, -0.3234253, -0.8008366, 0.7806292
7: -8.5092411, -6.7643991, -8.5059147, -6.7682772, -0.9015744, 0.8956015
8: -1.6987901, -0.7250752, -1.6907377, -0.7279563, -0.7128062, 0.7078018
9: -6.3969998, -4.8874621, -6.3741169, -4.9508686, -0.7400894, 0.7627331

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4582975
time: 3.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4650425
time: 3.23 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142933, 6.1262040, 7.4177022, -0.7293692, 0.7298698
1: -8.8136292, -7.1427507, -8.8166075, -7.1395674, -0.8707385, 0.8705263
2: -2.9745994, -1.7009498, -2.9784284, -1.6974344, -0.7630057, 0.7632055
3: -10.3794327, -9.0533953, -10.3806105, -9.0521479, -0.8453341, 0.8445928
4: -8.3419619, -6.9315615, -8.3439150, -6.9303513, -0.7685440, 0.7693615
5: -5.8642211, -4.9291921, -5.8668003, -4.9261341, -0.6693411, 0.6688721
6: -1.6036339, -0.3190961, -1.6046915, -0.3184533, -0.7968788, 0.7978163
7: -8.5061531, -6.7696247, -8.5091648, -6.7662053, -0.8998661, 0.8995025
8: -1.6978998, -0.7261963, -1.6987062, -0.7254524, -0.7175474, 0.7175410
9: -6.3964672, -4.8880863, -6.3969517, -4.8876557, -0.7543693, 0.7541478

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4590629
time: 3.36 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658082
time: 3.13 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 6.1240406, 7.4177561, 6.1240358, 7.4177566, -0.7289894, 0.7355535
1: -8.8166847, -7.1378784, -8.8166838, -7.1378736, -0.8755574, 0.8704367
2: -2.9785833, -1.6955433, -2.9785857, -1.6955366, -0.7687871, 0.7631021
3: -10.3806286, -9.0515003, -10.3806295, -9.0514984, -0.8446865, 0.8466191
4: -8.3440266, -6.9297123, -8.3440275, -6.9297090, -0.7713532, 0.7696896
5: -5.8682647, -4.9259901, -5.8682671, -4.9259892, -0.6688907, 0.6735089
6: -1.6049752, -0.3183823, -1.6049767, -0.3183823, -0.7986422, 0.7983291
7: -8.5092411, -6.7643991, -8.5092430, -6.7643976, -0.9048104, 0.8991966
8: -1.6987901, -0.7250752, -1.6987901, -0.7250738, -0.7187581, 0.7181854
9: -6.3969998, -4.8874621, -6.3969998, -4.8874607, -0.7545187, 0.7549720

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631
time: 3.12 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4658081
time: 3.20 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.84 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4582975
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4650425
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4582975
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4650425
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4590629
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658082
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.84
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4658081

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142933, 6.1331668, 7.3987856, -0.7206345, 0.7343490
1: -8.8136292, -7.1427507, -8.7575932, -7.1591563, -0.8758035, 0.8518863
2: -2.9745994, -1.7009498, -2.9724133, -1.7035999, -0.7573736, 0.7590940
3: -10.3794327, -9.0533953, -10.3559055, -9.1091185, -0.8334341, 0.8501680
4: -8.3419619, -6.9315615, -8.2852478, -6.9487371, -0.7883749, 0.7581501
5: -5.8642211, -4.9291921, -5.8633556, -4.9384723, -0.6547976, 0.6619458
6: -1.6036339, -0.3190961, -1.5791955, -0.3234253, -0.7991095, 0.7798004
7: -8.5061531, -6.7696247, -8.5059137, -6.7682810, -0.8984506, 0.8959880
8: -1.6978998, -0.7261963, -1.6907372, -0.7279558, -0.7119870, 0.7072010
9: -6.3964672, -4.8880863, -6.3741179, -4.9508715, -0.7396306, 0.7621385

Time for backsubstitution: 22.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513975, upper bound: 0.4650253
time: 3.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524252, upper bound: 0.4650254
time: 3.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 6.1240406, 7.4177561, 6.1331668, 7.3987856, -0.7202065, 0.7312748
1: -8.8166847, -7.1378784, -8.7575932, -7.1591563, -0.8789575, 0.8541954
2: -2.9785833, -1.6955433, -2.9724133, -1.7035999, -0.7555439, 0.7588680
3: -10.3806286, -9.0515003, -10.3559055, -9.1091185, -0.8353424, 0.8515649
4: -8.3440266, -6.9297123, -8.2852478, -6.9487371, -0.7902937, 0.7584040
5: -5.8682647, -4.9259901, -5.8633556, -4.9384723, -0.6542270, 0.6604991
6: -1.6049752, -0.3183823, -1.5791955, -0.3234253, -0.8008366, 0.7809422
7: -8.5092411, -6.7643991, -8.5059137, -6.7682810, -0.8959603, 0.8956010
8: -1.6987901, -0.7250752, -1.6907372, -0.7279558, -0.7122326, 0.7078013
9: -6.3969998, -4.8874621, -6.3741179, -4.9508715, -0.7403923, 0.7627329

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4581425, upper bound: 0.4582803
time: 3.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4591702, upper bound: 0.4582804
time: 3.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142933, 6.1240406, 7.4177561, -0.7294176, 0.7320397
1: -8.8136292, -7.1427507, -8.8166847, -7.1378784, -0.8724642, 0.8705826
2: -2.9745994, -1.7009498, -2.9785833, -1.6955433, -0.7649319, 0.7633288
3: -10.3794327, -9.0533953, -10.3806286, -9.0515003, -0.8441749, 0.8444567
4: -8.3419619, -6.9315615, -8.3440266, -6.9297123, -0.7692192, 0.7694361
5: -5.8642211, -4.9291921, -5.8682647, -4.9259901, -0.6694622, 0.6703405
6: -1.6036339, -0.3190961, -1.6049752, -0.3183823, -0.7969308, 0.7975008
7: -8.5061531, -6.7696247, -8.5092411, -6.7643991, -0.9016867, 0.8995833
8: -1.6978998, -0.7261963, -1.6987901, -0.7250752, -0.7179456, 0.7175882
9: -6.3964672, -4.8880863, -6.3969998, -4.8874621, -0.7541029, 0.7541565

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4525622, upper bound: 0.4657908
time: 3.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4535897, upper bound: 0.4657901
time: 3.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 6.1240406, 7.4177561, 6.1240406, 7.4177561, -0.7289891, 0.7289891
1: -8.8166847, -7.1378784, -8.8166847, -7.1378784, -0.8704360, 0.8704360
2: -2.9785833, -1.6955433, -2.9785833, -1.6955433, -0.7631023, 0.7631021
3: -10.3806286, -9.0515003, -10.3806286, -9.0515003, -0.8466187, 0.8466184
4: -8.3440266, -6.9297123, -8.3440266, -6.9297123, -0.7696891, 0.7696891
5: -5.8682647, -4.9259901, -5.8682647, -4.9259901, -0.6688905, 0.6688907
6: -1.6049752, -0.3183823, -1.6049752, -0.3183823, -0.7986417, 0.7986419
7: -8.5092411, -6.7643991, -8.5092411, -6.7643991, -0.8991959, 0.8991959
8: -1.6987901, -0.7250752, -1.6987901, -0.7250752, -0.7181849, 0.7181852
9: -6.3969998, -4.8874621, -6.3969998, -4.8874621, -0.7549722, 0.7549721

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4593072, upper bound: 0.4590450
time: 3.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4603347, upper bound: 0.4590452
time: 3.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.67 seconds
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4513975, upper bound: 0.4650253
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4524252, upper bound: 0.4650254
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4581425, upper bound: 0.4582803
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4591702, upper bound: 0.4582804
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4525622, upper bound: 0.4657908
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4535897, upper bound: 0.4657901
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4593072, upper bound: 0.4590450
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 29.67
Output dim: 0, lower bound: -0.4603347, upper bound: 0.4590452

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 6.1312599, 7.4142342, 6.1331668, 7.3987856, -0.7195539, 0.7342684
1: -8.8135529, -7.1436462, -8.7575932, -7.1591563, -0.8757062, 0.8510026
2: -2.9743452, -1.7010705, -2.9724133, -1.7035999, -0.7569652, 0.7586017
3: -10.3785572, -9.0534172, -10.3559055, -9.1091185, -0.8324914, 0.8501356
4: -8.3418064, -6.9317846, -8.2852478, -6.9487371, -0.7880869, 0.7579052
5: -5.8642044, -4.9298997, -5.8633556, -4.9384723, -0.6547852, 0.6612356
6: -1.6034226, -0.3197289, -1.5791955, -0.3234253, -0.7986155, 0.7789078
7: -8.5052242, -6.7696819, -8.5059137, -6.7682810, -0.8975618, 0.8959417
8: -1.6976781, -0.7262716, -1.6907372, -0.7279558, -0.7116032, 0.7069740
9: -6.3963490, -4.8884711, -6.3741179, -4.9508715, -0.7395183, 0.7617283

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513948, upper bound: 0.4639934
time: 3.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513948, upper bound: 0.4650245
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 6.1280203, 7.4351730, 6.1331692, 7.3987856, -0.7262378, 0.7466843
1: -8.8325806, -7.1368275, -8.7575932, -7.1591635, -0.8772082, 0.8618655
2: -2.9896121, -1.7000388, -2.9724145, -1.7036015, -0.7717884, 0.7682736
3: -10.3802013, -9.0338612, -10.3559027, -9.1091194, -0.8385072, 0.8516622
4: -8.3464203, -6.9311352, -8.2852497, -6.9487381, -0.7893345, 0.7600861
5: -5.8769798, -4.9258566, -5.8633566, -4.9384747, -0.6609473, 0.6657681
6: -1.6201305, -0.3182101, -1.5791960, -0.3234282, -0.8072052, 0.7886686
7: -8.5071831, -6.7489767, -8.5059118, -6.7682800, -0.9040780, 0.9146193
8: -1.7034917, -0.7192974, -1.6907368, -0.7279572, -0.7164869, 0.7176564
9: -6.4075823, -4.8855085, -6.3741183, -4.9508719, -0.7417508, 0.7652228

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524259, upper bound: 0.4639934
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524259, upper bound: 0.4650238
time: 3.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 6.1312599, 7.4142342, 6.1240406, 7.4177561, -0.7283368, 0.7319589
1: -8.8135529, -7.1436462, -8.8166847, -7.1378784, -0.8723657, 0.8697004
2: -2.9743452, -1.7010705, -2.9785833, -1.6955433, -0.7645228, 0.7628362
3: -10.3785572, -9.0534172, -10.3806286, -9.0515003, -0.8432317, 0.8444250
4: -8.3418064, -6.9317846, -8.3440266, -6.9297123, -0.7689307, 0.7691908
5: -5.8642044, -4.9298997, -5.8682647, -4.9259901, -0.6694503, 0.6696298
6: -1.6034226, -0.3197289, -1.6049752, -0.3183823, -0.7964435, 0.7966075
7: -8.5052242, -6.7696819, -8.5092411, -6.7643991, -0.9007978, 0.8995366
8: -1.6976781, -0.7262716, -1.6987901, -0.7250752, -0.7175612, 0.7173638
9: -6.3963490, -4.8884711, -6.3969998, -4.8874621, -0.7539895, 0.7537524

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4525595, upper bound: 0.4647593
time: 3.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4525595, upper bound: 0.4657904
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 6.1280203, 7.4351730, 6.1240416, 7.4177561, -0.7350204, 0.7528119
1: -8.8325806, -7.1368275, -8.8166866, -7.1378798, -0.8840411, 0.8805528
2: -2.9896121, -1.7000388, -2.9785843, -1.6955432, -0.7793460, 0.7738631
3: -10.3802013, -9.0338612, -10.3806248, -9.0515003, -0.8509011, 0.8589253
4: -8.3464203, -6.9311352, -8.3440275, -6.9297123, -0.7730501, 0.7713723
5: -5.8769798, -4.9258566, -5.8682642, -4.9259915, -0.6786950, 0.6741250
6: -1.6201305, -0.3182101, -1.6049743, -0.3183842, -0.8124914, 0.8063676
7: -8.5071831, -6.7489767, -8.5092354, -6.7643995, -0.9073136, 0.9181651
8: -1.7034917, -0.7192974, -1.6987906, -0.7250752, -0.7223601, 0.7280548
9: -6.4075823, -4.8855085, -6.3969979, -4.8874631, -0.7653182, 0.7573007

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4535905, upper bound: 0.4647585
time: 3.14 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4535905, upper bound: 0.4657903
time: 3.27 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 28.04 seconds
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4513948, upper bound: 0.4639934
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4513948, upper bound: 0.4650245
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4524259, upper bound: 0.4639934
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4524259, upper bound: 0.4650238
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4525595, upper bound: 0.4647593
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4525595, upper bound: 0.4657904
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4535905, upper bound: 0.4647585
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 28.04
Output dim: 0, lower bound: -0.4535905, upper bound: 0.4657903

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 6.1312599, 7.4142342, 6.1342211, 7.3987284, -0.7194734, 0.7331901
1: -8.8135529, -7.1436462, -8.7575169, -7.1600533, -0.8748229, 0.8509026
2: -2.9743452, -1.7010705, -2.9721637, -1.7037189, -0.7564740, 0.7581925
3: -10.3785572, -9.0534172, -10.3550262, -9.1091413, -0.8324592, 0.8491926
4: -8.3418064, -6.9317846, -8.2850924, -6.9489598, -0.7878418, 0.7576272
5: -5.8642044, -4.9298997, -5.8633409, -4.9391813, -0.6540747, 0.6612234
6: -1.6034226, -0.3197289, -1.5789905, -0.3240576, -0.7977228, 0.7784200
7: -8.5052242, -6.7696819, -8.5049877, -6.7683377, -0.8975151, 0.8950531
8: -1.6976781, -0.7262716, -1.6905150, -0.7280326, -0.7113771, 0.7065897
9: -6.3963490, -4.8884711, -6.3739996, -4.9512529, -0.7391090, 0.7616166

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 541

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4505900, upper bound: 0.4639926
time: 3.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513966, upper bound: 0.4639926
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 6.1312599, 7.4142342, 6.1309614, 7.4196882, -0.7375927, 0.7360868
1: -8.8135529, -7.1436462, -8.7765751, -7.1532393, -0.8824122, 0.8522422
2: -2.9743452, -1.7010705, -2.9874182, -1.7026954, -0.7578020, 0.7729337
3: -10.3785572, -9.0534172, -10.3566723, -9.0895786, -0.8339849, 0.8513119
4: -8.3418064, -6.9317846, -8.2898054, -6.9483056, -0.7886212, 0.7619016
5: -5.8642044, -4.9298997, -5.8761187, -4.9351797, -0.6571457, 0.6652510
6: -1.6034226, -0.3197289, -1.5957518, -0.3225369, -0.7988458, 0.7933617
7: -8.5052242, -6.7696819, -8.5069504, -6.7476215, -0.9150150, 0.8980803
8: -1.6976781, -0.7262716, -1.6962838, -0.7210488, -0.7183805, 0.7113936
9: -6.3963490, -4.8884711, -6.3852434, -4.9482956, -0.7416161, 0.7638448

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 541

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4505900, upper bound: 0.4650237
time: 3.10 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513966, upper bound: 0.4650238
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 6.1280203, 7.4351730, 6.1342211, 7.3987284, -0.7223549, 0.7456048
1: -8.8325806, -7.1368275, -8.7575169, -7.1600533, -0.8763244, 0.8584944
2: -2.9896121, -1.7000388, -2.9721637, -1.7037189, -0.7712984, 0.7595260
3: -10.3802013, -9.0338612, -10.3550262, -9.1091413, -0.8345993, 0.8507195
4: -8.3464203, -6.9311352, -8.2850924, -6.9489598, -0.7890897, 0.7583978
5: -5.8769798, -4.9258566, -5.8633409, -4.9391813, -0.6602323, 0.6643524
6: -1.6201305, -0.3182101, -1.5789905, -0.3240576, -0.8063107, 0.7795432
7: -8.5071831, -6.7489767, -8.5049877, -6.7683377, -0.9005353, 0.9137261
8: -1.7034917, -0.7192974, -1.6905150, -0.7280326, -0.7162619, 0.7136004
9: -6.4075823, -4.8855085, -6.3739996, -4.9512529, -0.7413416, 0.7641374

Time for backsubstitution: 22.33 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.57 + 549.50 = 607.07 seconds
