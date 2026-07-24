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
execution time: IAR + RelationalAnalysis = 23.70 + 33.26 = 56.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4660768, upper bound: 0.4660770

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4653043, upper bound: 0.4594494
time: 3.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660691, upper bound: 0.4660686
time: 3.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 0, lower bound: -0.4653043, upper bound: 0.4594494
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.40
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

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594491
time: 3.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594499
time: 2.95 seconds

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

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4653046
time: 3.17 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4660701
time: 2.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.53 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 28.53
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594491
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 28.53
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4594499
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.53
Output dim: 0, lower bound: -0.4594490, upper bound: 0.4653046
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.53
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

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4650425
time: 3.13 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4650426
time: 3.03 seconds

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

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4658080
time: 3.52 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4658080
time: 3.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.49 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.49
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4650425
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.49
Output dim: 0, lower bound: -0.4591876, upper bound: 0.4650426
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.49
Output dim: 0, lower bound: -0.4524428, upper bound: 0.4658080
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.49
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

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4582975
time: 3.06 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4650425
time: 3.17 seconds

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

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4582975
time: 3.18 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4650425
time: 3.17 seconds

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

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4590629
time: 3.11 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658082
time: 3.05 seconds

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

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 891

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4658081
time: 3.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 27.89 seconds
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4582975
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4524431, upper bound: 0.4650425
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4582975
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4591881, upper bound: 0.4650425
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4590629
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4536078, upper bound: 0.4658082
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 27.89
Output dim: 0, lower bound: -0.4603528, upper bound: 0.4590631
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.89
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

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 541

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524422, upper bound: 0.4642349
time: 3.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524422, upper bound: 0.4650416
time: 3.12 seconds

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

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 541

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4591872, upper bound: 0.4574900
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4591872, upper bound: 0.4582966
time: 3.07 seconds

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

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 541

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4528002, upper bound: 0.4658073
time: 3.18 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4536068, upper bound: 0.4658064
time: 3.15 seconds

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

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 541

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4595452, upper bound: 0.4590616
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4603518, upper bound: 0.4590622
time: 3.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 27.91 seconds
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4524422, upper bound: 0.4642349
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4524422, upper bound: 0.4650416
NS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4591872, upper bound: 0.4574900
NS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4591872, upper bound: 0.4582966
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4528002, upper bound: 0.4658073
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4536068, upper bound: 0.4658064
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4595452, upper bound: 0.4590616
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 27.91
Output dim: 0, lower bound: -0.4603518, upper bound: 0.4590622

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 6.1321702, 7.4142590, 6.1395206, 7.3973980, -0.7162313, 0.7280056
1: -8.8133478, -7.1447582, -8.7554379, -7.1656179, -0.8681335, 0.8463277
2: -2.9732685, -1.7012784, -2.9662521, -1.7048497, -0.7539690, 0.7523854
3: -10.3790588, -9.0536118, -10.3533592, -9.1104288, -0.8306940, 0.8474333
4: -8.3417349, -6.9327030, -8.2833614, -6.9536076, -0.7830534, 0.7549939
5: -5.8640523, -4.9309840, -5.8610897, -4.9446907, -0.6486731, 0.6576927
6: -1.6029220, -0.3195429, -1.5759120, -0.3248720, -0.7971830, 0.7754974
7: -8.5032444, -6.7702594, -8.4967051, -6.7738037, -0.8897870, 0.8863759
8: -1.6972275, -0.7285485, -1.6862235, -0.7350950, -0.7036858, 0.6998897
9: -6.3918958, -4.8884435, -6.3603306, -4.9571905, -0.7246091, 0.7480942

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513967, upper bound: 0.4642171
time: 3.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524243, upper bound: 0.4642172
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142923, 6.1331687, 7.3987861, -0.7198954, 0.7348382
1: -8.8136282, -7.1427517, -8.7575941, -7.1591606, -0.8728921, 0.8507619
2: -2.9745991, -1.7009505, -2.9724119, -1.7036011, -0.7568605, 0.7602844
3: -10.3794327, -9.0533943, -10.3559055, -9.1091185, -0.8328428, 0.8490870
4: -8.3419590, -6.9315634, -8.2852478, -6.9487371, -0.7877343, 0.7581499
5: -5.8642197, -4.9291911, -5.8633547, -4.9384742, -0.6517491, 0.6619456
6: -1.6036334, -0.3190975, -1.5791950, -0.3234262, -0.7990546, 0.7824621
7: -8.5061531, -6.7696271, -8.5059109, -6.7682800, -0.8984499, 0.8906138
8: -1.6979003, -0.7261958, -1.6907358, -0.7279601, -0.7084451, 0.7072010
9: -6.3964682, -4.8880858, -6.3741112, -4.9508715, -0.7345138, 0.7493937

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4639953
time: 3.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4650229
time: 3.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 6.1365643, 7.4129024, 6.1260066, 7.4177227, -0.7230878, 0.7276199
1: -8.8114777, -7.1492219, -8.8164043, -7.1398869, -0.8677382, 0.8628922
2: -2.9684234, -1.7021984, -2.9772458, -1.6958692, -0.7581975, 0.7599189
3: -10.3769226, -9.0547066, -10.3802538, -9.0517197, -0.8414440, 0.8417103
4: -8.3400745, -6.9364271, -8.3437986, -6.9308534, -0.7660511, 0.7640910
5: -5.8619642, -4.9354200, -5.8680983, -4.9277816, -0.6652138, 0.6642070
6: -1.6003404, -0.3205442, -1.6042647, -0.3188276, -0.7926693, 0.7955897
7: -8.4969406, -6.7751379, -8.5063334, -6.7650318, -0.8920760, 0.8909228
8: -1.6933889, -0.7333341, -1.6981192, -0.7274265, -0.7106385, 0.7092888
9: -6.3826828, -4.8944044, -6.3924274, -4.8878188, -0.7400608, 0.7431873

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4517546, upper bound: 0.4657894
time: 3.17 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4527822, upper bound: 0.4657894
time: 3.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 6.1302066, 7.4142928, 6.1240392, 7.4177561, -0.7298927, 0.7312989
1: -8.8136282, -7.1427555, -8.8166847, -7.1378770, -0.8724310, 0.8676772
2: -2.9745982, -1.7009510, -2.9785850, -1.6955438, -0.7661231, 0.7628107
3: -10.3794327, -9.0533943, -10.3806295, -9.0515003, -0.8439193, 0.8448462
4: -8.3419609, -6.9315634, -8.3440266, -6.9297113, -0.7692189, 0.7693369
5: -5.8642197, -4.9291935, -5.8682642, -4.9259892, -0.6694615, 0.6672785
6: -1.6036334, -0.3190980, -1.6049752, -0.3183827, -0.7995911, 0.7974455
7: -8.5061493, -6.7696285, -8.5092392, -6.7644019, -0.8963125, 0.8995824
8: -1.6978993, -0.7261992, -1.6987910, -0.7250757, -0.7179451, 0.7140472
9: -6.3964601, -4.8880868, -6.3969994, -4.8874621, -0.7413983, 0.7541562

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4525613, upper bound: 0.4657893
time: 3.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4535888, upper bound: 0.4657893
time: 3.13 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 27.84 seconds
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4513967, upper bound: 0.4642171
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4524243, upper bound: 0.4642172
NS_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4639953
NS_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4650229
NS_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4517546, upper bound: 0.4657894
NS_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4527822, upper bound: 0.4657894
NS_A2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4525613, upper bound: 0.4657893
NS_A2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.84
Output dim: 0, lower bound: -0.4535888, upper bound: 0.4657893

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: 6.1332254, 7.4142008, 6.1395206, 7.3973980, -0.7151470, 0.7279248
1: -8.8132734, -7.1456566, -8.7554379, -7.1656179, -0.8680363, 0.8454443
2: -2.9730153, -1.7013980, -2.9662521, -1.7048497, -0.7535608, 0.7518930
3: -10.3781834, -9.0536356, -10.3533592, -9.1104288, -0.8297513, 0.8474011
4: -8.3415794, -6.9329252, -8.2833614, -6.9536076, -0.7827659, 0.7547483
5: -5.8640370, -4.9316926, -5.8610897, -4.9446907, -0.6486611, 0.6569824
6: -1.6027117, -0.3201747, -1.5759120, -0.3248720, -0.7966895, 0.7746048
7: -8.5023174, -6.7703128, -8.4967051, -6.7738037, -0.8888984, 0.8863297
8: -1.6970062, -0.7286248, -1.6862235, -0.7350950, -0.7033019, 0.6996634
9: -6.3917780, -4.8888283, -6.3603306, -4.9571905, -0.7244971, 0.7476838

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4513940, upper bound: 0.4631859
time: 3.06 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513940, upper bound: 0.4642170
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: 6.1299868, 7.4351406, 6.1395235, 7.3973970, -0.7218018, 0.7403371
1: -8.8323002, -7.1388407, -8.7554407, -7.1656218, -0.8695271, 0.8562953
2: -2.9882526, -1.7003648, -2.9662514, -1.7048498, -0.7683468, 0.7604980
3: -10.3798294, -9.0340729, -10.3533573, -9.1104279, -0.8357635, 0.8489230
4: -8.3461933, -6.9322767, -8.2833614, -6.9536085, -0.7840114, 0.7569253
5: -5.8768129, -4.9276490, -5.8610902, -4.9446936, -0.6548247, 0.6615129
6: -1.6194205, -0.3186536, -1.5759110, -0.3248730, -0.8052766, 0.7843654
7: -8.5042763, -6.7496109, -8.4967003, -6.7738042, -0.8954160, 0.9050494
8: -1.7028399, -0.7216511, -1.6862235, -0.7350950, -0.7082052, 0.7103376
9: -6.4030128, -4.8858604, -6.3603287, -4.9571915, -0.7267305, 0.7511853

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4631859
time: 3.16 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4524251, upper bound: 0.4642170
time: 3.27 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: 6.1302028, 7.4142923, 6.1342235, 7.3987284, -0.7198148, 0.7337584
1: -8.8136282, -7.1427517, -8.7575159, -7.1600542, -0.8720090, 0.8506620
2: -2.9745991, -1.7009505, -2.9721618, -1.7037194, -0.7563689, 0.7598751
3: -10.3794327, -9.0533943, -10.3550262, -9.1091423, -0.8328099, 0.8481441
4: -8.3419590, -6.9315634, -8.2850924, -6.9489584, -0.7874886, 0.7578721
5: -5.8642197, -4.9291911, -5.8633404, -4.9391832, -0.6510377, 0.6619332
6: -1.6036334, -0.3190975, -1.5789890, -0.3240576, -0.7981625, 0.7819736
7: -8.5061531, -6.7696271, -8.5049858, -6.7683396, -0.8984036, 0.8897252
8: -1.6979003, -0.7261958, -1.6905136, -0.7280350, -0.7082195, 0.7068162
9: -6.3964682, -4.8880858, -6.3739953, -4.9512539, -0.7341039, 0.7492814

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513939, upper bound: 0.4639926
time: 3.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4513939, upper bound: 0.4639924
time: 3.47 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: 6.1302061, 7.4142919, 6.1309628, 7.4196892, -0.7368393, 0.7404358
1: -8.8136272, -7.1427565, -8.7765751, -7.1532421, -0.8828251, 0.8520012
2: -2.9745986, -1.7009507, -2.9874156, -1.7026945, -0.7671032, 0.7734871
3: -10.3794308, -9.0533943, -10.3566723, -9.0895777, -0.8343284, 0.8541391
4: -8.3419600, -6.9315615, -8.2898035, -6.9483056, -0.7888541, 0.7621448
5: -5.8642197, -4.9291925, -5.8761191, -4.9351816, -0.6555204, 0.6642880
6: -1.6036339, -0.3190994, -1.5957513, -0.3225369, -0.8075495, 0.7955639
7: -8.5061474, -6.7696252, -8.5069466, -6.7476249, -0.9125941, 0.8962488
8: -1.6978984, -0.7261968, -1.6962843, -0.7210526, -0.7188921, 0.7116196
9: -6.3964672, -4.8880863, -6.3852363, -4.9482970, -0.7375925, 0.7515094

Time for backsubstitution: 21.93 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.95 + 561.47 = 618.42 seconds
