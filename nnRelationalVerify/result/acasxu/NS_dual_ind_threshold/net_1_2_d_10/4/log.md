## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 47.0393777385


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003)
1: (-10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643)
2: (-10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861)
3: (-15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287)
4: (-17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.50 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809205, upper bound: 47.1809205

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
time: 0.80 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -7.7389174, 26.9503441, -31.2792778, 23.3882599
1: -4.9227071, 18.1375618, -8.9462147, 31.2036190, -36.1263275, 27.0837765
2: -5.4355350, 17.6055393, -9.5667200, 30.5406857, -35.9762154, 27.1722584
3: -7.7107296, 19.0719624, -13.7645473, 32.7263031, -40.4370308, 32.8365059
4: -9.0420084, 16.7446327, -15.3124866, 29.4453449, -38.4873466, 32.0571213

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.48 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.67 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -8.7006226, 29.9893379, -43.8563232, 56.4315414
1: -16.4424095, 55.6123199, -10.0655127, 34.7013779, -51.1437759, 65.5285339
2: -17.0157623, 54.3669052, -10.6880798, 34.0129738, -51.0287285, 64.9423981
3: -25.0840302, 58.4735527, -15.4150457, 36.3897820, -61.4738121, 73.7443390
4: -27.2452755, 52.4822922, -16.9911537, 32.8825493, -60.1278229, 69.4734497

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1755632, upper bound: 47.1684286
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760125, upper bound: 47.1686230
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -13.0472050, 44.9861183, -49.1897354, 28.6965466
1: -4.9227071, 18.1375618, -15.3804827, 52.3511429, -57.1140938, 33.5180435
2: -5.4355350, 17.6055393, -16.0566273, 51.1194839, -56.4232712, 33.6621666
3: -7.7107296, 19.0719624, -23.5816936, 55.1130180, -62.6925659, 42.6536484
4: -9.0420084, 16.7446327, -25.8167171, 49.3024445, -58.3444481, 42.5613441

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1755632, upper bound: 47.1684286
time: 0.43 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760125, upper bound: 47.1686230
time: 0.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
time: 0.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -13.0686092, 45.0560722, -58.8123169, 60.7855453
1: -16.4424095, 55.6123199, -15.4047651, 52.4332733, -68.6767883, 70.8160324
2: -17.0157623, 54.3669052, -16.0822449, 51.1983566, -68.0413132, 70.2737198
3: -25.0840302, 58.4735527, -23.6169319, 55.1978226, -80.0572128, 81.8583908
4: -27.2452755, 52.4822922, -25.8539028, 49.3803787, -76.5030136, 78.2091293

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
time: 0.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.38 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1755632, upper bound: 47.1684286
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1760125, upper bound: 47.1686230
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1755632, upper bound: 47.1684286
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1760125, upper bound: 47.1686230
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1646659, upper bound: 47.1619752
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 4, lower bound: -47.1622348, upper bound: 47.1622348

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.1957455, 15.1547375, -4.3289332, 15.6493454, -19.8450909, 19.4836674
1: -4.7651372, 17.5528717, -4.9227071, 18.1375618, -22.9026985, 22.4755783
2: -5.2724891, 17.0362396, -5.4355350, 17.6055393, -22.8780251, 22.4717751
3: -7.4683981, 18.4647770, -7.7107296, 19.0719624, -26.5403595, 26.1755066
4: -8.7734575, 16.2064075, -9.0420084, 16.7446327, -25.5180874, 25.2484131

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.2849174, 15.4173374, -4.3289332, 15.6493454, -19.9342556, 19.7462673
1: -4.8534427, 17.8700066, -4.9227071, 18.1375618, -22.9910049, 22.7927132
2: -5.3807850, 17.3343678, -5.4355350, 17.6055393, -22.9863243, 22.7699032
3: -7.5899143, 18.7914963, -7.7107296, 19.0719624, -26.6618767, 26.5022259
4: -8.9349260, 16.4891949, -9.0420084, 16.7446327, -25.6795559, 25.5312004

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.1957455, 15.1547375, -13.0472050, 44.9861183, -49.0561523, 28.2019367
1: -4.7651372, 17.5528717, -15.3804827, 52.3511429, -56.9560852, 32.9333534
2: -5.2724891, 17.0362396, -16.0566273, 51.1194839, -56.2599945, 33.0928650
3: -7.4683981, 18.4647770, -23.5816936, 55.1130180, -62.4492340, 42.0464706
4: -8.7734575, 16.2064075, -25.8167171, 49.3024445, -58.0759010, 42.0231247

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.2849174, 15.4173374, -13.0472050, 44.9861183, -49.1478348, 28.4645405
1: -4.8534427, 17.8700066, -15.3804827, 52.3511429, -57.0458183, 33.2504883
2: -5.3807850, 17.3343678, -16.0566273, 51.1194839, -56.3696060, 33.3909950
3: -7.5899143, 18.7914963, -23.5816936, 55.1130180, -62.5730667, 42.3731918
4: -8.9349260, 16.4891949, -25.8167171, 49.3024445, -58.2373695, 42.3059120

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1713870, upper bound: 47.1647151
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.7235136, 44.1252480, -4.0675898, 14.8100910, -27.5336018, 48.0587921
1: -15.1153173, 51.2670212, -4.6172080, 17.1669483, -32.2822647, 55.7218132
2: -15.6082811, 50.1621857, -5.1129775, 16.6522598, -32.2605362, 55.1369324
3: -23.0708389, 53.8880844, -7.2489748, 18.0465260, -41.1173630, 60.9858627
4: -24.9927807, 48.4072723, -8.5434341, 15.8111591, -40.8039398, 56.9427338

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1650841, upper bound: 47.1740180
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652783, upper bound: 47.1745486
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.6422262, 47.0618324, -4.3289332, 15.6493454, -29.2915688, 51.2752151
1: -16.1728191, 54.7214432, -4.9227071, 18.1375618, -34.3103790, 59.4951324
2: -16.7433243, 53.4933968, -5.4355350, 17.6055393, -34.3488579, 58.8068466
3: -24.6783352, 57.5419312, -7.7107296, 19.0719624, -43.7502899, 65.1278000
4: -26.8195095, 51.6375656, -9.0420084, 16.7446327, -43.5641327, 60.6795731

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1624981, upper bound: 47.1751233
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1626886, upper bound: 47.1755132
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.7235136, 44.1252480, -12.7423143, 43.9971199, -56.5891113, 56.7596931
1: -15.1153173, 51.2670212, -15.0227795, 51.1977768, -66.0855331, 66.0907974
2: -15.6082811, 50.1621857, -15.6895828, 49.9927864, -65.4106674, 65.6766129
3: -23.0708389, 53.8880844, -23.0406990, 53.9002075, -76.7142639, 76.6854248
4: -24.9927807, 48.4072723, -25.2372856, 48.2131767, -73.0579071, 73.5071182

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1646462, upper bound: 47.1607082
time: 0.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619553
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.6422262, 47.0618324, -13.0686092, 45.0560722, -58.5865936, 60.0307312
1: -16.1728191, 54.7214432, -15.4047651, 52.4332733, -68.4088135, 69.9349899
2: -16.7433243, 53.4933968, -16.0822449, 51.1983566, -67.7683411, 69.4091187
3: -24.6783352, 57.5419312, -23.6169319, 55.1978226, -79.6553802, 80.9365921
4: -26.8195095, 51.6375656, -25.8539028, 49.3803787, -76.0805588, 77.3708496

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1622151, upper bound: 47.1607662
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1620134, upper bound: 47.1620134
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.67 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1801766
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1801766, upper bound: 47.1803884
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1804113
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1803884, upper bound: 47.1806231
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1669999, upper bound: 47.1636863
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1713870, upper bound: 47.1647151
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1650841, upper bound: 47.1740180
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1652783, upper bound: 47.1745486
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1624981, upper bound: 47.1751233
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1626886, upper bound: 47.1755132
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1646462, upper bound: 47.1607082
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1644444, upper bound: 47.1619553
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1622151, upper bound: 47.1607662
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 4, lower bound: -47.1620134, upper bound: 47.1620134

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.1957455, 15.1547375, -4.1957455, 15.1547375, -19.3504810, 19.3504829
1: -4.7651372, 17.5528717, -4.7651372, 17.5528717, -22.3180084, 22.3180084
2: -5.2724891, 17.0362396, -5.2724891, 17.0362396, -22.3087254, 22.3087254
3: -7.4683981, 18.4647770, -7.4683981, 18.4647770, -25.9331741, 25.9331741
4: -8.7734575, 16.2064075, -8.7734575, 16.2064075, -24.9798641, 24.9798641

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0904885, upper bound: 47.1183996
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1799103
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.1957455, 15.1547375, -4.2849174, 15.4173374, -19.6130829, 19.4396477
1: -4.7651372, 17.5528717, -4.8534427, 17.8700066, -22.6351433, 22.4063129
2: -5.2724891, 17.0362396, -5.3807850, 17.3343678, -22.6068497, 22.4170246
3: -7.4683981, 18.4647770, -7.5899143, 18.7914963, -26.2598953, 26.0546913
4: -8.7734575, 16.2064075, -8.9349260, 16.4891949, -25.2626514, 25.1413326

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0904885, upper bound: 47.1183996
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1801665
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2849174, 15.4173374, -4.1957455, 15.1547375, -19.4396477, 19.6130829
1: -4.8534427, 17.8700066, -4.7651372, 17.5528717, -22.4063148, 22.6351433
2: -5.3807850, 17.3343678, -5.2724891, 17.0362396, -22.4170246, 22.6068516
3: -7.5899143, 18.7914963, -7.4683981, 18.4647770, -26.0546913, 26.2598953
4: -8.9349260, 16.4891949, -8.7734575, 16.2064075, -25.1413326, 25.2626514

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1739304, upper bound: 47.1677654
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1669025
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2849174, 15.4173374, -4.2849174, 15.4173374, -19.7022495, 19.7022495
1: -4.8534427, 17.8700066, -4.8534427, 17.8700066, -22.7234478, 22.7234478
2: -5.3807850, 17.3343678, -5.3807850, 17.3343678, -22.7151527, 22.7151527
3: -7.5899143, 18.7914963, -7.5899143, 18.7914963, -26.3814106, 26.3814106
4: -8.9349260, 16.4891949, -8.9349260, 16.4891949, -25.4241199, 25.4241199

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1743499, upper bound: 47.1677654
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.1957455, 15.1547375, -12.7374916, 43.9600601, -48.0253868, 27.8922195
1: -4.7651372, 17.5528717, -15.0070114, 51.1597061, -55.7569504, 32.5598831
2: -5.2724891, 17.0362396, -15.6790133, 49.9404526, -55.0740204, 32.7152519
3: -7.4683981, 18.4647770, -23.0181007, 53.8667870, -61.1951599, 41.4828720
4: -8.7734575, 16.2064075, -25.2448215, 48.1450424, -56.9184990, 41.4512291

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1636863
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1636863
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.1786513, 15.0981417, -14.8774261, 50.6136818, -54.6290169, 29.9755669
1: -4.7437177, 17.4879913, -17.4754829, 58.9075317, -63.4401512, 34.9634743
2: -5.2521553, 16.9711304, -18.2150249, 57.5058784, -62.5781326, 35.1861572
3: -7.4366279, 18.3976631, -26.7148514, 61.9878883, -69.2384109, 45.1125107
4: -8.7429113, 16.1421490, -29.1225872, 55.5106049, -64.2334290, 45.2647362

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.2849174, 15.4173374, -12.7374916, 43.9600601, -48.1170731, 28.1548195
1: -4.8534427, 17.8700066, -15.0070114, 51.1597061, -55.8466797, 32.8770142
2: -5.3807850, 17.3343678, -15.6790133, 49.9404526, -55.1836319, 33.0133820
3: -7.5899143, 18.7914963, -23.0181007, 53.8667870, -61.3189926, 41.8095894
4: -8.9349260, 16.4891949, -25.2448215, 48.1450424, -57.0799637, 41.7340164

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.2685270, 15.3639469, -14.8774261, 50.6136818, -54.7215080, 30.2413731
1: -4.8332577, 17.8089714, -17.4754829, 58.9075317, -63.5311241, 35.2844505
2: -5.3614259, 17.2732334, -18.2150249, 57.5058784, -62.6886711, 35.4882584
3: -7.5599661, 18.7283268, -26.7148514, 61.9878883, -69.3640823, 45.4431763
4: -8.9059954, 16.4286861, -29.1225872, 55.5106049, -64.4022598, 45.5512733

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.7235136, 44.1252480, -3.9329579, 14.3158875, -27.0393963, 47.9237862
1: -15.1153173, 51.2670212, -4.4591498, 16.5783882, -31.6937065, 55.5634079
2: -15.6082811, 50.1621857, -4.9472294, 16.0807228, -31.6890030, 54.9708138
3: -23.0708389, 53.8880844, -7.0048699, 17.4327965, -40.5036354, 60.7406731
4: -24.9927807, 48.4072723, -8.2697201, 15.2698154, -40.2625961, 56.6664391

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563591, upper bound: 47.1715462
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563591, upper bound: 47.1740180
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.7235136, 44.1252480, -3.9689188, 14.4071035, -27.1306133, 47.9625664
1: -15.1153173, 51.2670212, -4.4856615, 16.7032986, -31.8186150, 55.5907898
2: -15.6082811, 50.1621857, -4.9947381, 16.1872997, -31.7955818, 55.0191841
3: -23.0708389, 53.8880844, -7.0344286, 17.5631390, -40.6339798, 60.7724800
4: -24.9927807, 48.4072723, -8.3418322, 15.3657484, -40.3585243, 56.7431374

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1565631, upper bound: 47.1722264
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1565631, upper bound: 47.1745486
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.6422262, 47.0618324, -4.1957455, 15.1547375, -28.7969608, 51.1416283
1: -16.1728191, 54.7214432, -4.7651372, 17.5528717, -33.7256927, 59.3371201
2: -16.7433243, 53.4933968, -5.2724891, 17.0362396, -33.7795563, 58.6435699
3: -24.6783352, 57.5419312, -7.4683981, 18.4647770, -43.1431122, 64.8844681
4: -26.8195095, 51.6375656, -8.7734575, 16.2064075, -43.0259171, 60.4110222

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13.6422262, 47.0618324, -4.2849174, 15.4173374, -29.0595627, 51.2333107
1: -16.1728191, 54.7214432, -4.8534427, 17.8700066, -34.0428238, 59.4268494
2: -16.7433243, 53.4933968, -5.3807850, 17.3343678, -34.0776901, 58.7531815
3: -24.6783352, 57.5419312, -7.5899143, 18.7914963, -43.4698334, 65.0083008
4: -26.8195095, 51.6375656, -8.9349260, 16.4891949, -43.3087006, 60.5724869

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679343
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.7235136, 44.1252480, -12.4301510, 42.9610252, -55.5463486, 56.4429512
1: -15.1153173, 51.2670212, -14.6452684, 49.9956398, -64.8739014, 65.7111130
2: -15.6082811, 50.1621857, -15.3084545, 48.8023529, -64.2123718, 65.2919693
3: -23.0708389, 53.8880844, -22.4719238, 52.6424446, -75.4476929, 76.1154709
4: -24.9927807, 48.4072723, -24.6611824, 47.0431328, -71.8832092, 72.9284134

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1344414
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1575481
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.7016439, 44.0515480, -14.5619154, 49.5908203, -62.1115685, 58.4875641
1: -15.0880919, 51.1823273, -17.1041660, 57.7127953, -72.5034637, 68.0823822
2: -15.5822115, 50.0774918, -17.8347073, 56.3376579, -71.6656342, 67.7260132
3: -23.0304356, 53.7997780, -26.1533985, 60.7304382, -83.4346237, 79.6957932
4: -24.9535313, 48.3245087, -28.5245361, 54.3785667, -79.1421204, 76.6960678

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634579, upper bound: 47.1367871
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1617991, upper bound: 47.1589411
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13.6422262, 47.0618324, -12.7545090, 44.0154343, -57.5409431, 59.7120857
1: -16.1728191, 54.7214432, -15.0263443, 51.2245064, -67.1912994, 69.5548172
2: -16.7433243, 53.4933968, -15.6993704, 50.0029755, -66.5655670, 69.0229721
3: -24.6783352, 57.5419312, -23.0462093, 53.9339752, -78.3832092, 80.3649750
4: -26.8195095, 51.6375656, -25.2743416, 48.2068672, -74.9033737, 76.7889404

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1607662
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1607662
time: 0.46 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13.6227036, 46.9964943, -14.8939104, 50.6654510, -64.1384277, 61.7753067
1: -16.1486626, 54.6462173, -17.4935570, 58.9680939, -74.8672867, 71.9459076
2: -16.7200089, 53.4183350, -18.2337780, 57.5640450, -74.0615387, 71.4773331
3: -24.6424618, 57.4634933, -26.7410030, 62.0500793, -86.4165192, 83.9701614
4: -26.7844238, 51.5642395, -29.1498108, 55.5681763, -82.2011642, 80.5790558

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134
time: 0.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.58 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.0904885, upper bound: 47.1183996
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1799103
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.0904885, upper bound: 47.1183996
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1799102, upper bound: 47.1801665
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1739304, upper bound: 47.1677654
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1669025
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1743499, upper bound: 47.1677654
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1712896
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1636863
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1636863
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1668931, upper bound: 47.1667574
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1647151
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1712801, upper bound: 47.1677862
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1563591, upper bound: 47.1715462
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1563591, upper bound: 47.1740180
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1565631, upper bound: 47.1722264
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1565631, upper bound: 47.1745486
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1679343
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1344414
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1575481
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.0634579, upper bound: 47.1367871
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1617991, upper bound: 47.1589411
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1607662
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1607662
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 4, lower bound: -47.1607662, upper bound: 47.1620134

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9499731, 14.3560972, -4.1223440, 14.9127922, -18.8627644, 18.4784393
1: -4.4719973, 16.6005650, -4.6758928, 17.2710228, -21.7430153, 21.2764568
2: -4.9373617, 16.1121292, -5.1788611, 16.7574921, -21.6948528, 21.2909870
3: -7.0062442, 17.3973083, -7.3296018, 18.1617737, -25.1680183, 24.7269096
4: -8.1801634, 15.2708197, -8.6207657, 15.9282513, -24.1084118, 23.8915844

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0767665, upper bound: 47.0767665
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0767665, upper bound: 47.1437740
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.0273848, 14.6178732, -4.1957455, 15.1547375, -19.1821175, 18.8136177
1: -4.5628810, 16.9356003, -4.7651372, 17.5528717, -22.1157532, 21.7007370
2: -5.0649624, 16.4214878, -5.2724891, 17.0362396, -22.1012020, 21.6939754
3: -7.1684437, 17.8108006, -7.4683981, 18.4647770, -25.6332188, 25.2791977
4: -8.4600706, 15.5963697, -8.7734575, 16.2064075, -24.6664772, 24.3698273

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1437740, upper bound: 47.0924772
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1437740, upper bound: 47.1799104
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9499731, 14.3560972, -4.2109075, 15.1732330, -19.1232033, 18.5670033
1: -4.4719973, 16.6005650, -4.7621074, 17.5860195, -22.0580120, 21.3626709
2: -4.9373617, 16.1121292, -5.2861915, 17.0529823, -21.9903450, 21.3983192
3: -7.0062442, 17.3973083, -7.4485340, 18.4853268, -25.4915714, 24.8458405
4: -8.1801634, 15.2708197, -8.7805204, 16.2081470, -24.3883095, 24.0513401

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0895802, upper bound: 47.1175230
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.0273848, 14.6178732, -4.2849174, 15.4173374, -19.4447193, 18.9027901
1: -4.5628810, 16.9356003, -4.8534427, 17.8700066, -22.4328880, 21.7890415
2: -5.0649624, 16.4214878, -5.3807850, 17.3343678, -22.3993301, 21.8022728
3: -7.1684437, 17.8108006, -7.5899143, 18.7914963, -25.9599380, 25.4007149
4: -8.4600706, 15.5963697, -8.9349260, 16.4891949, -24.9492645, 24.5312958

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1648539, upper bound: 47.1729075
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -4.1957455, 15.1547375, -19.2203579, 18.8864975
1: -4.5916448, 17.0283775, -4.7651372, 17.5528717, -22.1445160, 21.7935143
2: -5.1096640, 16.4983921, -5.2724891, 17.0362396, -22.1459045, 21.7708778
3: -7.1945591, 17.9123573, -7.4683981, 18.4647770, -25.6593342, 25.3807564
4: -8.5250015, 15.6656189, -8.7734575, 16.2064075, -24.7314091, 24.4390717

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -4.1786513, 15.0981417, -21.2974129, 25.3908920
1: -7.0349321, 24.6053638, -4.7437177, 17.4879913, -24.5229225, 29.3490810
2: -7.6394072, 23.9174843, -5.2521553, 16.9711304, -24.6105366, 29.1696396
3: -10.8425484, 25.8447590, -7.4366279, 18.3976631, -29.2402115, 33.2813873
4: -12.3402233, 22.9567204, -8.7429113, 16.1421490, -28.4823723, 31.6996307

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -4.2849174, 15.4173374, -19.4829559, 18.9756699
1: -4.5916448, 17.0283775, -4.8534427, 17.8700066, -22.4616489, 21.8818188
2: -5.1096640, 16.4983921, -5.3807850, 17.3343678, -22.4440308, 21.8791771
3: -7.1945591, 17.9123573, -7.5899143, 18.7914963, -25.9860535, 25.5022717
4: -8.5250015, 15.6656189, -8.9349260, 16.4891949, -25.0141964, 24.6005402

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -4.2685270, 15.3639469, -21.5632153, 25.4807663
1: -7.0349321, 24.6053638, -4.8332577, 17.8089714, -24.8439007, 29.4386215
2: -7.6394072, 23.9174843, -5.3614259, 17.2732334, -24.9126396, 29.2789097
3: -10.8425484, 25.8447590, -7.5599661, 18.7283268, -29.5708733, 33.4047241
4: -12.3402233, 22.9567204, -8.9059954, 16.4286861, -28.7689095, 31.8627167

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -12.7374916, 43.9600601, -47.7894096, 27.1171494
1: -4.4894509, 16.6531467, -15.0070114, 51.1597061, -55.4821892, 31.6601562
2: -4.9817009, 16.1492329, -15.6790133, 49.9404526, -54.7848320, 31.8282413
3: -7.0482631, 17.5268631, -23.0181007, 53.8667870, -60.7766800, 40.5449524
4: -8.3357964, 15.3324518, -25.2448215, 48.1450424, -56.4808388, 40.5772667

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -12.7374916, 43.9600601, -49.8982086, 33.6530190
1: -6.9165897, 24.2520409, -15.0070114, 51.1597061, -57.8971558, 39.2590408
2: -7.5073729, 23.5865192, -15.6790133, 49.9404526, -57.3107986, 39.2655334
3: -10.6918325, 25.4853954, -23.0181007, 53.8667870, -64.4134140, 48.5034943
4: -12.1613083, 22.6388607, -25.2448215, 48.1450424, -60.3006554, 47.8836823

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -14.8774261, 50.6136818, -54.4102554, 29.2570896
1: -4.4894509, 16.6531467, -17.4754829, 58.9075317, -63.1869011, 34.1286316
2: -4.9817009, 16.1492329, -18.2150249, 57.5058784, -62.3091736, 34.3642540
3: -7.0482631, 17.5268631, -26.7148514, 61.9878883, -68.8516388, 44.2417107
4: -8.3357964, 15.3324518, -29.1225872, 55.5106049, -63.8283310, 44.4550323

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1649457, upper bound: 47.1616335
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1649055
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -14.8774261, 50.6136818, -56.5190544, 35.7929611
1: -6.9165897, 24.2520409, -17.4754829, 58.9075317, -65.6018906, 41.7275124
2: -7.5073729, 23.5865192, -18.2150249, 57.5058784, -64.8351517, 41.8015442
3: -10.6918325, 25.4853954, -26.7148514, 61.9878883, -72.4883881, 52.2002487
4: -12.1613083, 22.6388607, -29.1225872, 55.5106049, -67.6382294, 51.7614479

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1649233, upper bound: 47.1600725
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1620842
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -12.7374916, 43.9600601, -47.8972397, 27.4282417
1: -4.5916448, 17.0283775, -15.0070114, 51.1597061, -55.5862427, 32.0353889
2: -5.1096640, 16.4983921, -15.6790133, 49.9404526, -54.9141846, 32.1774063
3: -7.1945591, 17.9123573, -23.0181007, 53.8667870, -60.9255486, 40.9304504
4: -8.5250015, 15.6656189, -25.2448215, 48.1450424, -56.6700401, 40.9104385

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1645569
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1647151
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -12.7374916, 43.9600601, -50.0180969, 33.9497299
1: -7.0349321, 24.6053638, -15.0070114, 51.1597061, -58.0107918, 39.6123695
2: -7.6394072, 23.9174843, -15.6790133, 49.9404526, -57.4444962, 39.5964966
3: -10.8425484, 25.8447590, -23.0181007, 53.8667870, -64.5653534, 48.8628540
4: -12.3402233, 22.9567204, -25.2448215, 48.1450424, -60.4843102, 48.2015419

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1645569
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1647151
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -14.8774261, 50.6136818, -54.5180855, 29.5681782
1: -4.5916448, 17.0283775, -17.4754829, 58.9075317, -63.2909508, 34.5038605
2: -5.1096640, 16.4983921, -18.2150249, 57.5058784, -62.4385262, 34.7134171
3: -7.1945591, 17.9123573, -26.7148514, 61.9878883, -69.0005341, 44.6272087
4: -8.5250015, 15.6656189, -29.1225872, 55.5106049, -64.0240250, 44.7882080

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1695807, upper bound: 47.1619300
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686647, upper bound: 47.1652020
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -14.8774261, 50.6136818, -56.6389427, 36.0896683
1: -7.0349321, 24.6053638, -17.4754829, 58.9075317, -65.7155075, 42.0808449
2: -7.6394072, 23.9174843, -18.2150249, 57.5058784, -64.9688416, 42.1325073
3: -10.8425484, 25.8447590, -26.7148514, 61.9878883, -72.6403275, 52.5596085
4: -12.3402233, 22.9567204, -29.1225872, 55.5106049, -67.8218689, 52.0793076

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1695807, upper bound: 47.1616649
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1648223
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.9329579, 14.3158875, -26.7683220, 46.9536743
1: -14.7858372, 50.1457520, -4.4591498, 16.5783882, -31.3642254, 54.4262009
2: -15.2835922, 49.0636292, -4.9472294, 16.0807228, -31.3643150, 53.8594437
3: -22.5760632, 52.7194977, -7.0048699, 17.4327965, -40.0088501, 59.5589485
4: -24.4824944, 47.3548775, -8.2697201, 15.2698154, -39.7523003, 55.6054077

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1647260
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1704367
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.9329579, 14.3158875, -26.8202248, 47.1297684
1: -14.8310118, 50.3433380, -4.4591498, 16.5783882, -31.4094009, 54.6400528
2: -15.3490219, 49.2404137, -4.9472294, 16.0807228, -31.4297447, 54.0499916
3: -22.6377640, 52.9216881, -7.0048699, 17.4327965, -40.0705605, 59.7783470
4: -24.5850430, 47.5082436, -8.2697201, 15.2698154, -39.8548584, 55.7689095

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1664033
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1720397
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.9689188, 14.4071035, -26.8595371, 46.9924507
1: -14.7858372, 50.1457520, -4.4856615, 16.7032986, -31.4891357, 54.4535789
2: -15.2835922, 49.0636292, -4.9947381, 16.1872997, -31.4708920, 53.9078140
3: -22.5760632, 52.7194977, -7.0344286, 17.5631390, -40.1391907, 59.5907555
4: -24.4824944, 47.3548775, -8.3418322, 15.3657484, -39.8482285, 55.6821060

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1673611
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1707618
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.9689188, 14.4071035, -26.9114399, 47.1685410
1: -14.8310118, 50.3433380, -4.4856615, 16.7032986, -31.5343094, 54.6674309
2: -15.3490219, 49.2404137, -4.9947381, 16.1872997, -31.5363216, 54.0983620
3: -22.6377640, 52.9216881, -7.0344286, 17.5631390, -40.2009010, 59.8101540
4: -24.5850430, 47.5082436, -8.3418322, 15.3657484, -39.9507866, 55.8456078

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1686385
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1721874
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -4.1957455, 15.1547375, -28.4894295, 50.1209717
1: -15.7996721, 53.5419350, -4.7651372, 17.5528717, -33.3525429, 58.1507683
2: -16.3717403, 52.3268585, -5.2724891, 17.0362396, -33.4079819, 57.4700089
3: -24.1192474, 56.3102837, -7.4683981, 18.4647770, -42.5840225, 63.6449127
4: -26.2549019, 50.4941635, -8.7734575, 16.2064075, -42.4613113, 59.2676201

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -4.1786513, 15.0981417, -30.5926514, 56.7729645
1: -18.2942677, 61.3451691, -4.7437177, 17.4879913, -35.7822571, 65.8901596
2: -18.9236336, 59.9523926, -5.2521553, 16.9711304, -35.8947639, 65.0365219
3: -27.8473854, 64.4847412, -7.4366279, 18.3976631, -46.2450409, 71.7442169
4: -30.1493359, 57.9222450, -8.7429113, 16.1421490, -46.2914848, 66.6508102

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -4.2849174, 15.4173374, -28.7520313, 50.2126541
1: -15.7996721, 53.5419350, -4.8534427, 17.8700066, -33.6696777, 58.2404976
2: -16.3717403, 52.3268585, -5.3807850, 17.3343678, -33.7061081, 57.5796242
3: -24.1192474, 56.3102837, -7.5899143, 18.7914963, -42.9107437, 63.7687454
4: -26.2549019, 50.4941635, -8.9349260, 16.4891949, -42.7440948, 59.4290848

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -4.2685270, 15.3639469, -30.8584557, 56.8654556
1: -18.2942677, 61.3451691, -4.8332577, 17.8089714, -36.1032333, 65.9811478
2: -18.9236336, 59.9523926, -5.3614259, 17.2732334, -36.1968689, 65.1470718
3: -27.8473854, 64.4847412, -7.5599661, 18.7283268, -46.5757065, 71.8698959
4: -30.1493359, 57.9222450, -8.9059954, 16.4286861, -46.5780220, 66.8196487

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.6461382, 43.8475800, -12.3644972, 42.7447701, -55.2383652, 56.0856018
1: -15.0353851, 50.8989334, -14.5658293, 49.7420044, -64.5354538, 65.2443008
2: -15.4672871, 49.8316040, -15.2262402, 48.5527840, -63.8126450, 64.8609848
3: -22.8897457, 53.4193306, -22.3490353, 52.3676338, -74.9854279, 75.5059586
4: -24.6493130, 48.0615349, -24.5220108, 46.7958031, -71.2848663, 72.4256287

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1344414
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634251, upper bound: 47.1288935
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.3680058, 42.9620438, -12.4301510, 42.9610252, -55.1873207, 55.2708549
1: -14.6970406, 49.9281464, -14.6452684, 49.9956398, -64.4532776, 64.3575211
2: -15.1878767, 48.8440247, -15.3084545, 48.8023529, -63.7858315, 63.9606972
3: -22.4419918, 52.4952850, -22.4719238, 52.6424446, -74.8168640, 74.7112274
4: -24.3620014, 47.1144180, -24.6611824, 47.0431328, -71.2483368, 71.6275635

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1006516, upper bound: 47.0786965
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1006516, upper bound: 47.1575481
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.6256218, 43.7787056, -14.4895830, 49.3510590, -61.7817497, 58.1269226
1: -15.0098400, 50.8197365, -17.0130863, 57.4321022, -72.1398239, 67.6090469
2: -15.4425802, 49.7524643, -17.7422733, 56.0596657, -71.2390060, 67.2900238
3: -22.8520451, 53.3366394, -26.0131588, 60.4258575, -82.9455719, 79.0742416
4: -24.6123676, 47.9842224, -28.3715267, 54.1007576, -78.5157242, 76.1846924

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634579, upper bound: 47.1367871
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0627673, upper bound: 47.1211938
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.3462753, 42.8892822, -14.5619154, 49.5908203, -61.7526436, 57.3163872
1: -14.6699352, 49.8443909, -17.1041660, 57.7127953, -72.0829773, 66.7297592
2: -15.1618128, 48.7604256, -17.8347073, 56.3376579, -71.2390823, 66.3958893
3: -22.4018936, 52.4078178, -26.1533985, 60.7304382, -82.8041077, 78.2923431
4: -24.3229198, 47.0327644, -28.5245361, 54.3785667, -78.5073929, 75.3963242

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633490, upper bound: 47.1589411
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -12.7545090, 44.0154343, -57.2297707, 58.6914291
1: -15.7996721, 53.5419350, -15.0263443, 51.2245064, -66.8164825, 68.3684692
2: -16.3717403, 52.3268585, -15.6993704, 50.0029755, -66.1911087, 67.8494110
3: -24.1192474, 56.3102837, -23.0462093, 53.9339752, -77.8234177, 79.1254272
4: -26.2549019, 50.4941635, -25.2743416, 48.2068672, -74.3367462, 75.6402130

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1079512, upper bound: 47.0844995
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1577168, upper bound: 47.1575763
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -12.7545090, 44.0154343, -59.3794365, 65.3606262
1: -18.2942677, 61.3451691, -15.0263443, 51.2245064, -69.3087158, 76.1293793
2: -18.9236336, 59.9523926, -15.6993704, 50.0029755, -68.7391281, 75.4361572
3: -27.8473854, 64.4847412, -23.0462093, 53.9339752, -81.5414810, 87.2564468
4: -30.1493359, 57.9222450, -25.2743416, 48.2068672, -78.2205429, 83.0423508

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1079512, upper bound: 47.0844995
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1577168, upper bound: 47.1575763
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -14.8939104, 50.6654510, -63.8467789, 60.8199844
1: -15.7996721, 53.5419350, -17.4935570, 58.9680939, -74.5166702, 70.8347778
2: -16.3717403, 52.3268585, -18.2337780, 57.5640450, -73.7104263, 70.3787842
3: -24.1192474, 56.3102837, -26.7410030, 62.0500793, -85.8926163, 82.8090057
4: -26.2549019, 50.4941635, -29.1498108, 55.5681763, -81.6696472, 79.5036774

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1613683
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1584041, upper bound: 47.1557092
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1574411, upper bound: 47.1589705
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -14.8939104, 50.6654510, -65.9964447, 67.4891891
1: -18.2942677, 61.3451691, -17.4935570, 58.9680939, -77.0089035, 78.5956726
2: -18.9236336, 59.9523926, -18.2337780, 57.5640450, -76.2584381, 77.9655304
3: -27.8473854, 64.4847412, -26.7410030, 62.0500793, -89.6106720, 90.9400330
4: -30.1493359, 57.9222450, -29.1498108, 55.5681763, -85.5534363, 86.9058075

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1607882
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345
time: 0.53 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.24 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0767665, upper bound: 47.0767665
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0767665, upper bound: 47.1437740
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1437740, upper bound: 47.0924772
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1437740, upper bound: 47.1799104
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1648539, upper bound: 47.1729075
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1658737, upper bound: 47.1658737
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1702607, upper bound: 47.1669025
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1708973, upper bound: 47.1712896
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1635280
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1663147, upper bound: 47.1636863
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1649457, upper bound: 47.1616335
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1649055
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1649233, upper bound: 47.1600725
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1620842
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1645569
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1647151
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1645569
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1707017, upper bound: 47.1647151
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1695807, upper bound: 47.1619300
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1686647, upper bound: 47.1652020
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1695807, upper bound: 47.1616649
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1640298, upper bound: 47.1648223
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1647260
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1704367
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1664033
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1720397
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1673611
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1707618
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1539809, upper bound: 47.1686385
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1532511, upper bound: 47.1721874
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1630569
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1603417, upper bound: 47.1630569
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1603529, upper bound: 47.1677345
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1615719, upper bound: 47.1677345
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1344414
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0634251, upper bound: 47.1288935
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1006516, upper bound: 47.0786965
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1006516, upper bound: 47.1575481
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0634579, upper bound: 47.1367871
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0627673, upper bound: 47.1211938
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1617567, upper bound: 47.1589411
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.0633490, upper bound: 47.1589411
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1079512, upper bound: 47.0844995
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1577168, upper bound: 47.1575763
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1079512, upper bound: 47.0844995
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1577168, upper bound: 47.1575763
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1584041, upper bound: 47.1557092
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1574411, upper bound: 47.1589705
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1459795, upper bound: 47.1607882
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 4, lower bound: -47.1441747, upper bound: 47.1453345

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.9499731, 14.3560972, -3.9499731, 14.3560972, -18.3060665, 18.3060665
1: -4.4719973, 16.6005650, -4.4719973, 16.6005650, -21.0725574, 21.0725574
2: -4.9373617, 16.1121292, -4.9373617, 16.1121292, -21.0494900, 21.0494900
3: -7.0062442, 17.3973083, -7.0062442, 17.3973083, -24.4035530, 24.4035530
4: -8.1801634, 15.2708197, -8.1801634, 15.2708197, -23.4509830, 23.4509830

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0387314, upper bound: 47.0400160
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0388421, upper bound: 47.0388421
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9499731, 14.3560972, -4.0273848, 14.6178732, -18.5678463, 18.3834801
1: -4.4719973, 16.6005650, -4.5628810, 16.9356003, -21.4075966, 21.1634464
2: -4.9373617, 16.1121292, -5.0649624, 16.4214878, -21.3588486, 21.1770916
3: -7.0062442, 17.3973083, -7.1684437, 17.8108006, -24.8170414, 24.5657520
4: -8.1801634, 15.2708197, -8.4600706, 15.5963697, -23.7765331, 23.7308903

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0387314, upper bound: 47.0954325
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0388421, upper bound: 47.1061951
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.0273848, 14.6178732, -3.9499731, 14.3560972, -18.3834801, 18.5678463
1: -4.5628810, 16.9356003, -4.4719973, 16.6005650, -21.1634464, 21.4075947
2: -5.0649624, 16.4214878, -4.9373617, 16.1121292, -21.1770916, 21.3588486
3: -7.1684437, 17.8108006, -7.0062442, 17.3973083, -24.5657520, 24.8170433
4: -8.4600706, 15.5963697, -8.1801634, 15.2708197, -23.7308903, 23.7765331

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1110884, upper bound: 47.0650937
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1061950, upper bound: 47.0653095
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.0273848, 14.6178732, -4.0273848, 14.6178732, -18.6452579, 18.6452579
1: -4.5628810, 16.9356003, -4.5628810, 16.9356003, -21.4984818, 21.4984818
2: -5.0649624, 16.4214878, -5.0649624, 16.4214878, -21.4864502, 21.4864502
3: -7.1684437, 17.8108006, -7.1684437, 17.8108006, -24.9792404, 24.9792423
4: -8.4600706, 15.5963697, -8.4600706, 15.5963697, -24.0564404, 24.0564404

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1110885, upper bound: 47.1559525
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1061951, upper bound: 47.1536603
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.0273848, 14.6178732, -4.0656214, 14.6907520, -18.7181358, 18.6834946
1: -4.5628810, 16.9356003, -4.5916448, 17.0283775, -21.5912590, 21.5272427
2: -5.0649624, 16.4214878, -5.1096640, 16.4983921, -21.5633545, 21.5311508
3: -7.1684437, 17.8108006, -7.1945591, 17.9123573, -25.0808010, 25.0053558
4: -8.4600706, 15.5963697, -8.5250015, 15.6656189, -24.1256905, 24.1213722

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0111818, 14.5649567, -6.1992731, 21.2122402, -25.2234230, 20.7642269
1: -4.5426331, 16.8749542, -7.0349321, 24.6053638, -29.1479969, 23.9098854
2: -5.0455117, 16.3607502, -7.6394072, 23.9174843, -28.9629955, 24.0001564
3: -7.1382613, 17.7478085, -10.8425484, 25.8447590, -32.9830132, 28.5903530
4: -8.4311571, 15.5360546, -12.3402233, 22.9567204, -31.3878765, 27.8762779

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1640355, upper bound: 47.1673695
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -3.9600012, 14.3796635, -18.4452858, 18.6507530
1: -4.5916448, 17.0283775, -4.4894509, 16.6531467, -21.2447891, 21.5178280
2: -5.1096640, 16.4983921, -4.9817009, 16.1492329, -21.2588959, 21.4800930
3: -7.1945591, 17.9123573, -7.0482631, 17.5268631, -24.7214203, 24.9606171
4: -8.5250015, 15.6656189, -8.3357964, 15.3324518, -23.8574524, 24.0014133

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1702456, upper bound: 47.1658456
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677526, upper bound: 47.1650639
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -6.0743117, 20.9155369, -24.9811573, 20.7650642
1: -4.5916448, 17.0283775, -6.9165897, 24.2520409, -28.8436832, 23.9449654
2: -5.1096640, 16.4983921, -7.5073729, 23.5865192, -28.6961823, 24.0057640
3: -7.1945591, 17.9123573, -10.6918325, 25.4853954, -32.6799545, 28.6041908
4: -8.5250015, 15.6656189, -12.1613083, 22.6388607, -31.1638622, 27.8269253

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677526, upper bound: 47.1658456
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1730694, upper bound: 47.1650639
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -3.9600012, 14.3796635, -20.5789337, 25.1722393
1: -7.0349321, 24.6053638, -4.4894509, 16.6531467, -23.6880779, 29.0948143
2: -7.6394072, 23.9174843, -4.9817009, 16.1492329, -23.7886391, 28.8991852
3: -10.8425484, 25.8447590, -7.0482631, 17.5268631, -28.3694096, 32.8930168
4: -12.3402233, 22.9567204, -8.3357964, 15.3324518, -27.6726761, 31.2925148

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652912, upper bound: 47.1649233
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686617, upper bound: 47.1643232
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -6.0743117, 20.9155369, -27.1148052, 27.2865505
1: -7.0349321, 24.6053638, -6.9165897, 24.2520409, -31.2869720, 31.5219498
2: -7.6394072, 23.9174843, -7.5073729, 23.5865192, -31.2259254, 31.4248581
3: -10.8425484, 25.8447590, -10.6918325, 25.4853954, -36.3279419, 36.5365906
4: -12.3402233, 22.9567204, -12.1613083, 22.6388607, -34.9790840, 35.1180267

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661550, upper bound: 47.1632359
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1619832, upper bound: 47.1619832
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -4.0656214, 14.6907520, -18.7563744, 18.7563744
1: -4.5916448, 17.0283775, -4.5916448, 17.0283775, -21.6200199, 21.6200199
2: -5.1096640, 16.4983921, -5.1096640, 16.4983921, -21.6080551, 21.6080551
3: -7.1945591, 17.9123573, -7.1945591, 17.9123573, -25.1069145, 25.1069145
4: -8.5250015, 15.6656189, -8.5250015, 15.6656189, -24.1906185, 24.1906166

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712697, upper bound: 47.1705583
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1733072, upper bound: 47.1698127
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -6.1992731, 21.2122402, -25.2778625, 20.8900261
1: -4.5916448, 17.0283775, -7.0349321, 24.6053638, -29.1970081, 24.0633087
2: -5.1096640, 16.4983921, -7.6394072, 23.9174843, -29.0271492, 24.1377983
3: -7.1945591, 17.9123573, -10.8425484, 25.8447590, -33.0393143, 28.7549057
4: -8.5250015, 15.6656189, -12.3402233, 22.9567204, -31.4817200, 28.0058403

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712697, upper bound: 47.1705583
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1733072, upper bound: 47.1698127
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -4.0656214, 14.6907520, -20.8900261, 25.2778625
1: -7.0349321, 24.6053638, -4.5916448, 17.0283775, -24.0633087, 29.1970062
2: -7.6394072, 23.9174843, -5.1096640, 16.4983921, -24.1377983, 29.0271492
3: -10.8425484, 25.8447590, -7.1945591, 17.9123573, -28.7549057, 33.0393143
4: -12.3402233, 22.9567204, -8.5250015, 15.6656189, -28.0058403, 31.4817200

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1665553, upper bound: 47.1679135
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670535, upper bound: 47.1678911
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -6.1992731, 21.2122402, -27.4115124, 27.4115124
1: -7.0349321, 24.6053638, -7.0349321, 24.6053638, -31.6402969, 31.6402912
2: -7.6394072, 23.9174843, -7.6394072, 23.9174843, -31.5568924, 31.5568924
3: -10.8425484, 25.8447590, -10.8425484, 25.8447590, -36.6873055, 36.6873055
4: -12.3402233, 22.9567204, -12.3402233, 22.9567204, -35.2969437, 35.2969398

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1665553, upper bound: 47.1679135
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670535, upper bound: 47.1678911
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -12.5137920, 43.1676254, -46.9864006, 26.8934517
1: -4.4894509, 16.6531467, -14.7390337, 50.2288094, -54.5382080, 31.3921757
2: -4.9817009, 16.1492329, -15.4095001, 49.0306587, -53.8612251, 31.5587292
3: -7.0482631, 17.5268631, -22.6095047, 52.8917618, -59.7878990, 40.1363678
4: -8.3357964, 15.3324518, -24.8135319, 47.2763710, -55.6121674, 40.1459694

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1672964, upper bound: 47.1625631
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1726947, upper bound: 47.1619907
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -12.5754843, 43.3694687, -47.1999626, 26.9551468
1: -4.4894509, 16.6531467, -14.7950506, 50.4736938, -54.7993584, 31.4481926
2: -4.9817009, 16.1492329, -15.4829912, 49.2483826, -54.0908127, 31.6322193
3: -7.0482631, 17.5268631, -22.6857243, 53.1380005, -60.0503998, 40.2125854
4: -8.3357964, 15.3324518, -24.9261703, 47.4630890, -55.7988853, 40.2586098

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1672964, upper bound: 47.1625631
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1726947, upper bound: 47.1619907
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -12.5137920, 43.1676254, -49.0951996, 33.4293251
1: -6.9165897, 24.2520409, -14.7390337, 50.2288094, -56.9531746, 38.9910622
2: -7.5073729, 23.5865192, -15.4095001, 49.0306587, -56.3871880, 38.9960175
3: -10.6918325, 25.4853954, -22.6095047, 52.8917618, -63.4246445, 48.0949020
4: -12.1613083, 22.6388607, -24.8135319, 47.2763710, -59.4229927, 47.4523926

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -12.5754843, 43.3694687, -49.3087616, 33.4910202
1: -6.9165897, 24.2520409, -14.7950506, 50.4736938, -57.2143288, 39.0470886
2: -7.5073729, 23.5865192, -15.4829912, 49.2483826, -56.6167755, 39.0695076
3: -10.6918325, 25.4853954, -22.6857243, 53.1380005, -63.6871452, 48.1711197
4: -12.1613083, 22.6388607, -24.9261703, 47.4630890, -59.6211853, 47.5650291

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -14.5762272, 49.6765594, -53.4659386, 28.9558887
1: -4.4894509, 16.6531467, -17.1232414, 57.8143005, -62.0841980, 33.7763901
2: -4.9817009, 16.1492329, -17.8523102, 56.4393082, -61.2350121, 34.0015297
3: -7.0482631, 17.5268631, -26.1893234, 60.8468552, -67.7020111, 43.7161865
4: -8.3357964, 15.3324518, -28.5748024, 54.4696541, -62.7820091, 43.9072456

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671161, upper bound: 47.1625279
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671161, upper bound: 47.1625279
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9600012, 14.3796635, -14.6747475, 50.0361366, -53.8473320, 29.0544109
1: -4.4894509, 16.6531467, -17.2565861, 58.2409706, -62.5381546, 33.9097290
2: -4.9817009, 16.1492329, -17.9725628, 56.8512306, -61.6701393, 34.1217957
3: -7.0482631, 17.5268631, -26.3906784, 61.2944298, -68.1756363, 43.9175415
4: -8.3357964, 15.3324518, -28.7681179, 54.8636856, -63.1929779, 44.1005707

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671161, upper bound: 47.1658045
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1671161, upper bound: 47.1658045
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -14.5762272, 49.6765594, -55.5747375, 35.4917603
1: -6.9165897, 24.2520409, -17.1232414, 57.8143005, -64.4991684, 41.3752747
2: -7.5073729, 23.5865192, -17.8523102, 56.4393082, -63.7609749, 41.4388199
3: -10.6918325, 25.4853954, -26.1893234, 60.8468552, -71.3387680, 51.6747208
4: -12.1613083, 22.6388607, -28.5748024, 54.4696541, -66.5919113, 51.2136612

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.0743117, 20.9155369, -14.6747475, 50.0361366, -55.9561310, 35.5902824
1: -6.9165897, 24.2520409, -17.2565861, 58.2409706, -64.9531326, 41.5086098
2: -7.5073729, 23.5865192, -17.9725628, 56.8512306, -64.1961136, 41.5590820
3: -10.6918325, 25.4853954, -26.3906784, 61.2944298, -71.8123932, 51.8760757
4: -12.1613083, 22.6388607, -28.7681179, 54.8636856, -67.0028687, 51.4069786

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -12.5137920, 43.1676254, -47.0942268, 27.2045441
1: -4.5916448, 17.0283775, -14.7390337, 50.2288094, -54.6422577, 31.7674065
2: -5.1096640, 16.4983921, -15.4095001, 49.0306587, -53.9905777, 31.9078922
3: -7.1945591, 17.9123573, -22.6095047, 52.8917618, -59.9367676, 40.5218620
4: -8.5250015, 15.6656189, -24.8135319, 47.2763710, -55.8013687, 40.4791451

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1672964, upper bound: 47.1628022
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1727233, upper bound: 47.1619906
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -12.5754843, 43.3694687, -47.3077888, 27.2662354
1: -4.5916448, 17.0283775, -14.7950506, 50.4736938, -54.9034119, 31.8234234
2: -5.1096640, 16.4983921, -15.4829912, 49.2483826, -54.2201691, 31.9813824
3: -7.1945591, 17.9123573, -22.6857243, 53.1380005, -60.1992683, 40.5980835
4: -8.5250015, 15.6656189, -24.9261703, 47.4630890, -55.9880867, 40.5917854

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1698649, upper bound: 47.1628022
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1727233, upper bound: 47.1619906
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -12.5137920, 43.1676254, -49.2150841, 33.7260323
1: -7.0349321, 24.6053638, -14.7390337, 50.2288094, -57.0668182, 39.3443871
2: -7.6394072, 23.9174843, -15.4095001, 49.0306587, -56.5208893, 39.3269844
3: -10.8425484, 25.8447590, -22.6095047, 52.8917618, -63.5765762, 48.4542618
4: -12.3402233, 22.9567204, -24.8135319, 47.2763710, -59.6066475, 47.7702446

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652465, upper bound: 47.1619328
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686169, upper bound: 47.1613327
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -12.5754843, 43.3694687, -49.4286499, 33.7877235
1: -7.0349321, 24.6053638, -14.7950506, 50.4736938, -57.3279648, 39.4004135
2: -7.6394072, 23.9174843, -15.4829912, 49.2483826, -56.7504768, 39.4004745
3: -10.8425484, 25.8447590, -22.6857243, 53.1380005, -63.8390770, 48.5304832
4: -12.3402233, 22.9567204, -24.9261703, 47.4630890, -59.8033104, 47.8828850

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652465, upper bound: 47.1619328
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1686169, upper bound: 47.1613327
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -14.5762272, 49.6765594, -53.5737648, 29.2669792
1: -4.5916448, 17.0283775, -17.1232414, 57.8143005, -62.1882515, 34.1516190
2: -5.1096640, 16.4983921, -17.8523102, 56.4393082, -61.3643684, 34.3506889
3: -7.1945591, 17.9123573, -26.1893234, 60.8468552, -67.8509064, 44.1016808
4: -8.5250015, 15.6656189, -28.5748024, 54.4696541, -62.9777069, 44.2404213

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1694553, upper bound: 47.1625288
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1694553, upper bound: 47.1625288
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0656214, 14.6907520, -14.6747475, 50.0361366, -53.9551582, 29.3654995
1: -4.5916448, 17.0283775, -17.2565861, 58.2409706, -62.6422081, 34.2849541
2: -5.1096640, 16.4983921, -17.9725628, 56.8512306, -61.7994919, 34.4709549
3: -7.1945591, 17.9123573, -26.3906784, 61.2944298, -68.3245316, 44.3030357
4: -8.5250015, 15.6656189, -28.7681179, 54.8636856, -63.3886757, 44.4337387

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1694553, upper bound: 47.1658052
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1694553, upper bound: 47.1658052
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -14.5762272, 49.6765594, -55.6946259, 35.7884674
1: -7.0349321, 24.6053638, -17.1232414, 57.8143005, -64.6127930, 41.7286034
2: -7.6394072, 23.9174843, -17.8523102, 56.4393082, -63.8946724, 41.7697868
3: -10.8425484, 25.8447590, -26.1893234, 60.8468552, -71.4906998, 52.0340805
4: -12.3402233, 22.9567204, -28.5748024, 54.4696541, -66.7755508, 51.5315208

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1992731, 21.2122402, -14.6747475, 50.0361366, -56.0760155, 35.8869858
1: -7.0349321, 24.6053638, -17.2565861, 58.2409706, -65.0667572, 41.8619385
2: -7.6394072, 23.9174843, -17.9725628, 56.8512306, -64.3298111, 41.8900452
3: -10.8425484, 25.8447590, -26.3906784, 61.2944298, -71.9643250, 52.2354355
4: -12.3402233, 22.9567204, -28.7681179, 54.8636856, -67.1865082, 51.7248383

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652942, upper bound: 47.1647890
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1652942, upper bound: 47.1648223
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.6725693, 13.5066309, -25.9590645, 46.6884346
1: -14.7858372, 50.1457520, -4.1583838, 15.6416941, -30.4275322, 54.1236954
2: -15.2835922, 49.0636292, -4.6303711, 15.1652069, -30.4487991, 53.5391350
3: -22.5760632, 52.7194977, -6.5486879, 16.4483948, -39.0244560, 59.1030426
4: -24.4824944, 47.3548775, -7.7941556, 14.3680058, -38.8505020, 55.1257706

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539197, upper bound: 47.1611466
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1287511, upper bound: 47.1557170
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.7931173, 13.9393730, -26.3918076, 46.8162422
1: -14.7858372, 50.1457520, -4.3157716, 16.1453743, -30.9312115, 54.2828674
2: -15.2835922, 49.0636292, -4.7731657, 15.6630831, -30.9466724, 53.6862793
3: -22.5760632, 52.7194977, -6.7912087, 16.9764385, -39.5524940, 59.3464127
4: -24.4824944, 47.3548775, -8.0259409, 14.8444347, -39.3269234, 55.3667221

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1530142, upper bound: 47.1604907
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1279684, upper bound: 47.1559060
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.6725693, 13.5066309, -26.0109711, 46.8645287
1: -14.8310118, 50.3433380, -4.1583838, 15.6416941, -30.4727058, 54.3375473
2: -15.3490219, 49.2404137, -4.6303711, 15.1652069, -30.5142288, 53.7296829
3: -22.6377640, 52.9216881, -6.5486879, 16.4483948, -39.0861588, 59.3224411
4: -24.5850430, 47.5082436, -7.7941556, 14.3680058, -38.9530487, 55.2892723

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539197, upper bound: 47.1625551
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1281530, upper bound: 47.1557528
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.7931173, 13.9393730, -26.4437084, 46.9923363
1: -14.8310118, 50.3433380, -4.3157716, 16.1453743, -30.9763870, 54.4967194
2: -15.3490219, 49.2404137, -4.7731657, 15.6630831, -31.0121040, 53.8768272
3: -22.6377640, 52.9216881, -6.7912087, 16.9764385, -39.6142044, 59.5658112
4: -24.5850430, 47.5082436, -8.0259409, 14.8444347, -39.4294777, 55.5302238

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1530142, upper bound: 47.1616999
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1273686, upper bound: 47.1559396
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.7198942, 13.6461706, -26.0986042, 46.7382431
1: -14.7858372, 50.1457520, -4.1976194, 15.8170490, -30.6028862, 54.1650696
2: -15.2835922, 49.0636292, -4.6927671, 15.3243132, -30.6079025, 53.6032143
3: -22.5760632, 52.7194977, -6.5995531, 16.6336346, -39.2096939, 59.1563530
4: -24.4824944, 47.3548775, -7.8902025, 14.5124760, -38.9949684, 55.2267380

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1540401, upper bound: 47.1609397
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1291870, upper bound: 47.1557430
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.4524336, 43.1678200, -3.8021708, 13.9561367, -26.4085693, 46.8266029
1: -14.7858372, 50.1457520, -4.3070755, 16.1786804, -30.9645176, 54.2748642
2: -15.2835922, 49.0636292, -4.7891006, 15.6824722, -30.9660645, 53.7035370
3: -22.5760632, 52.7194977, -6.7690907, 17.0041504, -39.5802040, 59.3269043
4: -24.4824944, 47.3548775, -8.0424089, 14.8515644, -39.3340492, 55.3913422

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1244081, upper bound: 47.1648016
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1244081, upper bound: 47.1707619
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.7198942, 13.6461706, -26.1505089, 46.9143333
1: -14.8310118, 50.3433380, -4.1976194, 15.8170490, -30.6480598, 54.3789215
2: -15.3490219, 49.2404137, -4.6927671, 15.3243132, -30.6733341, 53.7937622
3: -22.6377640, 52.9216881, -6.5995531, 16.6336346, -39.2714005, 59.3757515
4: -24.5850430, 47.5082436, -7.8902025, 14.5124760, -39.0975189, 55.3902397

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1539197, upper bound: 47.1623261
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1281530, upper bound: 47.1557756
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.5043392, 43.3323669, -3.8021708, 13.9561367, -26.4604759, 47.0026970
1: -14.8310118, 50.3433380, -4.3070755, 16.1786804, -31.0096912, 54.4887161
2: -15.3490219, 49.2404137, -4.7891006, 15.6824722, -31.0314941, 53.8940849
3: -22.6377640, 52.9216881, -6.7690907, 17.0041504, -39.6419144, 59.5463028
4: -24.5850430, 47.5082436, -8.0424089, 14.8515644, -39.4366074, 55.5506516

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591842, upper bound: 47.1721875
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591842, upper bound: 47.1721875
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -3.9600012, 14.3796635, -27.7143593, 49.8849945
1: -15.7996721, 53.5419350, -4.4894509, 16.6531467, -32.4528198, 57.8760071
2: -16.3717403, 52.3268585, -4.9817009, 16.1492329, -32.5209732, 57.1808205
3: -24.1192474, 56.3102837, -7.0482631, 17.5268631, -41.6461105, 63.2264290
4: -26.2549019, 50.4941635, -8.3357964, 15.3324518, -41.5873489, 58.8299599

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1626661
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -6.0743117, 20.9155369, -34.2502251, 51.9937935
1: -15.7996721, 53.5419350, -6.9165897, 24.2520409, -40.0517044, 60.2909775
2: -16.3717403, 52.3268585, -7.5073729, 23.5865192, -39.9582596, 59.7067833
3: -24.1192474, 56.3102837, -10.6918325, 25.4853954, -49.6046448, 66.8631744
4: -26.2549019, 50.4941635, -12.1613083, 22.6388607, -48.8937607, 62.6534843

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1626661
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1632567
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -3.9600012, 14.3796635, -29.8741703, 56.5542030
1: -18.2942677, 61.3451691, -4.4894509, 16.6531467, -34.9474144, 65.6369095
2: -18.9236336, 59.9523926, -4.9817009, 16.1492329, -35.0728683, 64.7675781
3: -27.8473854, 64.4847412, -7.0482631, 17.5268631, -45.3742371, 71.3574371
4: -30.1493359, 57.9222450, -8.3357964, 15.3324518, -45.4817848, 66.2457199

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1598706, upper bound: 47.1619827
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1598706, upper bound: 47.1630569
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -6.0743117, 20.9155369, -36.4100380, 58.6630058
1: -18.2942677, 61.3451691, -6.9165897, 24.2520409, -42.5462952, 68.0518951
2: -18.9236336, 59.9523926, -7.5073729, 23.5865192, -42.5101547, 67.2935486
3: -27.8473854, 64.4847412, -10.6918325, 25.4853954, -53.3327789, 74.9941864
4: -30.1493359, 57.9222450, -12.1613083, 22.6388607, -52.7881966, 70.0556107

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1598706, upper bound: 47.1619827
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1598706, upper bound: 47.1630569
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -4.0656214, 14.6907520, -28.0254478, 49.9928207
1: -15.7996721, 53.5419350, -4.5916448, 17.0283775, -32.8280487, 57.9800606
2: -16.3717403, 52.3268585, -5.1096640, 16.4983921, -32.8701324, 57.3101768
3: -24.1192474, 56.3102837, -7.1945591, 17.9123573, -42.0316048, 63.3753014
4: -26.2549019, 50.4941635, -8.5250015, 15.6656189, -41.9205208, 59.0191574

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1673386
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1679343
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -6.1992731, 21.2122402, -34.5469360, 52.1136818
1: -15.7996721, 53.5419350, -7.0349321, 24.6053638, -40.4050369, 60.4046135
2: -16.3717403, 52.3268585, -7.6394072, 23.9174843, -40.2892227, 59.8404884
3: -24.1192474, 56.3102837, -10.8425484, 25.8447590, -49.9640045, 67.0151062
4: -26.2549019, 50.4941635, -12.3402233, 22.9567204, -49.2116241, 62.8343849

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1673386
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591226, upper bound: 47.1679343
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -4.0656214, 14.6907520, -30.1852608, 56.6620331
1: -18.2942677, 61.3451691, -4.5916448, 17.0283775, -35.3226357, 65.7409744
2: -18.9236336, 59.9523926, -5.1096640, 16.4983921, -35.4220276, 64.8969040
3: -27.8473854, 64.4847412, -7.1945591, 17.9123573, -45.7597389, 71.5063324
4: -30.1493359, 57.9222450, -8.5250015, 15.6656189, -45.8149567, 66.4414139

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0670800, upper bound: 47.1377695
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0670800, upper bound: 47.1657228
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -6.1992731, 21.2122402, -36.7067490, 58.7828941
1: -18.2942677, 61.3451691, -7.0349321, 24.6053638, -42.8996277, 68.1655273
2: -18.9236336, 59.9523926, -7.6394072, 23.9174843, -42.8411179, 67.4272308
3: -27.8473854, 64.4847412, -10.8425484, 25.8447590, -53.6921387, 75.1461258
4: -30.1493359, 57.9222450, -12.3402233, 22.9567204, -53.1060562, 70.2392578

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0670800, upper bound: 47.1377695
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1586223, upper bound: 47.1657228
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.0809240, 42.0447922, -8.4888983, 29.9237366, -41.7976036, 50.3575935
1: -14.3679867, 48.8180733, -9.9612656, 34.8440018, -48.9255486, 58.5338631
2: -14.7996445, 47.7800980, -10.5535297, 33.9458389, -48.4923820, 58.1032486
3: -21.9019432, 51.2413292, -15.4596739, 36.7493782, -58.3407898, 66.4161301
4: -23.6488724, 46.0467529, -17.2898254, 32.6146545, -56.0824547, 63.1513100

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634560, upper bound: 47.1328574
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0632543, upper bound: 47.1333637
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.6461382, 43.8475800, -11.9649391, 41.5066605, -53.8647919, 55.6910629
1: -15.0353851, 50.8989334, -14.0966520, 48.3114662, -62.9374504, 64.7721939
2: -15.4672871, 49.8316040, -14.7553663, 47.1445961, -62.2542648, 64.3785019
3: -22.8897457, 53.4193306, -21.6540661, 50.8599434, -73.3150635, 74.7922592
4: -24.6493130, 48.0615349, -23.8034744, 45.4206161, -69.8067169, 71.6441803

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634198, upper bound: 47.1268516
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0632174, upper bound: 47.1274981
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.3680058, 42.9620438, -12.2086325, 42.2005005, -54.4199562, 55.0393219
1: -14.6970406, 49.9281464, -14.3693399, 49.0852661, -63.5349617, 64.0760651
2: -15.1878767, 48.8440247, -15.0035334, 47.9183235, -62.8933105, 63.6468010
3: -22.4419918, 52.4952850, -22.0292950, 51.6172180, -73.7837067, 74.2602539
4: -24.3620014, 47.1144180, -24.0937843, 46.1460915, -70.3372116, 71.0533676

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.3680058, 42.9620438, -12.0580769, 41.7441597, -53.9621429, 54.8893471
1: -14.6970406, 49.9281464, -14.2033033, 48.5890732, -63.0335655, 63.9128952
2: -15.1878767, 48.8440247, -14.8611126, 47.4124603, -62.3846474, 63.5062752
3: -22.4419918, 52.4952850, -21.8105984, 51.1855354, -73.3461761, 74.0473328
4: -24.3620014, 47.1144180, -23.9963131, 45.6823387, -69.8804092, 70.9571304

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0462485, upper bound: 47.1550777
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0948480, upper bound: 47.1568767
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.0604925, 41.9762154, -10.4762878, 36.1499863, -47.9616013, 52.2669182
1: -14.3424902, 48.7392769, -12.2804556, 42.1072693, -56.1037521, 60.7666740
2: -14.7751417, 47.7012482, -12.9199095, 41.0368958, -55.5034447, 60.3853416
3: -21.8643150, 51.1589661, -18.9337711, 44.3691521, -65.8659897, 69.7948685
4: -23.6122742, 45.9697227, -20.9103680, 39.5404396, -62.9315109, 66.6814194

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0634579, upper bound: 47.1352201
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0632563, upper bound: 47.1356621
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -12.6256218, 43.7787056, -14.0721521, 48.0491753, -60.3442802, 57.7123871
1: -15.0098400, 50.8197365, -16.5212135, 55.9232254, -70.4620209, 67.1147385
2: -15.4425802, 49.7524643, -17.2497845, 54.5756226, -69.6030350, 66.7855453
3: -22.8520451, 53.3366394, -25.2834396, 58.8358154, -81.1931992, 78.3291702
4: -24.6123676, 47.9842224, -27.6175613, 52.6529465, -76.9637146, 75.3716812

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0627448, upper bound: 47.1185312
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0625514, upper bound: 47.1193907
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -12.3462753, 42.8892822, -13.7523966, 47.0067673, -59.1741638, 56.5128746
1: -14.6699352, 49.8443909, -16.1948166, 54.6632195, -69.0421371, 65.8208923
2: -15.1618128, 48.7604256, -16.8338146, 53.4109421, -68.3211212, 65.4006805
3: -22.4018936, 52.4078178, -24.7555180, 57.5092468, -79.5913849, 76.8970261
4: -24.3229198, 47.0327644, -26.9120579, 51.5527725, -75.6850739, 73.7867203

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633490, upper bound: 47.1566890
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1612589, upper bound: 47.1583452
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -12.3462753, 42.8892822, -14.7070045, 50.0167313, -62.1772079, 57.4573669
1: -14.6699352, 49.8443909, -17.2721062, 58.2096634, -72.5779343, 66.8981094
2: -15.1618128, 48.7604256, -18.0042229, 56.8245049, -71.7242279, 66.5637512
3: -22.4018936, 52.4078178, -26.4022522, 61.2570343, -83.3281860, 78.5411987
4: -24.3229198, 47.0327644, -28.7846279, 54.8572006, -78.9845276, 75.6539764

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0633490, upper bound: 47.1566890
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0631240, upper bound: 47.1583452
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.2689610, 45.8275833, -12.5308905, 43.2453651, -56.3859329, 58.2394295
1: -15.7194643, 53.2859726, -14.7469082, 50.3030891, -65.8060760, 67.8283386
2: -16.2881584, 52.0751572, -15.3909693, 49.1077614, -65.2037277, 67.2816086
3: -23.9959946, 56.0326614, -22.5989418, 52.8967743, -76.6549911, 78.3945236
4: -26.1147709, 50.2447128, -24.7015343, 47.2985725, -73.2776566, 74.8124084

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0756881, upper bound: 47.0763334
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0756881, upper bound: 47.0890779
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -12.3790092, 42.7884178, -55.9955025, 58.3076630
1: -15.7996721, 53.5419350, -14.5795441, 49.8086128, -65.3926163, 67.9193420
2: -16.3717403, 52.3268585, -15.2485409, 48.6013947, -64.7792206, 67.3923264
3: -24.1192474, 56.3102837, -22.3771629, 52.4660530, -76.3426208, 78.4542160
4: -26.2549019, 50.4941635, -24.6049595, 46.8328056, -72.9564590, 74.9655609

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0890779, upper bound: 47.1389646
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0890779, upper bound: 47.1577168
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -15.4222975, 52.5052338, -12.5308905, 43.2453651, -58.5286293, 64.8858032
1: -18.2031841, 61.0632591, -14.7469082, 50.3030891, -68.2871552, 75.5636063
2: -18.8315601, 59.6730652, -15.3909693, 49.1077614, -67.7431107, 74.8409576
3: -27.7078972, 64.1794357, -22.5989418, 52.8967743, -80.3566437, 86.4981842
4: -29.9976788, 57.6430588, -24.7015343, 47.2985725, -77.1497726, 82.1850052

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0611593, upper bound: 47.0739853
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0611593, upper bound: 47.0844995
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -12.3790092, 42.7884178, -58.1451683, 64.9768600
1: -18.2942677, 61.3451691, -14.5795441, 49.8086128, -67.8848343, 75.6802444
2: -18.9236336, 59.9523926, -15.2485409, 48.6013947, -67.3272324, 74.9790726
3: -27.8473854, 64.4847412, -22.3771629, 52.4660530, -80.0606842, 86.5852356
4: -30.1493359, 57.9222450, -24.6049595, 46.8328056, -76.8402481, 82.3676987

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1591098, upper bound: 47.1539502
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1419862, upper bound: 47.1532321
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -14.5922365, 49.7267303, -62.9008331, 60.5121765
1: -15.7996721, 53.5419350, -17.1407185, 57.8730888, -73.4121475, 70.4785080
2: -16.3717403, 52.3268585, -17.8705082, 56.4956703, -72.6344223, 70.0156326
3: -24.1192474, 56.3102837, -26.2145882, 60.9071693, -84.7410660, 82.2804871
4: -26.2549019, 50.4941635, -28.6012669, 54.5252533, -80.6213531, 78.9508667

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1579448, upper bound: 47.1555566
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1573199, upper bound: 47.1555908
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13.3346949, 46.0455627, -14.6949282, 50.1020889, -63.2982903, 60.6338959
1: -15.7996721, 53.5419350, -17.2795525, 58.3180923, -73.8847656, 70.6207962
2: -16.3717403, 52.3268585, -17.9964409, 56.9254112, -73.0876846, 70.1458740
3: -24.1192474, 56.3102837, -26.4238262, 61.3736229, -85.2339478, 82.4944229
4: -26.2549019, 50.4941635, -28.8026962, 54.9368935, -81.0501862, 79.1618118

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1569729, upper bound: 47.1582459
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1563479, upper bound: 47.1582801
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.9019537, 50.8811989, -10.8763313, 37.4585953, -52.1529961, 61.5688286
1: -17.6015739, 59.1880608, -12.7556219, 43.6386833, -60.9528122, 71.6799393
2: -18.2234917, 57.8324242, -13.4084339, 42.5324059, -60.4932976, 70.9952774
3: -26.8242550, 62.2302742, -19.6549149, 45.9893227, -72.5020218, 81.5839920
4: -29.0983582, 55.8446999, -21.6875648, 40.9990387, -69.9155273, 77.3447418

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1444996, upper bound: 47.1453345
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0756598, upper bound: 47.1388253
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1282700, upper bound: 47.1576722
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -15.4945087, 52.7464104, -14.4793692, 49.3726082, -64.5674286, 67.0774231
1: -18.2942677, 61.3451691, -17.0046978, 57.4706764, -75.3417130, 78.1042862
2: -18.9236336, 59.9523926, -17.7444592, 56.0900230, -74.6318512, 77.4641418
3: -27.8473854, 64.4847412, -26.0158119, 60.4719162, -87.8689423, 90.1993484
4: -30.1493359, 57.9222450, -28.4011955, 54.1293335, -84.0098953, 86.0985031

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.89 + 417.20 = 420.09 seconds
