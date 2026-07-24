## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 1800 seconds
Split limit: 100
Threshold: 12.5603180091


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=94, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1254749, 3.6849804, -12.1254749, 3.6849804, -13.9000626, 13.9000664)
1: (-3.6725278, 7.4047136, -3.6725278, 7.4047136, -8.5087318, 8.5087318)
2: (-0.7476629, 13.4437408, -0.7476629, 13.4437408, -13.4614334, 13.4614372)
3: (-1.1331540, 11.3208055, -1.1331540, 11.3208055, -12.0319748, 12.0319729)
4: (-11.1167927, 5.4944596, -11.1167927, 5.4944596, -14.6887894, 14.6887856)
5: (1.8379555, 17.7576351, 1.8379555, 17.7576351, -15.9196796, 15.9196796)
6: (-39.9360962, -18.2054901, -39.9360962, -18.2054901, -15.2155037, 15.2154999)
7: (-3.5839579, 12.2702246, -3.5839579, 12.2702246, -13.6391602, 13.6391640)
8: (-6.7126245, 8.5791082, -6.7126245, 8.5791082, -12.1128159, 12.1128178)
9: (-4.8001633, 11.7221231, -4.8001633, 11.7221231, -13.0374336, 13.0374336)
10: (1.2905650, 25.7459221, 1.2905650, 25.7459221, -20.9535446, 20.9535446)
11: (-11.5140429, 4.2893047, -11.5140429, 4.2893047, -15.8033476, 15.8033476)
12: (-11.9207573, 9.8318863, -11.9207573, 9.8318863, -15.0364494, 15.0364494)
13: (-18.5813675, 6.7336874, -18.5813675, 6.7336874, -16.6293907, 16.6293945)
14: (4.9191580, 36.4232140, 4.9191580, 36.4232140, -26.7749405, 26.7749405)
15: (-8.7149639, 9.2961569, -8.7149639, 9.2961569, -18.0111198, 18.0111198)
16: (-16.7502575, 2.5494740, -16.7502575, 2.5494740, -14.8331146, 14.8331184)
17: (6.1821284, 30.6604042, 6.1821284, 30.6604042, -17.2550125, 17.2550125)
18: (-14.4006071, 5.1362762, -14.4006071, 5.1362762, -14.4317722, 14.4317741)
19: (-20.2854080, -4.3147249, -20.2854080, -4.3147249, -14.5550842, 14.5550919)
20: (-2.4302251, 11.2302246, -2.4302251, 11.2302246, -12.6365433, 12.6365433)
21: (-11.0869732, 3.2533813, -11.0869732, 3.2533813, -14.3403549, 14.3403549)
22: (-3.7107844, 13.1194658, -3.7107844, 13.1194658, -14.9717941, 14.9717941)
23: (-14.5878725, 0.3550858, -14.5878725, 0.3550858, -14.3402557, 14.3402557)
24: (-19.9429436, -5.1096311, -19.9429436, -5.1096311, -9.2769775, 9.2769775)
25: (-5.4693775, 10.8657360, -5.4693775, 10.8657360, -13.8234940, 13.8234940)
26: (-21.0305023, 1.2183309, -21.0305023, 1.2183309, -19.3724327, 19.3724365)
27: (-16.0133667, 2.1910443, -16.0133667, 2.1910443, -13.2311211, 13.2311211)
28: (-12.8032837, 4.6523304, -12.8032837, 4.6523304, -17.4556141, 17.4556141)
29: (-5.6058531, 11.8947334, -5.6058531, 11.8947334, -14.9777222, 14.9777222)
30: (-10.0615349, 6.2096872, -10.0615349, 6.2096872, -13.5620842, 13.5620842)
31: (-10.9839916, 6.9575529, -10.9839916, 6.9575529, -14.6624298, 14.6624260)
32: (-24.9283237, -4.5463600, -24.9283237, -4.5463600, -13.3205376, 13.3205414)
33: (-69.3171234, -40.0850372, -69.3171234, -40.0850372, -16.6724014, 16.6724014)
34: (-53.7657318, -30.8895817, -53.7657318, -30.8895817, -14.1694450, 14.1694450)
35: (-47.8248253, -26.0548592, -47.8248253, -26.0548592, -13.0219498, 13.0219536)
36: (-42.8267365, -19.2610664, -42.8267365, -19.2610664, -15.1241531, 15.1241570)
37: (-86.6817169, -55.5291786, -86.6817169, -55.5291786, -18.9252739, 18.9252701)
38: (-52.9550896, -24.3118515, -52.9550896, -24.3118515, -18.3915100, 18.3915100)
39: (-76.5640564, -44.6138229, -76.5640564, -44.6138229, -16.0968819, 16.0968819)
40: (-67.2565613, -43.5056458, -67.2565613, -43.5056458, -14.3520813, 14.3520813)
41: (-55.4326935, -32.9363213, -55.4326935, -32.9363213, -16.7036438, 16.7036476)
42: (-29.4732132, -9.8671646, -29.4732132, -9.8671646, -17.2716637, 17.2716675)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 21.78 = 24.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -12.5728910, upper bound: 12.5728909

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5724678, upper bound: 12.5558986
time: 7.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5725908, upper bound: 12.5725906
time: 7.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.05
Output dim: 14, lower bound: -12.5724678, upper bound: 12.5558986
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.05
Output dim: 14, lower bound: -12.5725908, upper bound: 12.5725906

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.1017818, 3.6611991, -12.1197681, 3.6712508, -13.8621559, 13.8703232
1: -3.6547575, 7.3940511, -3.6620624, 7.4011393, -8.4891586, 8.4909325
2: -0.7325034, 13.4366245, -0.7399806, 13.4395113, -13.4402390, 13.4432564
3: -1.1163485, 11.3100195, -1.1234565, 11.3155460, -12.0118713, 12.0187950
4: -11.0856485, 5.4692922, -11.0989685, 5.4896941, -14.6522293, 14.6472626
5: 1.8589754, 17.7522354, 1.8498549, 17.7549820, -15.8960066, 15.9023800
6: -39.8594513, -18.2476025, -39.8919449, -18.2059078, -15.1397629, 15.1285934
7: -3.5471683, 12.2531652, -3.5641108, 12.2674465, -13.6038513, 13.6076775
8: -6.7038574, 8.5692301, -6.7094212, 8.5737772, -12.0985031, 12.0996170
9: -4.7669401, 11.6840029, -4.7935429, 11.7011271, -12.9843025, 12.9941177
10: 1.3244152, 25.7353039, 1.3085909, 25.7406158, -20.9102707, 20.9096451
11: -11.4956112, 4.2867537, -11.5057850, 4.2874932, -15.7831039, 15.7925386
12: -11.9007454, 9.8139744, -11.9102669, 9.8294735, -15.0142097, 15.0075111
13: -18.5765648, 6.6995573, -18.5799103, 6.7163067, -16.6105042, 16.5964737
14: 4.9534321, 36.3851204, 4.9236336, 36.4020729, -26.7190094, 26.7324677
15: -8.6785145, 9.2261906, -8.7121706, 9.2562971, -17.9348106, 17.9383621
16: -16.7173405, 2.5491030, -16.7339058, 2.5480571, -14.7950592, 14.8170166
17: 6.2084293, 30.6335335, 6.1849432, 30.6461258, -17.2133446, 17.2254639
18: -14.3618526, 5.1195707, -14.3786507, 5.1308556, -14.3848076, 14.3907681
19: -20.2656155, -4.3310986, -20.2767677, -4.3243608, -14.5235710, 14.5280876
20: -2.4095845, 11.2180166, -2.4191089, 11.2270613, -12.6151199, 12.6163979
21: -11.0654163, 3.2477210, -11.0752153, 3.2523122, -14.3177280, 14.3229361
22: -3.6943929, 13.0758457, -3.7075269, 13.0947943, -14.9262466, 14.9237366
23: -14.5549335, 0.3106279, -14.5823956, 0.3291802, -14.2833862, 14.2903519
24: -19.9372997, -5.1216455, -19.9408035, -5.1158757, -9.2693291, 9.2615776
25: -5.4556541, 10.8353634, -5.4659843, 10.8495016, -13.7928963, 13.7886963
26: -21.0046768, 1.1619680, -21.0269852, 1.1857567, -19.3229980, 19.3139343
27: -15.9955969, 2.1778698, -16.0073490, 2.1837912, -13.2004051, 13.2127609
28: -12.7710333, 4.6087141, -12.7983513, 4.6269450, -17.3979778, 17.4070663
29: -5.5701151, 11.8446465, -5.6031952, 11.8661213, -14.9142456, 14.9248085
30: -10.0484409, 6.2025323, -10.0551987, 6.2047801, -13.5411797, 13.5456886
31: -10.9504271, 6.9559507, -10.9666729, 6.9567819, -14.6273308, 14.6429100
32: -24.8931160, -4.5641727, -24.9084740, -4.5472975, -13.2898331, 13.2848587
33: -69.2819290, -40.1192703, -69.2967834, -40.0912971, -16.6315269, 16.6192932
34: -53.7307205, -30.9128094, -53.7447586, -30.8933258, -14.1324272, 14.1208916
35: -47.8159103, -26.0664291, -47.8198471, -26.0577641, -13.0090141, 12.9983330
36: -42.8266373, -19.2753658, -42.8225403, -19.2677479, -15.1114769, 15.1042023
37: -86.6726685, -55.5478134, -86.6760254, -55.5393105, -18.9129791, 18.9049873
38: -52.9075508, -24.3456268, -52.9286156, -24.3149052, -18.3415375, 18.3308945
39: -76.5342484, -44.6367455, -76.5479584, -44.6179352, -16.0620308, 16.0539398
40: -67.2153320, -43.5132904, -67.2362518, -43.5061646, -14.3058510, 14.3168526
41: -55.4085159, -32.9441376, -55.4202232, -32.9377022, -16.6804199, 16.6798935
42: -29.4582462, -9.8698635, -29.4657574, -9.8683481, -17.2514153, 17.2555847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 887

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5706286, upper bound: 12.5219066
time: 6.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5714238, upper bound: 12.5548614
time: 14.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.1248283, 3.6839695, -12.1251097, 3.6843724, -13.8987846, 13.8952827
1: -3.6714399, 7.4042912, -3.6718955, 7.4044633, -8.4995384, 8.5069942
2: -0.7456257, 13.4433460, -0.7464219, 13.4434948, -13.4587250, 13.4607201
3: -1.1308843, 11.3202343, -1.1316938, 11.3204517, -12.0397644, 12.0264359
4: -11.1157112, 5.4937677, -11.1161671, 5.4940524, -14.6770325, 14.6873207
5: 1.8409958, 17.7571621, 1.8399315, 17.7573414, -15.9163456, 15.9172306
6: -39.9335022, -18.2056465, -39.9346085, -18.2055702, -15.1444550, 15.2137985
7: -3.5822625, 12.2698469, -3.5828798, 12.2700033, -13.6180267, 13.6364784
8: -6.7087135, 8.5787354, -6.7103753, 8.5788879, -12.1109390, 12.1097965
9: -4.7995515, 11.7205381, -4.7998047, 11.7212009, -13.0360031, 13.0114365
10: 1.2949228, 25.7452583, 1.2930946, 25.7455120, -20.9452362, 20.9605713
11: -11.5118647, 4.2889900, -11.5127563, 4.2891335, -15.8009987, 15.8017464
12: -11.9199543, 9.8311443, -11.9202929, 9.8314581, -15.0261383, 15.0352097
13: -18.5811005, 6.7288871, -18.5812092, 6.7309394, -16.6211586, 16.6256371
14: 4.9205532, 36.4218025, 4.9199829, 36.4223404, -26.7726440, 26.7466202
15: -8.7146034, 9.2935839, -8.7147598, 9.2946539, -18.0092583, 18.0083427
16: -16.7441368, 2.5490685, -16.7466125, 2.5492330, -14.8317413, 14.8292694
17: 6.1833868, 30.6595211, 6.1828785, 30.6598873, -17.2534027, 17.2524071
18: -14.3991117, 5.1356602, -14.3997059, 5.1359234, -14.4263821, 14.4302006
19: -20.2844944, -4.3187284, -20.2848701, -4.3170877, -14.5524368, 14.5485573
20: -2.4290915, 11.2298927, -2.4295418, 11.2300224, -12.6296387, 12.6346931
21: -11.0857487, 3.2530856, -11.0862389, 3.2532060, -14.3389549, 14.3393250
22: -3.7103286, 13.1171064, -3.7104969, 13.1181030, -14.9699631, 14.9575119
23: -14.5869579, 0.3534913, -14.5873108, 0.3541653, -14.3385696, 14.3155327
24: -19.9422855, -5.1117210, -19.9425564, -5.1108451, -9.2722740, 9.2794800
25: -5.4684858, 10.8635340, -5.4688425, 10.8641872, -13.8205338, 13.8119545
26: -21.0298195, 1.2155912, -21.0300865, 1.2164972, -19.3701477, 19.3325577
27: -16.0128708, 2.1896887, -16.0130692, 2.1902118, -13.2352676, 13.2282677
28: -12.8025074, 4.6507559, -12.8028088, 4.6514177, -17.4539261, 17.4535637
29: -5.6054258, 11.8929567, -5.6055794, 11.8937120, -14.9762497, 14.9559250
30: -10.0575771, 6.2092175, -10.0592594, 6.2094030, -13.5611267, 13.5593872
31: -10.9821529, 6.9574041, -10.9829082, 6.9574618, -14.6595497, 14.6585884
32: -24.9271355, -4.5468588, -24.9276352, -4.5466523, -13.2907639, 13.3168182
33: -69.3152161, -40.0859985, -69.3160172, -40.0855942, -16.6434402, 16.6704102
34: -53.7645683, -30.8907013, -53.7650681, -30.8902340, -14.1287613, 14.1619759
35: -47.8243065, -26.0556068, -47.8244781, -26.0552769, -13.0069809, 13.0151634
36: -42.8263817, -19.2700310, -42.8265152, -19.2661629, -15.1204948, 15.1150513
37: -86.6809082, -55.5361557, -86.6812592, -55.5332832, -18.9199219, 18.9324074
38: -52.9533539, -24.3130093, -52.9540215, -24.3125229, -18.3557281, 18.3892212
39: -76.5626831, -44.6148834, -76.5632706, -44.6144180, -16.0941315, 16.0963020
40: -67.2547455, -43.5058022, -67.2554092, -43.5057526, -14.3442841, 14.3509026
41: -55.4318657, -32.9370956, -55.4322090, -32.9367561, -16.6893539, 16.6982307
42: -29.4720421, -9.8673935, -29.4725113, -9.8672924, -17.2683525, 17.2668762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=94, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 887

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5707522, upper bound: 12.5375414
time: 14.41 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5715468, upper bound: 12.5715462
time: 15.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.97 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.97
Output dim: 14, lower bound: -12.5706286, upper bound: 12.5219066
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.97
Output dim: 14, lower bound: -12.5714238, upper bound: 12.5548614
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.97
Output dim: 14, lower bound: -12.5707522, upper bound: 12.5375414
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.97
Output dim: 14, lower bound: -12.5715468, upper bound: 12.5715462

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.1007586, 3.6558414, -12.1177988, 3.6608372, -13.8494835, 13.8623848
1: -3.6543782, 7.3890715, -3.6613111, 7.3914690, -8.4786911, 8.4850159
2: -0.7313762, 13.4284649, -0.7377838, 13.4236450, -13.4232979, 13.4330978
3: -1.1155777, 11.3029613, -1.1219474, 11.3017998, -11.9966507, 12.0100689
4: -11.0847082, 5.4633212, -11.0971756, 5.4781237, -14.6397934, 14.6395531
5: 1.8599935, 17.7445908, 1.8517919, 17.7401638, -15.8801708, 15.8927994
6: -39.8581543, -18.2536106, -39.8895035, -18.2176132, -15.1269455, 15.1187782
7: -3.5460868, 12.2421980, -3.5620089, 12.2461777, -13.5801773, 13.5941544
8: -6.7024965, 8.5631104, -6.7067699, 8.5618906, -12.0837326, 12.0902233
9: -4.7621765, 11.6821594, -4.7843356, 11.6975746, -12.9732094, 12.9789276
10: 1.3309317, 25.7338905, 1.3212471, 25.7379131, -20.8967667, 20.8925781
11: -11.4917736, 4.2859116, -11.4983540, 4.2858586, -15.7776318, 15.7842655
12: -11.8895378, 9.8126783, -11.8884754, 9.8269625, -15.0007973, 14.9836426
13: -18.5703735, 6.6985760, -18.5679054, 6.7143106, -16.5978317, 16.5853462
14: 4.9681034, 36.3847275, 4.9520807, 36.4012680, -26.6992188, 26.6988983
15: -8.6731644, 9.2241669, -8.7017441, 9.2524176, -17.9255829, 17.9259109
16: -16.7151833, 2.5403662, -16.7297077, 2.5311220, -14.7750626, 14.8032532
17: 6.2211599, 30.6326790, 6.2097735, 30.6444836, -17.1970634, 17.1971054
18: -14.3596849, 5.1139221, -14.3744764, 5.1199045, -14.3699036, 14.3790855
19: -20.2615013, -4.3312140, -20.2688160, -4.3245974, -14.5171242, 14.5168762
20: -2.4069703, 11.2170887, -2.4140718, 11.2252979, -12.6097183, 12.6094170
21: -11.0592766, 3.2472916, -11.0634403, 3.2514644, -14.3107414, 14.3107319
22: -3.6832595, 13.0748501, -3.6859460, 13.0929089, -14.9125748, 14.8999138
23: -14.5517578, 0.3102217, -14.5761766, 0.3283777, -14.2763863, 14.2790909
24: -19.9350700, -5.1221929, -19.9364815, -5.1169224, -9.2650986, 9.2543869
25: -5.4467726, 10.8349152, -5.4487448, 10.8486395, -13.7828064, 13.7705116
26: -20.9904861, 1.1608264, -20.9992771, 1.1835585, -19.3064079, 19.2847290
27: -15.9945374, 2.1730692, -16.0053062, 2.1744683, -13.1893730, 13.2024193
28: -12.7671404, 4.6080461, -12.7907658, 4.6256537, -17.3927937, 17.3988113
29: -5.5589356, 11.8439760, -5.5814552, 11.8647881, -14.9017334, 14.9024010
30: -10.0434284, 6.2017975, -10.0454187, 6.2033329, -13.5344391, 13.5348740
31: -10.9479008, 6.9538670, -10.9617939, 6.9527178, -14.6185951, 14.6321983
32: -24.8918457, -4.5687766, -24.9060440, -4.5562181, -13.2800598, 13.2769241
33: -69.2812805, -40.1232529, -69.2954483, -40.0990143, -16.6246643, 16.6138573
34: -53.7300758, -30.9176006, -53.7435150, -30.9026756, -14.1218758, 14.1116982
35: -47.8143387, -26.0681076, -47.8168106, -26.0609818, -13.0014267, 12.9877472
36: -42.8245239, -19.2766838, -42.8184204, -19.2701950, -15.1045380, 15.0950775
37: -86.6698761, -55.5508270, -86.6705933, -55.5451317, -18.9031487, 18.8909149
38: -52.9056358, -24.3481922, -52.9249649, -24.3198032, -18.3337593, 18.3222809
39: -76.5327530, -44.6382828, -76.5447617, -44.6210175, -16.0527267, 16.0412178
40: -67.2142944, -43.5251465, -67.2342453, -43.5292015, -14.2941132, 14.3086395
41: -55.4078140, -32.9519119, -55.4188614, -32.9527588, -16.6690750, 16.6696320
42: -29.4572792, -9.8717813, -29.4638824, -9.8720207, -17.2419662, 17.2441292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5556356, upper bound: 12.5213106
time: 18.14 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5701075, upper bound: 12.5213424
time: 9.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.1013680, 3.6594911, -12.1366901, 3.6697164, -13.8576851, 13.8885155
1: -3.6545968, 7.3925028, -3.6739528, 7.4003367, -8.4851227, 8.4966908
2: -0.7321740, 13.4337435, -0.7630535, 13.4366121, -13.4347458, 13.4639053
3: -1.1158904, 11.3084917, -1.1347768, 11.3148460, -12.0058746, 12.0339508
4: -11.0851955, 5.4671359, -11.1108847, 5.4886789, -14.6495819, 14.6570129
5: 1.8594379, 17.7495308, 1.8326588, 17.7521782, -15.8927402, 15.9168720
6: -39.8585434, -18.2559986, -39.8941765, -18.2172909, -15.1338844, 15.1134796
7: -3.5465283, 12.2495079, -3.5945017, 12.2641630, -13.5940857, 13.6425705
8: -6.7032619, 8.5669842, -6.7262053, 8.5714130, -12.0893211, 12.1150227
9: -4.7613125, 11.6832409, -4.7870636, 11.7052431, -12.9779472, 12.9861832
10: 1.3260155, 25.7349968, 1.3073220, 25.7462654, -20.9081802, 20.9042282
11: -11.4938030, 4.2862048, -11.5050774, 4.2874589, -15.7812614, 15.7912827
12: -11.8967686, 9.8135309, -11.9054165, 9.8577538, -15.0462914, 14.9996910
13: -18.5626831, 6.6987958, -18.5583992, 6.7252574, -16.5646324, 16.6090469
14: 4.9597979, 36.3848839, 4.9259148, 36.4146576, -26.7104187, 26.7249069
15: -8.6698847, 9.2249498, -8.6994629, 9.2567511, -17.9266357, 17.9244118
16: -16.7167854, 2.5479503, -16.7527466, 2.5480320, -14.7896271, 14.8142662
17: 6.2124515, 30.6332169, 6.1864967, 30.6652451, -17.2211761, 17.2211609
18: -14.3608370, 5.1171741, -14.3850050, 5.1290693, -14.3789940, 14.3858356
19: -20.2629757, -4.3312407, -20.2771111, -4.3255734, -14.5263443, 14.5276108
20: -2.4086034, 11.2166996, -2.4202905, 11.2270145, -12.6149559, 12.6158142
21: -11.0639629, 3.2475419, -11.0779152, 3.2653337, -14.3292961, 14.3254566
22: -3.6909952, 13.0752163, -3.7056377, 13.1257458, -14.9560852, 14.9154396
23: -14.5542278, 0.3101816, -14.5846615, 0.3291183, -14.2884140, 14.2828865
24: -19.9348202, -5.1220250, -19.9394341, -5.1143003, -9.2748222, 9.2609558
25: -5.4545937, 10.8349762, -5.4683666, 10.8736629, -13.8163223, 13.7840080
26: -21.0009956, 1.1615865, -21.0242195, 1.2273159, -19.3587952, 19.3021469
27: -15.9951744, 2.1757057, -16.0212975, 2.1813974, -13.2002792, 13.2014503
28: -12.7697601, 4.6083488, -12.7998857, 4.6281295, -17.3978901, 17.4082336
29: -5.5662351, 11.8442955, -5.5997744, 11.8997707, -14.9447060, 14.9175949
30: -10.0464106, 6.2020221, -10.0544147, 6.2192969, -13.5535889, 13.5431595
31: -10.9493818, 6.9545431, -10.9744244, 6.9544692, -14.6234436, 14.6389084
32: -24.8922329, -4.5684886, -24.9120140, -4.5518541, -13.2872925, 13.2782402
33: -69.2816391, -40.1206512, -69.3042297, -40.0896339, -16.6336174, 16.6165199
34: -53.7305069, -30.9133701, -53.7482529, -30.8906326, -14.1341248, 14.1119347
35: -47.8069878, -26.0671139, -47.8070374, -26.0636005, -13.0067596, 12.9896126
36: -42.8162155, -19.2759819, -42.8069878, -19.2690926, -15.1066666, 15.0960083
37: -86.6665421, -55.5493622, -86.6698456, -55.5413780, -18.9214172, 18.9025574
38: -52.9068604, -24.3465080, -52.9303780, -24.3147011, -18.3405838, 18.3292847
39: -76.5261917, -44.6376114, -76.5377579, -44.6197205, -16.0619965, 16.0460854
40: -67.2145233, -43.5149269, -67.2696228, -43.5075912, -14.3056679, 14.2954483
41: -55.4082489, -32.9459610, -55.4327583, -32.9372711, -16.6870995, 16.6597366
42: -29.4574909, -9.8802929, -29.4614029, -9.8836002, -17.2624092, 17.2274628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5556356, upper bound: 12.5213106
time: 8.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5709053, upper bound: 12.5542616
time: 11.11 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.1237946, 3.6785960, -12.1231461, 3.6739721, -13.8860779, 13.8873482
1: -3.6710610, 7.3993187, -3.6711597, 7.3947849, -8.4890556, 8.5010529
2: -0.7444839, 13.4351816, -0.7442364, 13.4276352, -13.4418030, 13.4505196
3: -1.1301105, 11.3131866, -1.1301976, 11.3067265, -12.0245705, 12.0177231
4: -11.1147680, 5.4877815, -11.1143875, 5.4824715, -14.6645889, 14.6796494
5: 1.8419876, 17.7495632, 1.8418751, 17.7425480, -15.9005604, 15.9076881
6: -39.9322395, -18.2116547, -39.9321213, -18.2172928, -15.1316566, 15.2039948
7: -3.5811963, 12.2588940, -3.5807912, 12.2487411, -13.5943680, 13.6229477
8: -6.7073808, 8.5726299, -6.7077389, 8.5670109, -12.0961609, 12.1003952
9: -4.7947950, 11.7187157, -4.7906017, 11.7176313, -13.0248985, 12.9962196
10: 1.3014503, 25.7438011, 1.3057494, 25.7427521, -20.9316940, 20.9435501
11: -11.5080032, 4.2881160, -11.5053196, 4.2874808, -15.7954845, 15.7934361
12: -11.9087524, 9.8298216, -11.8984909, 9.8289270, -15.0127373, 15.0113792
13: -18.5749359, 6.7278967, -18.5692348, 6.7289772, -16.6085052, 16.6145096
14: 4.9352007, 36.4213676, 4.9484663, 36.4215965, -26.7527771, 26.7130585
15: -8.7092571, 9.2915878, -8.7043390, 9.2907925, -18.0000496, 17.9959259
16: -16.7419510, 2.5403373, -16.7423401, 2.5322943, -14.8117828, 14.8154945
17: 6.1961489, 30.6586761, 6.2076960, 30.6582413, -17.2371063, 17.2240639
18: -14.3969736, 5.1300182, -14.3955441, 5.1249609, -14.4114914, 14.4185295
19: -20.2803707, -4.3188286, -20.2768688, -4.3173084, -14.5460281, 14.5373421
20: -2.4264717, 11.2289581, -2.4244976, 11.2282715, -12.6242180, 12.6276932
21: -11.0795851, 3.2526553, -11.0744896, 3.2523775, -14.3319626, 14.3271446
22: -3.6992049, 13.1160955, -3.6889188, 13.1161852, -14.9562607, 14.9336815
23: -14.5837517, 0.3530617, -14.5811024, 0.3533626, -14.3315353, 14.3042831
24: -19.9400597, -5.1122670, -19.9382591, -5.1118989, -9.2680626, 9.2723045
25: -5.4595976, 10.8630848, -5.4515762, 10.8633375, -13.8104248, 13.7937546
26: -21.0156326, 1.2144530, -21.0023956, 1.2143126, -19.3535728, 19.3033829
27: -16.0117989, 2.1849012, -16.0110073, 2.1809030, -13.2242203, 13.2179222
28: -12.7985821, 4.6500568, -12.7952528, 4.6500778, -17.4486599, 17.4453087
29: -5.5942135, 11.8923149, -5.5838404, 11.8924189, -14.9637718, 14.9335213
30: -10.0525389, 6.2084827, -10.0494938, 6.2079716, -13.5544052, 13.5485840
31: -10.9796333, 6.9553127, -10.9780092, 6.9533978, -14.6508179, 14.6478539
32: -24.9258995, -4.5514455, -24.9252129, -4.5556011, -13.2809944, 13.3088379
33: -69.3145523, -40.0899887, -69.3146591, -40.0932999, -16.6365623, 16.6649551
34: -53.7639542, -30.8955307, -53.7637787, -30.8996029, -14.1181793, 14.1527634
35: -47.8227234, -26.0572853, -47.8214722, -26.0585251, -12.9994354, 13.0045967
36: -42.8242569, -19.2712708, -42.8223724, -19.2686234, -15.1135864, 15.1059151
37: -86.6781311, -55.5391541, -86.6758118, -55.5391159, -18.9100990, 18.9183540
38: -52.9514389, -24.3155060, -52.9503937, -24.3174229, -18.3479385, 18.3806305
39: -76.5610733, -44.6164703, -76.5601349, -44.6175117, -16.0848312, 16.0835991
40: -67.2537079, -43.5176697, -67.2534180, -43.5287933, -14.3325462, 14.3426895
41: -55.4311371, -32.9448776, -55.4308357, -32.9518661, -16.6779823, 16.6879730
42: -29.4710617, -9.8693056, -29.4706421, -9.8709650, -17.2589378, 17.2554016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5557706, upper bound: 12.5370300
time: 7.05 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5702251, upper bound: 12.5370501
time: 15.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.1244087, 3.6822200, -12.1420412, 3.6828470, -13.8942986, 13.9134750
1: -3.6712875, 7.4027586, -3.6837780, 7.4036694, -8.4954872, 8.5127506
2: -0.7452819, 13.4404469, -0.7695004, 13.4405947, -13.4532394, 13.4813423
3: -1.1304086, 11.3187027, -1.1430260, 11.3197660, -12.0337791, 12.0416126
4: -11.1152496, 5.4916077, -11.1280718, 5.4930301, -14.6743240, 14.6970558
5: 1.8413954, 17.7544823, 1.8227692, 17.7545586, -15.9131632, 15.9317131
6: -39.9326134, -18.2140331, -39.9368286, -18.2169476, -15.1385956, 15.1987000
7: -3.5816479, 12.2662172, -3.6132894, 12.2667198, -13.6083069, 13.6713600
8: -6.7081327, 8.5764933, -6.7271733, 8.5765038, -12.1017570, 12.1252060
9: -4.7939262, 11.7197866, -4.7933092, 11.7253141, -13.0296478, 13.0034866
10: 1.2965531, 25.7449398, 1.2918324, 25.7511406, -20.9431610, 20.9551620
11: -11.5100279, 4.2884250, -11.5120239, 4.2890916, -15.7991199, 15.8004494
12: -11.9159956, 9.8307009, -11.9154530, 9.8597279, -15.0582237, 15.0274124
13: -18.5672340, 6.7281017, -18.5597343, 6.7398601, -16.5753059, 16.6382179
14: 4.9269094, 36.4215546, 4.9223022, 36.4349785, -26.7640610, 26.7390366
15: -8.7059736, 9.2923546, -8.7020655, 9.2950048, -18.0009785, 17.9944191
16: -16.7435417, 2.5479097, -16.7654343, 2.5491958, -14.8263168, 14.8264732
17: 6.1874700, 30.6592121, 6.1844325, 30.6789894, -17.2612038, 17.2481079
18: -14.3981113, 5.1332526, -14.4060764, 5.1341066, -14.4205799, 14.4252853
19: -20.2818699, -4.3188710, -20.2852402, -4.3183074, -14.5551720, 14.5480881
20: -2.4281094, 11.2285595, -2.4307060, 11.2299700, -12.6294479, 12.6341133
21: -11.0842648, 3.2529147, -11.0889435, 3.2662294, -14.3504944, 14.3418579
22: -3.7069349, 13.1164722, -3.7086439, 13.1490164, -14.9997482, 14.9491882
23: -14.5862322, 0.3530407, -14.5896082, 0.3541133, -14.3435593, 14.3080559
24: -19.9398079, -5.1120853, -19.9411907, -5.1092768, -9.2777748, 9.2788696
25: -5.4674373, 10.8631277, -5.4712496, 10.8883743, -13.8439484, 13.8072319
26: -21.0261345, 1.2151756, -21.0273285, 1.2580523, -19.4059296, 19.3207932
27: -16.0124359, 2.1874952, -16.0270157, 2.1878042, -13.2351341, 13.2169724
28: -12.8012247, 4.6503649, -12.8043566, 4.6526346, -17.4538593, 17.4547215
29: -5.6015172, 11.8926010, -5.6021428, 11.9273815, -15.0067291, 14.9487267
30: -10.0555506, 6.2086816, -10.0585012, 6.2239003, -13.5735588, 13.5568924
31: -10.9811306, 6.9559970, -10.9906635, 6.9551711, -14.6556702, 14.6545677
32: -24.9262638, -4.5511580, -24.9311676, -4.5512271, -13.2882309, 13.3101807
33: -69.3149414, -40.0874023, -69.3234329, -40.0839386, -16.6455498, 16.6676254
34: -53.7643356, -30.8912830, -53.7685661, -30.8875275, -14.1304474, 14.1530380
35: -47.8153610, -26.0562973, -47.8117104, -26.0611172, -13.0047569, 13.0064812
36: -42.8159409, -19.2705994, -42.8109283, -19.2675323, -15.1157036, 15.1068726
37: -86.6748047, -55.5376472, -86.6750565, -55.5353394, -18.9283638, 18.9299927
38: -52.9526367, -24.3139000, -52.9557991, -24.3123055, -18.3547859, 18.3875961
39: -76.5545807, -44.6157608, -76.5530701, -44.6161919, -16.0940742, 16.0884552
40: -67.2539291, -43.5074615, -67.2887878, -43.5071640, -14.3440971, 14.3295059
41: -55.4315987, -32.9389191, -55.4447174, -32.9363403, -16.6960068, 16.6780777
42: -29.4712601, -9.8777962, -29.4681854, -9.8825312, -17.2793350, 17.2387543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=93, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1729

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5565706, upper bound: 12.5710039
time: 6.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5710217, upper bound: 12.5710212
time: 6.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.49 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5556356, upper bound: 12.5213106
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5701075, upper bound: 12.5213424
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5556356, upper bound: 12.5213106
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5709053, upper bound: 12.5542616
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5557706, upper bound: 12.5370300
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5702251, upper bound: 12.5370501
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5565706, upper bound: 12.5710039
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.49
Output dim: 14, lower bound: -12.5710217, upper bound: 12.5710212

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.0999479, 3.6557434, -12.1173878, 3.6607852, -13.8426514, 13.8618546
1: -3.6535244, 7.3889503, -3.6609247, 7.3914218, -8.4781265, 8.4849167
2: -0.7307152, 13.4283733, -0.7374491, 13.4236145, -13.4205551, 13.4326706
3: -1.1149853, 11.3028402, -1.1216412, 11.3017292, -11.9934273, 12.0097542
4: -11.0845680, 5.4631505, -11.0971193, 5.4780407, -14.6373520, 14.6429329
5: 1.8604016, 17.7444878, 1.8520718, 17.7400932, -15.8796921, 15.8924160
6: -39.8580170, -18.2545376, -39.8894157, -18.2180977, -15.1263046, 15.1057243
7: -3.5449731, 12.2420931, -3.5613980, 12.2461424, -13.5794601, 13.5938225
8: -6.7016506, 8.5630417, -6.7063360, 8.5618496, -12.0765648, 12.0897255
9: -4.7611990, 11.6819897, -4.7838287, 11.6974745, -12.9646034, 12.9782257
10: 1.3325305, 25.7336845, 1.3221450, 25.7378101, -20.8848419, 20.8910522
11: -11.4915438, 4.2851782, -11.4982224, 4.2854719, -15.7770157, 15.7834005
12: -11.8894262, 9.8121777, -11.8883877, 9.8267078, -15.0004349, 14.9812393
13: -18.5703182, 6.6982756, -18.5678444, 6.7141590, -16.5955963, 16.5870361
14: 4.9699526, 36.3845749, 4.9530401, 36.4012146, -26.6880798, 26.6978989
15: -8.6724367, 9.2239399, -8.7013741, 9.2522869, -17.9247246, 17.9253139
16: -16.7138405, 2.5402632, -16.7289085, 2.5310876, -14.7639198, 14.8027878
17: 6.2223120, 30.6326141, 6.2104015, 30.6444283, -17.1827087, 17.1964340
18: -14.3595524, 5.1135616, -14.3744001, 5.1196647, -14.3695793, 14.3730354
19: -20.2612743, -4.3322287, -20.2686882, -4.3251214, -14.5161705, 14.5107841
20: -2.4068148, 11.2161465, -2.4139876, 11.2248116, -12.6090698, 12.6009178
21: -11.0590534, 3.2462993, -11.0633545, 3.2509475, -14.3100014, 14.3096542
22: -3.6831255, 13.0732574, -3.6858680, 13.0920229, -14.9115219, 14.8928947
23: -14.5515184, 0.3091955, -14.5760517, 0.3278384, -14.2784157, 14.2770195
24: -19.9349422, -5.1232533, -19.9364147, -5.1174951, -9.2647285, 9.2496300
25: -5.4465332, 10.8336096, -5.4486170, 10.8477974, -13.7823143, 13.7695007
26: -20.9903030, 1.1593375, -20.9992332, 1.1827374, -19.3054085, 19.2812576
27: -15.9944143, 2.1717861, -16.0052490, 2.1737738, -13.1881752, 13.2017670
28: -12.7669449, 4.6062145, -12.7906742, 4.6244721, -17.3914165, 17.3968887
29: -5.5587935, 11.8429356, -5.5814028, 11.8642645, -14.9011230, 14.9014397
30: -10.0432405, 6.2009025, -10.0453167, 6.2028785, -13.5341873, 13.5336914
31: -10.9474964, 6.9527426, -10.9615917, 6.9520712, -14.6177254, 14.6252251
32: -24.8917179, -4.5693178, -24.9059448, -4.5565081, -13.2796440, 13.2682953
33: -69.2810211, -40.1249237, -69.2953110, -40.0998688, -16.6236801, 16.5962410
34: -53.7299881, -30.9186420, -53.7434807, -30.9032402, -14.1212769, 14.0988312
35: -47.8142624, -26.0695248, -47.8167725, -26.0616951, -13.0006485, 12.9683685
36: -42.8244553, -19.2782249, -42.8183365, -19.2709808, -15.1037216, 15.0724297
37: -86.6696472, -55.5516777, -86.6705017, -55.5456085, -18.9026375, 18.8816757
38: -52.9054832, -24.3507500, -52.9249191, -24.3211594, -18.3322449, 18.2877655
39: -76.5324860, -44.6399727, -76.5446854, -44.6219025, -16.0517082, 16.0166855
40: -67.2140198, -43.5251999, -67.2341232, -43.5292130, -14.2937355, 14.3082237
41: -55.4076843, -32.9527588, -55.4187889, -32.9531975, -16.6685028, 16.6606865
42: -29.4570808, -9.8720112, -29.4637604, -9.8721733, -17.2486153, 17.2415085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682386, upper bound: 12.4825891
time: 20.75 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5693244, upper bound: 12.5205398
time: 10.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.1005802, 3.6593509, -12.1362686, 3.6696718, -13.8508606, 13.8879929
1: -3.6537619, 7.3924174, -3.6735542, 7.4002781, -8.4845581, 8.4966087
2: -0.7315204, 13.4336414, -0.7627200, 13.4365711, -13.4319916, 13.4634895
3: -1.1152962, 11.3083353, -1.1344643, 11.3147659, -12.0026627, 12.0336514
4: -11.0850697, 5.4669642, -11.1108141, 5.4885879, -14.6471252, 14.6604042
5: 1.8598237, 17.7494488, 1.8329530, 17.7521133, -15.8922901, 15.9164963
6: -39.8583984, -18.2569084, -39.8941116, -18.2177830, -15.1332550, 15.1004295
7: -3.5454276, 12.2494078, -3.5938838, 12.2640867, -13.5933685, 13.6422348
8: -6.7023993, 8.5669079, -6.7257633, 8.5713711, -12.0821342, 12.1145420
9: -4.7603045, 11.6830597, -4.7865496, 11.7051411, -12.9693336, 12.9854774
10: 1.3276606, 25.7348099, 1.3081975, 25.7461262, -20.8962631, 20.9027176
11: -11.4935598, 4.2854633, -11.5049610, 4.2870846, -15.7806444, 15.7904243
12: -11.8966265, 9.8130293, -11.9053249, 9.8575010, -15.0459290, 14.9972687
13: -18.5625992, 6.6984911, -18.5583401, 6.7250552, -16.5623856, 16.6107445
14: 4.9616442, 36.3847771, 4.9268789, 36.4146194, -26.6992798, 26.7238617
15: -8.6691399, 9.2246962, -8.6991005, 9.2566223, -17.9257622, 17.9237976
16: -16.7154503, 2.5478601, -16.7519798, 2.5480042, -14.7784767, 14.8137932
17: 6.2136064, 30.6331406, 6.1871257, 30.6652126, -17.2068214, 17.2204781
18: -14.3607111, 5.1168041, -14.3849459, 5.1288314, -14.3786736, 14.3797760
19: -20.2627468, -4.3322706, -20.2769985, -4.3260970, -14.5253601, 14.5215149
20: -2.4084435, 11.2157574, -2.4201953, 11.2265387, -12.6143074, 12.6073112
21: -11.0637207, 3.2465568, -11.0777969, 3.2648301, -14.3285503, 14.3243542
22: -3.6908736, 13.0736132, -3.7055869, 13.1248674, -14.9550476, 14.9084282
23: -14.5540161, 0.3091619, -14.5845432, 0.3285785, -14.2904167, 14.2807961
24: -19.9346428, -5.1230793, -19.9393520, -5.1148510, -9.2744560, 9.2561874
25: -5.4543514, 10.8336678, -5.4682674, 10.8728266, -13.8158417, 13.7829895
26: -21.0008602, 1.1600661, -21.0241261, 1.2265036, -19.3577690, 19.2986755
27: -15.9950485, 2.1744146, -16.0212212, 2.1807106, -13.1990929, 13.2008133
28: -12.7695618, 4.6065025, -12.7997990, 4.6269941, -17.3965569, 17.4063015
29: -5.5660896, 11.8432159, -5.5996871, 11.8992405, -14.9440994, 14.9165955
30: -10.0462341, 6.2011042, -10.0543327, 6.2188320, -13.5533409, 13.5419960
31: -10.9489803, 6.9534035, -10.9742270, 6.9538212, -14.6225739, 14.6319466
32: -24.8920803, -4.5690136, -24.9119453, -4.5521402, -13.2868881, 13.2696152
33: -69.2813797, -40.1223412, -69.3041306, -40.0904961, -16.6326065, 16.5989151
34: -53.7304230, -30.9144058, -53.7482224, -30.8911743, -14.1335144, 14.0990639
35: -47.8068581, -26.0685463, -47.8069801, -26.0642986, -13.0059662, 12.9702301
36: -42.8161316, -19.2775993, -42.8069572, -19.2698784, -15.1058350, 15.0733681
37: -86.6662750, -55.5502243, -86.6696701, -55.5418167, -18.9208832, 18.8933105
38: -52.9066467, -24.3491096, -52.9302902, -24.3160400, -18.3390808, 18.2947235
39: -76.5259933, -44.6393661, -76.5376511, -44.6206360, -16.0609589, 16.0215416
40: -67.2142715, -43.5149727, -67.2695236, -43.5076141, -14.3052788, 14.2950439
41: -55.4081345, -32.9467926, -55.4327011, -32.9377289, -16.6865158, 16.6507683
42: -29.4572849, -9.8805141, -29.4613113, -9.8837070, -17.2690315, 17.2248535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5690820, upper bound: 12.5171770
time: 17.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5701208, upper bound: 12.5534800
time: 6.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1230021, 3.6784854, -12.1227198, 3.6738973, -13.8792534, 13.8868294
1: -3.6700950, 7.3992071, -3.6706705, 7.3947377, -8.4881821, 8.5010262
2: -0.7433610, 13.4351025, -0.7436582, 13.4275799, -13.4388542, 13.4498634
3: -1.1291199, 11.3130283, -1.1297724, 11.3066664, -12.0211334, 12.0172977
4: -11.1146584, 5.4876075, -11.1143026, 5.4823785, -14.6619186, 14.6816254
5: 1.8430552, 17.7494392, 1.8423696, 17.7424774, -15.8994217, 15.9070702
6: -39.9320908, -18.2125816, -39.9320221, -18.2177582, -15.1310005, 15.1909409
7: -3.5797806, 12.2587986, -3.5800481, 12.2486887, -13.5934143, 13.6222954
8: -6.7065148, 8.5725441, -6.7073002, 8.5669708, -12.0889893, 12.0999088
9: -4.7938051, 11.7185402, -4.7900829, 11.7175407, -13.0162849, 12.9955063
10: 1.3037868, 25.7436142, 1.3069539, 25.7426491, -20.9198074, 20.9417114
11: -11.5077934, 4.2873878, -11.5051966, 4.2871070, -15.7949009, 15.7925844
12: -11.9086132, 9.8293610, -11.8984222, 9.8286924, -15.0123596, 15.0089836
13: -18.5748768, 6.7275686, -18.5691738, 6.7288027, -16.6064453, 16.6142731
14: 4.9370584, 36.4212456, 4.9494190, 36.4214973, -26.7416229, 26.7119675
15: -8.7085104, 9.2913628, -8.7039633, 9.2906513, -17.9991608, 17.9953270
16: -16.7403679, 2.5402837, -16.7415829, 2.5322461, -14.8008194, 14.8150177
17: 6.1973624, 30.6585751, 6.2083168, 30.6581688, -17.2226334, 17.2233505
18: -14.3968039, 5.1295357, -14.3954716, 5.1246877, -14.4111462, 14.4126129
19: -20.2801323, -4.3204036, -20.2767506, -4.3181033, -14.5448112, 14.5308952
20: -2.4263256, 11.2280006, -2.4244337, 11.2277927, -12.6235847, 12.6193199
21: -11.0793648, 3.2516599, -11.0743656, 3.2518559, -14.3312206, 14.3260250
22: -3.6990912, 13.1141357, -3.6888549, 13.1151524, -14.9550323, 14.9265633
23: -14.5835228, 0.3520103, -14.5809927, 0.3528376, -14.3324890, 14.3016777
24: -19.9399109, -5.1133423, -19.9381599, -5.1124411, -9.2676964, 9.2674789
25: -5.4593945, 10.8610516, -5.4514728, 10.8623428, -13.8098679, 13.7925034
26: -21.0154495, 1.2126031, -21.0023651, 1.2133400, -19.3523445, 19.2991028
27: -16.0116825, 2.1835871, -16.0109520, 2.1801753, -13.2229576, 13.2174721
28: -12.7984095, 4.6477857, -12.7951384, 4.6489239, -17.4473343, 17.4429245
29: -5.5940709, 11.8912735, -5.5837488, 11.8918886, -14.9631653, 14.9324112
30: -10.0523720, 6.2075582, -10.0493870, 6.2075000, -13.5539284, 13.5475540
31: -10.9792643, 6.9541788, -10.9777880, 6.9527578, -14.6497002, 14.6410408
32: -24.9257488, -4.5519867, -24.9251366, -4.5558691, -13.2805710, 13.3002281
33: -69.3143768, -40.0916367, -69.3145294, -40.0941315, -16.6355743, 16.6473579
34: -53.7638626, -30.8965416, -53.7637444, -30.9001427, -14.1175346, 14.1399078
35: -47.8226318, -26.0586586, -47.8213806, -26.0592384, -12.9986267, 12.9852028
36: -42.8241234, -19.2728500, -42.8223152, -19.2694874, -15.1128273, 15.0828247
37: -86.6779251, -55.5397682, -86.6756439, -55.5394135, -18.9094620, 18.9091034
38: -52.9512558, -24.3180733, -52.9502869, -24.3187580, -18.3464165, 18.3460999
39: -76.5609207, -44.6181717, -76.5599670, -44.6183929, -16.0838051, 16.0590363
40: -67.2534332, -43.5177231, -67.2532730, -43.5288239, -14.3321457, 14.3422794
41: -55.4310570, -32.9456940, -55.4307632, -32.9522705, -16.6773987, 16.6790352
42: -29.4708710, -9.8695545, -29.4705200, -9.8710737, -17.2634430, 17.2529030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683579, upper bound: 12.4982176
time: 8.30 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5694418, upper bound: 12.5362427
time: 76.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.0991354, 3.6697249, -12.1281586, 3.6814146, -13.8671341, 13.8867188
1: -3.6629944, 7.3957801, -3.6791668, 7.4016552, -8.4827385, 8.4972095
2: -0.7285329, 13.4304762, -0.7600752, 13.4392309, -13.4351540, 13.4628525
3: -1.1167990, 11.3071508, -1.1354311, 11.3177509, -12.0161781, 12.0208817
4: -11.1093655, 5.4869933, -11.1253138, 5.4907541, -14.6623764, 14.6822052
5: 1.8588099, 17.7450714, 1.8321376, 17.7530937, -15.8942833, 15.9129333
6: -39.9167480, -18.2428799, -39.9349747, -18.2335854, -15.1051521, 15.1664276
7: -3.5646968, 12.2607937, -3.6040759, 12.2654800, -13.5874481, 13.6542778
8: -6.6824141, 8.5612469, -6.7122841, 8.5751076, -12.0757675, 12.0959702
9: -4.7624950, 11.7045488, -4.7759838, 11.7230568, -12.9946632, 12.9700813
10: 1.3312078, 25.7257538, 1.3095980, 25.7482643, -20.9066620, 20.9258652
11: -11.5001764, 4.2827382, -11.5087194, 4.2861309, -15.7863073, 15.7914581
12: -11.9094744, 9.8174706, -11.9138918, 9.8525524, -15.0432167, 15.0114098
13: -18.5645695, 6.7160730, -18.5584984, 6.7340336, -16.5664368, 16.6215515
14: 4.9826508, 36.3964081, 4.9535894, 36.4324875, -26.7070923, 26.6838837
15: -8.6836205, 9.2727976, -8.6897926, 9.2904959, -17.9741173, 17.9625893
16: -16.7100239, 2.5315545, -16.7472267, 2.5475905, -14.7876816, 14.7894478
17: 6.2201438, 30.6460075, 6.2028055, 30.6774502, -17.2272224, 17.2172356
18: -14.3812256, 5.1198444, -14.4033985, 5.1270266, -14.3963146, 14.4083996
19: -20.2631035, -4.3411903, -20.2809830, -4.3315754, -14.5258484, 14.5228157
20: -2.4100506, 11.1999550, -2.4280427, 11.2134514, -12.5943642, 12.6021423
21: -11.0622559, 3.2217073, -11.0851822, 3.2481542, -14.3104095, 14.3068895
22: -3.6973715, 13.0964241, -3.7064610, 13.1378679, -14.9782753, 14.9253769
23: -14.5789404, 0.3466792, -14.5857391, 0.3510566, -14.3281784, 14.2956352
24: -19.9272041, -5.1382875, -19.9382744, -5.1244907, -9.2484322, 9.2494087
25: -5.4551711, 10.8431091, -5.4670362, 10.8771763, -13.8183403, 13.7791901
26: -21.0164070, 1.1916852, -21.0247879, 1.2451744, -19.3804169, 19.2929077
27: -16.0066242, 2.1778965, -16.0246582, 2.1824875, -13.2262039, 13.2040291
28: -12.7949705, 4.6373448, -12.8009872, 4.6457453, -17.4407158, 17.4383316
29: -5.5985155, 11.8825960, -5.6001396, 11.9222126, -14.9982452, 14.9362717
30: -10.0478306, 6.1994352, -10.0554705, 6.2195911, -13.5598221, 13.5447083
31: -10.9533062, 6.9351354, -10.9836798, 6.9430580, -14.6191330, 14.6303635
32: -24.9143696, -4.5673780, -24.9287529, -4.5604725, -13.2670784, 13.2907104
33: -69.2884445, -40.1371536, -69.3187485, -40.1120605, -16.5933609, 16.6145325
34: -53.7516251, -30.9236374, -53.7677841, -30.9056168, -14.0987244, 14.1170082
35: -47.7977524, -26.0999165, -47.8104439, -26.0857735, -12.9630814, 12.9613914
36: -42.7991295, -19.3197365, -42.8100624, -19.2955570, -15.0704803, 15.0551224
37: -86.6559372, -55.5653648, -86.6711884, -55.5509796, -18.8963699, 18.9001617
38: -52.9198151, -24.3936024, -52.9538116, -24.3580513, -18.2747955, 18.3036575
39: -76.5276337, -44.6661110, -76.5489960, -44.6449509, -16.0392876, 16.0344048
40: -67.2421722, -43.5094910, -67.2834167, -43.5081558, -14.3299713, 14.3228760
41: -55.4190598, -32.9644470, -55.4430161, -32.9507484, -16.6667862, 16.6473007
42: -29.4649296, -9.8824282, -29.4655762, -9.8847065, -17.2610168, 17.2289543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5547490, upper bound: 12.5335449
time: 6.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5557889, upper bound: 12.5702191
time: 55.30 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1236172, 3.6821358, -12.1416235, 3.6827779, -13.8874626, 13.9129562
1: -3.6703362, 7.4026442, -3.6833045, 7.4036050, -8.4946136, 8.5127239
2: -0.7441459, 13.4403629, -0.7689253, 13.4405613, -13.4502716, 13.4806938
3: -1.1294334, 11.3185654, -1.1426083, 11.3196850, -12.0303612, 12.0411835
4: -11.1151562, 5.4914522, -11.1279993, 5.4929214, -14.6716995, 14.6990585
5: 1.8424578, 17.7543812, 1.8232403, 17.7544956, -15.9120378, 15.9311409
6: -39.9324570, -18.2149620, -39.9367409, -18.2174225, -15.1379623, 15.1856270
7: -3.5802479, 12.2661152, -3.6125596, 12.2666388, -13.6073456, 13.6707306
8: -6.7072821, 8.5764160, -6.7267256, 8.5764780, -12.0945892, 12.1247139
9: -4.7929001, 11.7196369, -4.7927990, 11.7252016, -13.0210342, 13.0027771
10: 1.2988725, 25.7447681, 1.2930312, 25.7510109, -20.9312439, 20.9533539
11: -11.5097828, 4.2876954, -11.5118895, 4.2887063, -15.7984886, 15.7995853
12: -11.9158173, 9.8302364, -11.9153728, 9.8594933, -15.0578270, 15.0250053
13: -18.5671616, 6.7278080, -18.5596905, 6.7397451, -16.5732117, 16.6380119
14: 4.9287519, 36.4214096, 4.9232597, 36.4348793, -26.7528687, 26.7379913
15: -8.7052584, 9.2921333, -8.7016878, 9.2948771, -18.0001354, 17.9938202
16: -16.7419548, 2.5478382, -16.7646389, 2.5491614, -14.8153687, 14.8259621
17: 6.1886358, 30.6591110, 6.1850529, 30.6789474, -17.2467308, 17.2474174
18: -14.3979616, 5.1327724, -14.4059982, 5.1338720, -14.4202271, 14.4193497
19: -20.2816467, -4.3204374, -20.2850952, -4.3191013, -14.5540009, 14.5416412
20: -2.4279754, 11.2276258, -2.4306324, 11.2294846, -12.6288033, 12.6257286
21: -11.0840797, 3.2519274, -11.0888386, 3.2657175, -14.3497972, 14.3407660
22: -3.7068412, 13.1144962, -3.7085633, 13.1479378, -14.9985199, 14.9420738
23: -14.5860176, 0.3519988, -14.5894814, 0.3535860, -14.3444901, 14.3054695
24: -19.9396610, -5.1131663, -19.9411049, -5.1098423, -9.2774200, 9.2740631
25: -5.4672165, 10.8611012, -5.4711223, 10.8873558, -13.8433723, 13.8059769
26: -21.0259876, 1.2133424, -21.0272522, 1.2571101, -19.4047089, 19.3165588
27: -16.0123062, 2.1861968, -16.0269127, 2.1871066, -13.2338791, 13.2165070
28: -12.8009968, 4.6480737, -12.8042564, 4.6514864, -17.4524841, 17.4523296
29: -5.6013870, 11.8915539, -5.6020975, 11.9268589, -15.0061264, 14.9475861
30: -10.0553865, 6.2077756, -10.0583773, 6.2234678, -13.5730896, 13.5558739
31: -10.9807444, 6.9548321, -10.9904385, 6.9545164, -14.6545486, 14.6477509
32: -24.9261246, -4.5516834, -24.9311295, -4.5515327, -13.2878036, 13.3015556
33: -69.3147049, -40.0890350, -69.3233185, -40.0847397, -16.6445465, 16.6500282
34: -53.7642670, -30.8923073, -53.7685165, -30.8880730, -14.1297798, 14.1401978
35: -47.8152313, -26.0577030, -47.8116455, -26.0618305, -13.0039406, 12.9870415
36: -42.8158417, -19.2721615, -42.8108749, -19.2683372, -15.1149521, 15.0837898
37: -86.6745300, -55.5383301, -86.6748962, -55.5356216, -18.9277153, 18.9207649
38: -52.9523926, -24.3164215, -52.9557610, -24.3135643, -18.3532181, 18.3531036
39: -76.5543671, -44.6174927, -76.5529785, -44.6170883, -16.0930672, 16.0639191
40: -67.2536163, -43.5074997, -67.2886658, -43.5072021, -14.3436928, 14.3290863
41: -55.4314651, -32.9397583, -55.4446487, -32.9367638, -16.6954308, 16.6691399
42: -29.4710922, -9.8780518, -29.4680843, -9.8826571, -17.2838783, 17.2362747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=93, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5692003, upper bound: 12.5335621
time: 12.97 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5702374, upper bound: 12.5702366
time: 9.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.65 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5682386, upper bound: 12.4825891
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5693244, upper bound: 12.5205398
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5690820, upper bound: 12.5171770
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5701208, upper bound: 12.5534800
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5683579, upper bound: 12.4982176
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5694418, upper bound: 12.5362427
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5547490, upper bound: 12.5335449
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5557889, upper bound: 12.5702191
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5692003, upper bound: 12.5335621
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.65
Output dim: 14, lower bound: -12.5702374, upper bound: 12.5702366

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.0977669, 3.6549938, -12.1137829, 3.6609588, -13.8404655, 13.8574600
1: -3.6483727, 7.3876982, -3.6538043, 7.3886404, -8.4707298, 8.4767742
2: -0.7299025, 13.4261198, -0.7363734, 13.4196987, -13.4143448, 13.4292145
3: -1.1141425, 11.2969780, -1.1192820, 11.2917433, -11.9826622, 12.0005989
4: -11.0822029, 5.4609880, -11.0930710, 5.4760895, -14.6278534, 14.6341362
5: 1.8611612, 17.7400627, 1.8512297, 17.7327328, -15.8715715, 15.8888330
6: -39.8561478, -18.2673244, -39.8756104, -18.2396278, -15.1027641, 15.0784035
7: -3.5436904, 12.2397261, -3.5588336, 12.2419548, -13.5737839, 13.5852127
8: -6.6992006, 8.5618000, -6.7029376, 8.5595589, -12.0663414, 12.0841389
9: -4.7506166, 11.6813326, -4.7669482, 11.6939192, -12.9525146, 12.9607430
10: 1.3504267, 25.7325668, 1.3525734, 25.7268753, -20.8556976, 20.8588333
11: -11.4862585, 4.2850080, -11.4900265, 4.2855463, -15.7718048, 15.7750340
12: -11.8812046, 9.8112316, -11.8748817, 9.8217201, -14.9859161, 14.9661407
13: -18.5611572, 6.6960258, -18.5534172, 6.7081442, -16.5713768, 16.5652962
14: 4.9928732, 36.3834915, 4.9924164, 36.3854485, -26.6458893, 26.6565475
15: -8.6670456, 9.2210274, -8.6920671, 9.2456551, -17.9127007, 17.9130936
16: -16.6997738, 2.5395932, -16.7085667, 2.5332146, -14.7602539, 14.7836952
17: 6.2336783, 30.6317196, 6.2303448, 30.6354294, -17.1730461, 17.1767578
18: -14.3580246, 5.1104431, -14.3728905, 5.1144581, -14.3621979, 14.3687210
19: -20.2582741, -4.3364468, -20.2632561, -4.3311672, -14.5065155, 14.4998360
20: -2.4047618, 11.2067661, -2.4063923, 11.2102833, -12.5933418, 12.5868759
21: -11.0557766, 3.2443366, -11.0581293, 3.2484937, -14.3042698, 14.3024654
22: -3.6819150, 13.0680323, -3.6840148, 13.0857439, -14.9026070, 14.8841629
23: -14.5486517, 0.3038101, -14.5728865, 0.3198943, -14.2666283, 14.2664337
24: -19.9328632, -5.1240559, -19.9329510, -5.1183977, -9.2613716, 9.2450714
25: -5.4419804, 10.8328342, -5.4407625, 10.8448257, -13.7760048, 13.7601318
26: -20.9867592, 1.1573131, -20.9929428, 1.1789310, -19.2968330, 19.2707672
27: -15.9929857, 2.1619167, -15.9947615, 2.1575925, -13.1709518, 13.1797867
28: -12.7646999, 4.5989375, -12.7830343, 4.6122932, -17.3769932, 17.3819714
29: -5.5573025, 11.8407373, -5.5785613, 11.8611641, -14.8957100, 14.8917580
30: -10.0380402, 6.2001505, -10.0370998, 6.2005768, -13.5298805, 13.5256653
31: -10.9440422, 6.9482336, -10.9559574, 6.9462028, -14.6082726, 14.6160774
32: -24.8902206, -4.5832081, -24.8902740, -4.5795259, -13.2563210, 13.2402840
33: -69.2795715, -40.1351471, -69.2873077, -40.1166687, -16.6121597, 16.5902939
34: -53.7289848, -30.9317055, -53.7322731, -30.9252090, -14.0967407, 14.0733414
35: -47.8135071, -26.0762062, -47.8121643, -26.0717735, -12.9901047, 12.9568863
36: -42.8235970, -19.2902508, -42.8103104, -19.2906113, -15.0841446, 15.0528717
37: -86.6679840, -55.5564575, -86.6658401, -55.5534592, -18.8930664, 18.8762283
38: -52.9045029, -24.3663807, -52.9101372, -24.3466434, -18.3061523, 18.2562866
39: -76.5305634, -44.6437454, -76.5414200, -44.6278954, -16.0423355, 16.0098724
40: -67.2121887, -43.5340424, -67.2248688, -43.5440407, -14.2758980, 14.2833424
41: -55.4060898, -32.9657135, -55.4051552, -32.9749908, -16.6432114, 16.6323166
42: -29.4553185, -9.8830833, -29.4516621, -9.8908319, -17.2280502, 17.2195549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5501213, upper bound: 12.4808119
time: 13.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671078, upper bound: 12.4817813
time: 8.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.0992718, 3.6554825, -12.1162148, 3.6603768, -13.8409691, 13.8607216
1: -3.6524181, 7.3887510, -3.6590614, 7.3910704, -8.4766159, 8.4815311
2: -0.7306028, 13.4275532, -0.7372525, 13.4221802, -13.4210320, 13.4312859
3: -1.1148518, 11.3021431, -1.1214098, 11.3005333, -11.9894714, 12.0087948
4: -11.0833788, 5.4626055, -11.0950327, 5.4770999, -14.6374359, 14.6413879
5: 1.8605247, 17.7436123, 1.8523059, 17.7385559, -15.8780308, 15.8913059
6: -39.8569527, -18.2548161, -39.8875427, -18.2185993, -15.0981293, 15.1027641
7: -3.5446558, 12.2418118, -3.5608792, 12.2456322, -13.5719299, 13.5904541
8: -6.7011065, 8.5627193, -6.7054644, 8.5612946, -12.0792465, 12.0869370
9: -4.7594538, 11.6818142, -4.7807450, 11.6971340, -12.9630508, 12.9670334
10: 1.3343716, 25.7333088, 1.3252687, 25.7371368, -20.8822327, 20.8753204
11: -11.4906635, 4.2851372, -11.4967499, 4.2853956, -15.7760592, 15.7818871
12: -11.8883591, 9.8119478, -11.8865185, 9.8263645, -14.9990387, 14.9716339
13: -18.5678272, 6.6977978, -18.5634727, 6.7133789, -16.5947075, 16.5609741
14: 4.9719896, 36.3842545, 4.9566708, 36.4006882, -26.6851425, 26.6715851
15: -8.6721907, 9.2233210, -8.7009621, 9.2512980, -17.9234886, 17.9242821
16: -16.7119484, 2.5401611, -16.7256756, 2.5308633, -14.7606659, 14.8006287
17: 6.2235866, 30.6323318, 6.2125721, 30.6438961, -17.1806488, 17.1877823
18: -14.3593302, 5.1127090, -14.3739738, 5.1181622, -14.3710384, 14.3708725
19: -20.2608337, -4.3330317, -20.2679787, -4.3265305, -14.5139084, 14.5093613
20: -2.4065022, 11.2155724, -2.4134548, 11.2238770, -12.6021805, 12.5997696
21: -11.0585499, 3.2455318, -11.0625114, 3.2496090, -14.3081589, 14.3080435
22: -3.6829405, 13.0722532, -3.6855068, 13.0903015, -14.9087601, 14.8915710
23: -14.5511322, 0.3076348, -14.5753889, 0.3252072, -14.2751656, 14.2754974
24: -19.9342270, -5.1234002, -19.9352283, -5.1177416, -9.2641106, 9.2485580
25: -5.4460025, 10.8334112, -5.4477181, 10.8474617, -13.7791138, 13.7651825
26: -20.9897461, 1.1588595, -20.9982281, 1.1819785, -19.3026848, 19.2782135
27: -15.9941425, 2.1708841, -16.0047779, 2.1721513, -13.1645737, 13.2002182
28: -12.7664843, 4.6053343, -12.7898941, 4.6229882, -17.3894730, 17.3952293
29: -5.5583668, 11.8421917, -5.5806546, 11.8630371, -14.8985214, 14.9028625
30: -10.0420971, 6.2007437, -10.0433693, 6.2026520, -13.5323143, 13.5309448
31: -10.9468651, 6.9513683, -10.9604912, 6.9496374, -14.6141930, 14.6228676
32: -24.8912277, -4.5697584, -24.9051743, -4.5572367, -13.2570877, 13.2666550
33: -69.2805481, -40.1259193, -69.2945480, -40.1015930, -16.6189728, 16.5921402
34: -53.7295036, -30.9198761, -53.7426300, -30.9051113, -14.1015167, 14.0968628
35: -47.8139191, -26.0702591, -47.8162384, -26.0629349, -12.9987564, 12.9676285
36: -42.8237572, -19.2797050, -42.8171654, -19.2734070, -15.0915108, 15.0699120
37: -86.6690063, -55.5521240, -86.6694107, -55.5462837, -18.9033585, 18.8790703
38: -52.9049873, -24.3517990, -52.9239960, -24.3229465, -18.3066177, 18.2858505
39: -76.5317764, -44.6403809, -76.5435638, -44.6225204, -16.0533371, 16.0141869
40: -67.2130508, -43.5257187, -67.2324982, -43.5301666, -14.2679596, 14.3022614
41: -55.4069977, -32.9534760, -55.4176254, -32.9542923, -16.6425095, 16.6589394
42: -29.4562778, -9.8722849, -29.4623680, -9.8726578, -17.2281952, 17.2398834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5513718, upper bound: 12.5187837
time: 6.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681960, upper bound: 12.5196318
time: 28.33 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.0983791, 3.6586094, -12.1326723, 3.6698613, -13.8486748, 13.8835983
1: -3.6486108, 7.3911748, -3.6664205, 7.3975182, -8.4771652, 8.4884777
2: -0.7306873, 13.4313745, -0.7616494, 13.4326324, -13.4257698, 13.4600563
3: -1.1144649, 11.3025112, -1.1321145, 11.3047867, -11.9918785, 12.0244980
4: -11.0826864, 5.4648085, -11.1067657, 5.4866314, -14.6376343, 14.6515846
5: 1.8606014, 17.7450066, 1.8320718, 17.7447567, -15.8841553, 15.9129353
6: -39.8565216, -18.2696953, -39.8803024, -18.2392921, -15.1097221, 15.0731010
7: -3.5441508, 12.2470322, -3.5913348, 12.2599306, -13.5876923, 13.6336517
8: -6.6999578, 8.5656776, -6.7223606, 8.5690556, -12.0719299, 12.1089573
9: -4.7497120, 11.6824093, -4.7696829, 11.7015638, -12.9572334, 12.9679832
10: 1.3455491, 25.7336674, 1.3386488, 25.7352524, -20.8671341, 20.8704834
11: -11.4882736, 4.2852993, -11.4967089, 4.2871590, -15.7754326, 15.7820082
12: -11.8884411, 9.8120804, -11.8917866, 9.8525009, -15.0313835, 14.9821777
13: -18.5534821, 6.6962404, -18.5439339, 6.7190905, -16.5381927, 16.5890045
14: 4.9845772, 36.3836746, 4.9662571, 36.3988152, -26.6571274, 26.6825409
15: -8.6637859, 9.2217932, -8.6897745, 9.2500238, -17.9138107, 17.9115677
16: -16.7013855, 2.5471792, -16.7316418, 2.5501399, -14.7748108, 14.7947540
17: 6.2249422, 30.6322517, 6.2070718, 30.6561928, -17.1971779, 17.2008057
18: -14.3591433, 5.1136775, -14.3834496, 5.1236191, -14.3712883, 14.3754768
19: -20.2597427, -4.3364730, -20.2715378, -4.3321657, -14.5157166, 14.5105667
20: -2.4063814, 11.2063961, -2.4125836, 11.2119923, -12.5985680, 12.5932617
21: -11.0604496, 3.2445979, -11.0725670, 3.2623591, -14.3228092, 14.3171654
22: -3.6896431, 13.0683775, -3.7037463, 13.1186008, -14.9461517, 14.8996811
23: -14.5511665, 0.3037758, -14.5813684, 0.3206239, -14.2786713, 14.2702179
24: -19.9326401, -5.1238713, -19.9358788, -5.1157608, -9.2710915, 9.2516518
25: -5.4498034, 10.8329010, -5.4604082, 10.8698559, -13.8095245, 13.7736092
26: -20.9973335, 1.1580369, -21.0178337, 1.2226813, -19.3491974, 19.2881851
27: -15.9936314, 2.1645422, -16.0107574, 2.1644864, -13.1818619, 13.1788139
28: -12.7673426, 4.5992260, -12.7921209, 4.6148248, -17.3821678, 17.3913460
29: -5.5645800, 11.8410263, -5.5968666, 11.8961496, -14.9386635, 14.9069481
30: -10.0410252, 6.2003899, -10.0460854, 6.2164879, -13.5490189, 13.5339699
31: -10.9455338, 6.9489098, -10.9685917, 6.9479771, -14.6131210, 14.6227837
32: -24.8906021, -4.5829239, -24.8962841, -4.5751638, -13.2635498, 13.2415962
33: -69.2799225, -40.1325989, -69.2961273, -40.1072769, -16.6211433, 16.5929794
34: -53.7294540, -30.9274673, -53.7370110, -30.9131737, -14.1089935, 14.0735779
35: -47.8061447, -26.0752373, -47.8024063, -26.0743866, -12.9954224, 12.9587440
36: -42.8153114, -19.2895660, -42.7989044, -19.2894897, -15.0862541, 15.0537758
37: -86.6646271, -55.5549698, -86.6650696, -55.5496140, -18.9113426, 18.8878784
38: -52.9056664, -24.3647327, -52.9155502, -24.3415108, -18.3129578, 18.2632675
39: -76.5240860, -44.6431007, -76.5343399, -44.6265869, -16.0516129, 16.0147362
40: -67.2124176, -43.5237923, -67.2602768, -43.5224495, -14.2874451, 14.2701454
41: -55.4065475, -32.9597664, -55.4190445, -32.9595413, -16.6612282, 16.6224136
42: -29.4555168, -9.8915539, -29.4492149, -9.9023981, -17.2484856, 17.2029228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5510456, upper bound: 12.5150135
time: 13.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5679536, upper bound: 12.5160780
time: 7.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.0998812, 3.6591268, -12.1350908, 3.6692753, -13.8491707, 13.8868408
1: -3.6526515, 7.3922071, -3.6716862, 7.3999410, -8.4830589, 8.4932194
2: -0.7313927, 13.4328184, -0.7625140, 13.4351425, -13.4324722, 13.4621124
3: -1.1151682, 11.3076591, -1.1342332, 11.3135624, -11.9986954, 12.0326958
4: -11.0838680, 5.4664092, -11.1087284, 5.4876537, -14.6471863, 14.6588593
5: 1.8599544, 17.7485428, 1.8331718, 17.7505684, -15.8906136, 15.9153709
6: -39.8572998, -18.2571812, -39.8922272, -18.2182636, -15.1050720, 15.0974731
7: -3.5450807, 12.2491188, -3.5933623, 12.2635803, -13.5858307, 13.6388969
8: -6.7018690, 8.5665627, -6.7248807, 8.5707951, -12.0848351, 12.1117268
9: -4.7585177, 11.6828747, -4.7834659, 11.7048025, -12.9677811, 12.9742737
10: 1.3294744, 25.7344475, 1.3113585, 25.7454987, -20.8936615, 20.8870010
11: -11.4926720, 4.2854161, -11.5034351, 4.2870193, -15.7796917, 15.7888508
12: -11.8955717, 9.8128242, -11.9034729, 9.8571444, -15.0445099, 14.9876709
13: -18.5601368, 6.6980100, -18.5539646, 6.7242651, -16.5615234, 16.5846939
14: 4.9636784, 36.3844223, 4.9304686, 36.4140511, -26.6964111, 26.6976089
15: -8.6689177, 9.2240677, -8.6986904, 9.2556372, -17.9245548, 17.9227581
16: -16.7135544, 2.5477273, -16.7487278, 2.5477688, -14.7752075, 14.8116302
17: 6.2148938, 30.6328392, 6.1893206, 30.6646881, -17.2047615, 17.2118340
18: -14.3604479, 5.1159573, -14.3845253, 5.1273284, -14.3801346, 14.3776360
19: -20.2623196, -4.3330812, -20.2762699, -4.3274980, -14.5230865, 14.5200920
20: -2.4081326, 11.2151814, -2.4196532, 11.2255936, -12.6073990, 12.6061630
21: -11.0632324, 3.2457852, -11.0769806, 3.2634840, -14.3267164, 14.3227654
22: -3.6906476, 13.0726089, -3.7052217, 13.1231613, -14.9523010, 14.9071236
23: -14.5536146, 0.3076091, -14.5838766, 0.3259671, -14.2871780, 14.2792816
24: -19.9339657, -5.1232204, -19.9381542, -5.1151109, -9.2738342, 9.2551422
25: -5.4538441, 10.8334446, -5.4673595, 10.8724861, -13.8126221, 13.7786636
26: -21.0002632, 1.1595821, -21.0231171, 1.2257481, -19.3550262, 19.2956390
27: -15.9947863, 2.1734967, -16.0207443, 2.1790705, -13.1754951, 13.1992607
28: -12.7690926, 4.6056204, -12.7989845, 4.6255088, -17.3946018, 17.4046059
29: -5.5656748, 11.8424625, -5.5989532, 11.8979836, -14.9414520, 14.9180336
30: -10.0450840, 6.2009668, -10.0523539, 6.2185664, -13.5514793, 13.5392532
31: -10.9483404, 6.9520359, -10.9731236, 6.9514112, -14.6190453, 14.6295815
32: -24.8916130, -4.5694547, -24.9111671, -4.5528598, -13.2642937, 13.2679634
33: -69.2808990, -40.1233559, -69.3033600, -40.0922546, -16.6278992, 16.5948105
34: -53.7299309, -30.9156418, -53.7473602, -30.8930645, -14.1137848, 14.0971336
35: -47.8065414, -26.0692978, -47.8064423, -26.0655651, -13.0040665, 12.9694672
36: -42.8154602, -19.2789879, -42.8057671, -19.2723007, -15.0936317, 15.0708618
37: -86.6656418, -55.5506363, -86.6686172, -55.5425148, -18.9215660, 18.8906975
38: -52.9061966, -24.3501434, -52.9293785, -24.3177910, -18.3134346, 18.2928085
39: -76.5252686, -44.6397095, -76.5365067, -44.6212845, -16.0626030, 16.0190201
40: -67.2132721, -43.5154877, -67.2678680, -43.5085220, -14.2795067, 14.2890739
41: -55.4074936, -32.9475746, -55.4315300, -32.9388161, -16.6605072, 16.6490097
42: -29.4564819, -9.8808041, -29.4599056, -9.8842144, -17.2486420, 17.2232399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5522080, upper bound: 12.5513824
time: 9.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5689958, upper bound: 12.5523337
time: 12.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.1208038, 3.6777358, -12.1191292, 3.6741071, -13.8770676, 13.8824310
1: -3.6649594, 7.3979626, -3.6635396, 7.3919802, -8.4807777, 8.4929008
2: -0.7425451, 13.4328423, -0.7425787, 13.4236698, -13.4326401, 13.4464302
3: -1.1282762, 11.3071871, -1.1274126, 11.2966928, -12.0103607, 12.0081406
4: -11.1122723, 5.4854765, -11.1102562, 5.4804153, -14.6524200, 14.6727791
5: 1.8438025, 17.7450256, 1.8414974, 17.7351074, -15.8913050, 15.9035282
6: -39.9302063, -18.2253933, -39.9182396, -18.2392731, -15.1074524, 15.1635666
7: -3.5784950, 12.2564173, -3.5774992, 12.2445135, -13.5877304, 13.6137009
8: -6.7040615, 8.5712929, -6.7038927, 8.5646687, -12.0787888, 12.0943241
9: -4.7832527, 11.7178888, -4.7732038, 11.7139740, -13.0041580, 12.9780083
10: 1.3216915, 25.7424583, 1.3374076, 25.7317276, -20.8906784, 20.9094849
11: -11.5025177, 4.2872324, -11.4969711, 4.2871447, -15.7896624, 15.7842035
12: -11.9004040, 9.8284025, -11.8848896, 9.8237076, -14.9978333, 14.9938927
13: -18.5657101, 6.7253323, -18.5547447, 6.7227755, -16.5822754, 16.5925484
14: 4.9599571, 36.4201546, 4.9888067, 36.4057693, -26.6994934, 26.6706772
15: -8.7031527, 9.2884483, -8.6946688, 9.2840090, -17.9871616, 17.9831161
16: -16.7262917, 2.5395625, -16.7211876, 2.5344133, -14.7971115, 14.7958565
17: 6.2086959, 30.6576996, 6.2282424, 30.6491623, -17.2129898, 17.2036972
18: -14.3952599, 5.1264176, -14.3939896, 5.1194963, -14.4037457, 14.4082985
19: -20.2771740, -4.3246279, -20.2713337, -4.3241796, -14.5351334, 14.5199432
20: -2.4242539, 11.2186403, -2.4168141, 11.2132502, -12.6078529, 12.6052742
21: -11.0761051, 3.2496972, -11.0691557, 3.2494001, -14.3255053, 14.3188534
22: -3.6978617, 13.1088924, -3.6870131, 13.1088753, -14.9460945, 14.9178200
23: -14.5806885, 0.3466423, -14.5778217, 0.3448796, -14.3207397, 14.2911263
24: -19.9378777, -5.1141558, -19.9347057, -5.1133704, -9.2643127, 9.2629318
25: -5.4548311, 10.8602991, -5.4436126, 10.8593884, -13.8035583, 13.7831383
26: -21.0118942, 1.2105484, -20.9960575, 1.2094774, -19.3438072, 19.2885742
27: -16.0102501, 2.1736822, -16.0004845, 2.1639905, -13.2057381, 13.1954803
28: -12.7961884, 4.6404943, -12.7874966, 4.6367531, -17.4329414, 17.4279900
29: -5.5925608, 11.8890610, -5.5809193, 11.8887997, -14.9577255, 14.9227638
30: -10.0471697, 6.2068338, -10.0411491, 6.2051959, -13.5496063, 13.5395508
31: -10.9758120, 6.9496641, -10.9721861, 6.9468975, -14.6402283, 14.6318855
32: -24.9242096, -4.5658841, -24.9094601, -4.5788898, -13.2572441, 13.2722015
33: -69.3128662, -40.1018295, -69.3065872, -40.1108704, -16.6241226, 16.6413841
34: -53.7628899, -30.9095802, -53.7525673, -30.9221134, -14.0929909, 14.1144142
35: -47.8218880, -26.0653458, -47.8167915, -26.0692978, -12.9880905, 12.9737129
36: -42.8233109, -19.2848644, -42.8142776, -19.2890415, -15.0932846, 15.0632515
37: -86.6762085, -55.5445786, -86.6710510, -55.5472565, -18.8998947, 18.9036636
38: -52.9502258, -24.3337193, -52.9355659, -24.3442326, -18.3202972, 18.3146057
39: -76.5589981, -44.6219177, -76.5567169, -44.6243591, -16.0744324, 16.0522537
40: -67.2515869, -43.5265656, -67.2440338, -43.5436172, -14.3143044, 14.3174000
41: -55.4294357, -32.9586449, -55.4171219, -32.9740982, -16.6521034, 16.6506844
42: -29.4691238, -9.8806086, -29.4583950, -9.8897629, -17.2428665, 17.2309494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5502461, upper bound: 12.4964717
time: 15.98 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5672273, upper bound: 12.4974105
time: 6.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1223202, 3.6782651, -12.1215305, 3.6734991, -13.8775673, 13.8856964
1: -3.6689796, 7.3990030, -3.6688042, 7.3943949, -8.4866638, 8.4976521
2: -0.7432488, 13.4342852, -0.7434341, 13.4261932, -13.4393387, 13.4484901
3: -1.1289856, 11.3123474, -1.1295443, 11.3054628, -12.0171738, 12.0163403
4: -11.1134529, 5.4870791, -11.1122198, 5.4814320, -14.6620102, 14.6800880
5: 1.8431792, 17.7485580, 1.8425803, 17.7409191, -15.8977394, 15.9059772
6: -39.9309921, -18.2128716, -39.9301682, -18.2182522, -15.1028290, 15.1879654
7: -3.5794756, 12.2584982, -3.5795357, 12.2481918, -13.5858994, 13.6189423
8: -6.7059765, 8.5722141, -6.7064409, 8.5664129, -12.0916977, 12.0971031
9: -4.7920523, 11.7183466, -4.7869921, 11.7171965, -13.0147324, 12.9843025
10: 1.3055911, 25.7432308, 1.3100991, 25.7420082, -20.9172287, 20.9260025
11: -11.5069160, 4.2873325, -11.5036764, 4.2870092, -15.7939253, 15.7910089
12: -11.9075556, 9.8291397, -11.8965654, 9.8283272, -15.0109482, 14.9993668
13: -18.5723991, 6.7270460, -18.5647869, 6.7280107, -16.6055756, 16.5882683
14: 4.9391193, 36.4209404, 4.9530411, 36.4209595, -26.7387466, 26.6857147
15: -8.7083015, 9.2907524, -8.7035589, 9.2896566, -17.9979591, 17.9943123
16: -16.7384739, 2.5401249, -16.7383270, 2.5320344, -14.7975845, 14.8128281
17: 6.1986070, 30.6582947, 6.2104993, 30.6576481, -17.2205963, 17.2146988
18: -14.3965578, 5.1286836, -14.3950281, 5.1232224, -14.4125748, 14.4104614
19: -20.2797012, -4.3212295, -20.2760353, -4.3195124, -14.5425110, 14.5294914
20: -2.4260156, 11.2274446, -2.4238899, 11.2268314, -12.6166916, 12.6181984
21: -11.0788984, 3.2509017, -11.0735264, 3.2505178, -14.3294163, 14.3244286
22: -3.6988754, 13.1131010, -3.6884909, 13.1134109, -14.9522400, 14.9252281
23: -14.5831470, 0.3504434, -14.5803137, 0.3501971, -14.3292160, 14.3002014
24: -19.9392223, -5.1134667, -19.9369888, -5.1127009, -9.2670746, 9.2664185
25: -5.4588590, 10.8608599, -5.4505672, 10.8620005, -13.8066864, 13.7881927
26: -21.0148735, 1.2121227, -21.0013885, 1.2125399, -19.3496437, 19.2960968
27: -16.0114059, 2.1826549, -16.0104675, 2.1785610, -13.1993790, 13.2158966
28: -12.7979221, 4.6469202, -12.7943420, 4.6474528, -17.4453754, 17.4412613
29: -5.5936456, 11.8905201, -5.5830183, 11.8906527, -14.9605255, 14.9338379
30: -10.0512428, 6.2073860, -10.0474539, 6.2072515, -13.5520363, 13.5448303
31: -10.9786186, 6.9528046, -10.9767265, 6.9503479, -14.6461525, 14.6386909
32: -24.9252872, -4.5524273, -24.9243183, -4.5565805, -13.2580032, 13.2985802
33: -69.3138733, -40.0926285, -69.3137665, -40.0958862, -16.6308899, 16.6432343
34: -53.7633781, -30.8977928, -53.7629280, -30.9020119, -14.0977631, 14.1379700
35: -47.8223305, -26.0594196, -47.8208771, -26.0604839, -12.9967270, 12.9844475
36: -42.8234787, -19.2742691, -42.8211861, -19.2718849, -15.1006317, 15.0803108
37: -86.6772308, -55.5401764, -86.6745758, -55.5400925, -18.9101562, 18.9064941
38: -52.9507904, -24.3191090, -52.9494209, -24.3204994, -18.3207703, 18.3441696
39: -76.5602264, -44.6185226, -76.5588837, -44.6190109, -16.0854301, 16.0565491
40: -67.2524567, -43.5182266, -67.2516632, -43.5297432, -14.3064079, 14.3363152
41: -55.4303665, -32.9464226, -55.4295959, -32.9533730, -16.6513863, 16.6772537
42: -29.4700775, -9.8698215, -29.4691257, -9.8715878, -17.2430458, 17.2512589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5514907, upper bound: 12.5346193
time: 9.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5683136, upper bound: 12.5354364
time: 23.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.0984364, 3.6694834, -12.1269646, 3.6810281, -13.8654251, 13.8855858
1: -3.6618834, 7.3955774, -3.6773052, 7.4012899, -8.4812355, 8.4938240
2: -0.7284061, 13.4296474, -0.7598637, 13.4377975, -13.4356384, 13.4615021
3: -1.1166521, 11.3064756, -1.1352004, 11.3165646, -12.0122223, 12.0199165
4: -11.1081600, 5.4864478, -11.1232395, 5.4897909, -14.6624680, 14.6806831
5: 1.8589473, 17.7441673, 1.8323364, 17.7515869, -15.8905716, 15.9118309
6: -39.9156799, -18.2431602, -39.9331093, -18.2341118, -15.0769615, 15.1634865
7: -3.5643883, 12.2605190, -3.6035502, 12.2649708, -13.5799255, 13.6509247
8: -6.6818733, 8.5609026, -6.7114162, 8.5745392, -12.0784798, 12.0931683
9: -4.7607393, 11.7043743, -4.7728920, 11.7227201, -12.9930801, 12.9588432
10: 1.3330102, 25.7253380, 1.3127394, 25.7475815, -20.9040604, 20.9101639
11: -11.4993076, 4.2826815, -11.5072050, 4.2860518, -15.7853594, 15.7898865
12: -11.9083729, 9.8172417, -11.9120216, 9.8521795, -15.0418129, 15.0017853
13: -18.5620937, 6.7156096, -18.5540943, 6.7332077, -16.5656052, 16.5955353
14: 4.9846897, 36.3961182, 4.9572048, 36.4319496, -26.7041473, 26.6576157
15: -8.6833801, 9.2721901, -8.6893806, 9.2894983, -17.9728775, 17.9615707
16: -16.7081318, 2.5314262, -16.7439690, 2.5473680, -14.7844467, 14.7872543
17: 6.2213950, 30.6457005, 6.2049823, 30.6769428, -17.2251816, 17.2085876
18: -14.3809662, 5.1189995, -14.4029484, 5.1255226, -14.3977509, 14.4062347
19: -20.2626705, -4.3420243, -20.2802620, -4.3329697, -14.5235672, 14.5214081
20: -2.4097364, 11.1993675, -2.4275024, 11.2124996, -12.5874672, 12.6009827
21: -11.0617561, 3.2209363, -11.0843296, 3.2468061, -14.3085623, 14.3052654
22: -3.6971467, 13.0954075, -3.7060871, 13.1361694, -14.9754868, 14.9240723
23: -14.5785427, 0.3451283, -14.5850487, 0.3484182, -14.3249435, 14.2941132
24: -19.9264908, -5.1384363, -19.9370975, -5.1247292, -9.2478065, 9.2483521
25: -5.4546270, 10.8429031, -5.4661322, 10.8767948, -13.8151321, 13.7748947
26: -21.0158272, 1.1912084, -21.0237732, 1.2444091, -19.3777084, 19.2899170
27: -16.0063210, 2.1769719, -16.0241661, 2.1808934, -13.2026062, 13.2024612
28: -12.7945347, 4.6364565, -12.8001881, 4.6442413, -17.4387760, 17.4366455
29: -5.5981164, 11.8818684, -5.5993919, 11.9209938, -14.9955940, 14.9376907
30: -10.0467186, 6.1992769, -10.0535011, 6.2193298, -13.5579491, 13.5419769
31: -10.9526739, 6.9337683, -10.9825764, 6.9406323, -14.6156120, 14.6280098
32: -24.9138889, -4.5678053, -24.9279652, -4.5612183, -13.2444878, 13.2890625
33: -69.2880096, -40.1381073, -69.3180084, -40.1138153, -16.5886612, 16.6104317
34: -53.7511253, -30.9248734, -53.7669373, -30.9075108, -14.0789909, 14.1150475
35: -47.7974548, -26.1006451, -47.8099060, -26.0870247, -12.9611588, 12.9606514
36: -42.7984695, -19.3210945, -42.8089142, -19.2979660, -15.0582581, 15.0525742
37: -86.6552505, -55.5657730, -86.6700974, -55.5516891, -18.8970413, 18.8975372
38: -52.9192657, -24.3946686, -52.9529533, -24.3598328, -18.2491531, 18.3017578
39: -76.5269318, -44.6665077, -76.5477905, -44.6455994, -16.0409088, 16.0319252
40: -67.2412186, -43.5100288, -67.2818222, -43.5090675, -14.3042336, 14.3169060
41: -55.4183884, -32.9651794, -55.4418564, -32.9518661, -16.6407471, 16.6455193
42: -29.4641037, -9.8827400, -29.4641590, -9.8852234, -17.2406197, 17.2273293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5380587, upper bound: 12.5681544
time: 12.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5546432, upper bound: 12.5690953
time: 7.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.1214256, 3.6813674, -12.1379995, 3.6829793, -13.8852844, 13.9085693
1: -3.6651926, 7.4014015, -3.6761551, 7.4008479, -8.4872017, 8.5045815
2: -0.7433346, 13.4380817, -0.7678592, 13.4366074, -13.4440765, 13.4772415
3: -1.1286072, 11.3127060, -1.1402540, 11.3097115, -12.0195847, 12.0320435
4: -11.1127691, 5.4892898, -11.1239529, 5.4909682, -14.6621857, 14.6902618
5: 1.8432341, 17.7499275, 1.8223820, 17.7471142, -15.9038801, 15.9275455
6: -39.9305534, -18.2277489, -39.9229507, -18.2389374, -15.1143990, 15.1582909
7: -3.5789645, 12.2637205, -3.6100006, 12.2624760, -13.6016693, 13.6621170
8: -6.7048254, 8.5751686, -6.7233315, 8.5741653, -12.0843735, 12.1191292
9: -4.7823238, 11.7189598, -4.7759204, 11.7216272, -13.0088921, 12.9852333
10: 1.3167715, 25.7436142, 1.3234744, 25.7400913, -20.9020691, 20.9211578
11: -11.5045128, 4.2875175, -11.5036507, 4.2887673, -15.7932796, 15.7911682
12: -11.9076366, 9.8292627, -11.9018288, 9.8544884, -15.0432968, 15.0099144
13: -18.5580082, 6.7255487, -18.5452843, 6.7337217, -16.5490646, 16.6162567
14: 4.9516783, 36.4203415, 4.9626493, 36.4190979, -26.7107391, 26.6966629
15: -8.6998816, 9.2891970, -8.6923676, 9.2882719, -17.9881535, 17.9815636
16: -16.7278957, 2.5471399, -16.7442589, 2.5513396, -14.8116646, 14.8068695
17: 6.1999817, 30.6582127, 6.2050066, 30.6699181, -17.2370834, 17.2277374
18: -14.3964205, 5.1296635, -14.4045429, 5.1286550, -14.4128456, 14.4150352
19: -20.2786446, -4.3246546, -20.2796555, -4.3251629, -14.5443268, 14.5306778
20: -2.4259024, 11.2182617, -2.4230130, 11.2149487, -12.6130791, 12.6116867
21: -11.0807877, 3.2499542, -11.0836000, 3.2632446, -14.3440323, 14.3335543
22: -3.7055988, 13.1092377, -3.7067220, 13.1416874, -14.9896049, 14.9333458
23: -14.5831642, 0.3466299, -14.5863209, 0.3456368, -14.3327560, 14.2948990
24: -19.9376259, -5.1139660, -19.9376354, -5.1107378, -9.2740593, 9.2695084
25: -5.4626813, 10.8603497, -5.4632664, 10.8844185, -13.8370895, 13.7966003
26: -21.0224419, 1.2113016, -21.0209484, 1.2532196, -19.3961411, 19.3060150
27: -16.0108910, 2.1763115, -16.0164642, 2.1709037, -13.2166481, 13.1945152
28: -12.7987900, 4.6408043, -12.7966118, 4.6393065, -17.4380970, 17.4374161
29: -5.5998693, 11.8893547, -5.5992279, 11.9237814, -15.0006866, 14.9379578
30: -10.0501862, 6.2070556, -10.0501490, 6.2211208, -13.5687866, 13.5478516
31: -10.9772930, 6.9503403, -10.9847908, 6.9486485, -14.6450806, 14.6386108
32: -24.9245987, -4.5655880, -24.9154587, -4.5745487, -13.2644958, 13.2735405
33: -69.3132858, -40.0992661, -69.3153458, -40.1015854, -16.6330872, 16.6440697
34: -53.7632980, -30.9053440, -53.7573242, -30.9100723, -14.1052475, 14.1146774
35: -47.8145447, -26.0643902, -47.8070450, -26.0719185, -12.9934082, 12.9755783
36: -42.8150063, -19.2841644, -42.8028641, -19.2878723, -15.0953979, 15.0641975
37: -86.6728668, -55.5430679, -86.6702881, -55.5434685, -18.9181519, 18.9152985
38: -52.9514160, -24.3320465, -52.9409447, -24.3390808, -18.3271065, 18.3216095
39: -76.5524979, -44.6212311, -76.5496521, -44.6230316, -16.0837059, 16.0571098
40: -67.2518082, -43.5163345, -67.2794266, -43.5220032, -14.3258514, 14.3042221
41: -55.4298744, -32.9526939, -55.4310455, -32.9585915, -16.6701202, 16.6407700
42: -29.4693108, -9.8891106, -29.4559784, -9.9013309, -17.2632942, 17.2142906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5511664, upper bound: 12.5315277
time: 23.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5680720, upper bound: 12.5325460
time: 7.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.1229210, 3.6818955, -12.1404152, 3.6824074, -13.8857651, 13.9118271
1: -3.6692214, 7.4024429, -3.6814475, 7.4032497, -8.4931030, 8.5093479
2: -0.7440368, 13.4395494, -0.7687323, 13.4391270, -13.4507561, 13.4793091
3: -1.1293004, 11.3178759, -1.1423732, 11.3185005, -12.0263977, 12.0402527
4: -11.1139288, 5.4908895, -11.1259403, 5.4919791, -14.6717987, 14.6975174
5: 1.8425989, 17.7534904, 1.8234520, 17.7529507, -15.9102631, 15.9300385
6: -39.9313812, -18.2152424, -39.9348564, -18.2179375, -15.1097832, 15.1826706
7: -3.5799429, 12.2658205, -3.6120276, 12.2661200, -13.5998077, 13.6673813
8: -6.7067146, 8.5760784, -6.7258587, 8.5759029, -12.0972595, 12.1219196
9: -4.7911615, 11.7194300, -4.7897148, 11.7248659, -13.0194626, 12.9915619
10: 1.3006835, 25.7443581, 1.2961783, 25.7503510, -20.9286804, 20.9376373
11: -11.5089254, 4.2876339, -11.5104008, 4.2886372, -15.7975626, 15.7980347
12: -11.9147949, 9.8300056, -11.9135065, 9.8591156, -15.0564308, 15.0153999
13: -18.5646553, 6.7273083, -18.5552864, 6.7389107, -16.5723801, 16.6119766
14: 4.9308357, 36.4210968, 4.9268093, 36.4343300, -26.7499847, 26.7117157
15: -8.7050190, 9.2915125, -8.7013063, 9.2938805, -17.9988995, 17.9928188
16: -16.7400551, 2.5477030, -16.7613640, 2.5489554, -14.8121262, 14.8237801
17: 6.1898818, 30.6588402, 6.1872544, 30.6784096, -17.2446938, 17.2387505
18: -14.3976917, 5.1319056, -14.4055576, 5.1323686, -14.4216843, 14.4172173
19: -20.2812099, -4.3212481, -20.2843781, -4.3204794, -14.5517349, 14.5402298
20: -2.4276443, 11.2270527, -2.4300885, 11.2285538, -12.6219177, 12.6246033
21: -11.0836010, 3.2511463, -11.0879917, 3.2643626, -14.3479633, 14.3391380
22: -3.7066259, 13.1134624, -3.7082205, 13.1462479, -14.9957466, 14.9407692
23: -14.5856285, 0.3504376, -14.5888176, 0.3509650, -14.3412552, 14.3039703
24: -19.9389820, -5.1132994, -19.9399223, -5.1100559, -9.2768021, 9.2729950
25: -5.4667053, 10.8608942, -5.4702191, 10.8870220, -13.8401871, 13.8016624
26: -21.0254173, 1.2128725, -21.0262222, 1.2563057, -19.4020042, 19.3135376
27: -16.0120392, 2.1852489, -16.0264492, 2.1854906, -13.2102890, 13.2149467
28: -12.8005772, 4.6471763, -12.8034458, 4.6499929, -17.4505692, 17.4506226
29: -5.6009474, 11.8908157, -5.6013498, 11.9256248, -15.0034790, 14.9490280
30: -10.0542326, 6.2076397, -10.0564413, 6.2231836, -13.5712051, 13.5531425
31: -10.9800911, 6.9534731, -10.9893322, 6.9520831, -14.6510010, 14.6453934
32: -24.9256802, -4.5521040, -24.9303246, -4.5522208, -13.2652245, 13.2999153
33: -69.3142395, -40.0900497, -69.3225708, -40.0865059, -16.6398468, 16.6459312
34: -53.7637825, -30.8935680, -53.7676544, -30.8899841, -14.1100235, 14.1382446
35: -47.8149757, -26.0584564, -47.8111153, -26.0631218, -13.0020599, 12.9863205
36: -42.8151855, -19.2735958, -42.8097305, -19.2707253, -15.1027489, 15.0812798
37: -86.6739349, -55.5387268, -86.6738510, -55.5363159, -18.9284325, 18.9181442
38: -52.9518929, -24.3174744, -52.9548607, -24.3153744, -18.3276100, 18.3511581
39: -76.5536728, -44.6178741, -76.5518188, -44.6177216, -16.0947151, 16.0614052
40: -67.2526855, -43.5079956, -67.2870636, -43.5081062, -14.3179398, 14.3231373
41: -55.4308167, -32.9404907, -55.4434738, -32.9378815, -16.6694031, 16.6673622
42: -29.4702740, -9.8783360, -29.4666901, -9.8831644, -17.2634506, 17.2346420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=92, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 889
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 889

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5523269, upper bound: 12.5681728
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5691131, upper bound: 12.5691125
time: 9.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.27 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5501213, upper bound: 12.4808119
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5671078, upper bound: 12.4817813
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5513718, upper bound: 12.5187837
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5681960, upper bound: 12.5196318
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5510456, upper bound: 12.5150135
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5679536, upper bound: 12.5160780
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5522080, upper bound: 12.5513824
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5689958, upper bound: 12.5523337
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5502461, upper bound: 12.4964717
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5672273, upper bound: 12.4974105
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5514907, upper bound: 12.5346193
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5683136, upper bound: 12.5354364
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5380587, upper bound: 12.5681544
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5546432, upper bound: 12.5690953
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5511664, upper bound: 12.5315277
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5680720, upper bound: 12.5325460
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5523269, upper bound: 12.5681728
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.27
Output dim: 14, lower bound: -12.5691131, upper bound: 12.5691125

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.0972233, 3.6533482, -12.1134863, 3.6600888, -13.8387299, 13.8388824
1: -3.6478662, 7.3860331, -3.6535065, 7.3877416, -8.4694023, 8.4588947
2: -0.7294435, 13.4247570, -0.7361196, 13.4189301, -13.4131508, 13.4156227
3: -1.1137763, 11.2963104, -1.1190724, 11.2913742, -11.9816666, 11.9955330
4: -11.0815811, 5.4591422, -11.0927200, 5.4750433, -14.6261902, 14.6202850
5: 1.8621159, 17.7391968, 1.8517466, 17.7322426, -15.8701267, 15.8874502
6: -39.8545303, -18.2751999, -39.8747215, -18.2439346, -15.0965576, 15.0857010
7: -3.5425017, 12.2348814, -3.5581863, 12.2392311, -13.5716171, 13.5569954
8: -6.6977468, 8.5600777, -6.7021422, 8.5585728, -12.0622864, 12.0871887
9: -4.7499542, 11.6792164, -4.7665858, 11.6927500, -12.9520912, 12.9582977
10: 1.3514819, 25.7313766, 1.3531537, 25.7262154, -20.8608475, 20.8542633
11: -11.4851322, 4.2843776, -11.4893980, 4.2852039, -15.7703362, 15.7737751
12: -11.8795300, 9.8106928, -11.8739529, 9.8214207, -14.9676819, 14.9646225
13: -18.5532684, 6.6943655, -18.5490856, 6.7072535, -16.6028214, 16.5528831
14: 5.0001402, 36.3830185, 4.9965010, 36.3852272, -26.6534042, 26.6501160
15: -8.6546249, 9.2203331, -8.6852455, 9.2452717, -17.8998966, 17.9055786
16: -16.6988335, 2.5350420, -16.7080498, 2.5307493, -14.7578545, 14.7723083
17: 6.2369652, 30.6309433, 6.2321477, 30.6349983, -17.1569214, 17.1741524
18: -14.3566856, 5.1092958, -14.3721809, 5.1138120, -14.3601074, 14.3642578
19: -20.2572708, -4.3366323, -20.2626991, -4.3312683, -14.5044479, 14.5006714
20: -2.4031968, 11.2065182, -2.4055197, 11.2101603, -12.5855217, 12.5854111
21: -11.0544930, 3.2441194, -11.0573063, 3.2483594, -14.3028526, 14.3014259
22: -3.6782529, 13.0671196, -3.6819692, 13.0852728, -14.8745270, 14.8819618
23: -14.5477381, 0.3034105, -14.5723591, 0.3196590, -14.2641373, 14.2682228
24: -19.9277954, -5.1243343, -19.9300537, -5.1185603, -9.2543106, 9.2439041
25: -5.4346218, 10.8322506, -5.4367418, 10.8445063, -13.7533684, 13.7580032
26: -20.9840221, 1.1568921, -20.9914265, 1.1786950, -19.2517433, 19.2688065
27: -15.9914742, 2.1595578, -15.9939203, 2.1563222, -13.1650772, 13.1908264
28: -12.7639236, 4.5985274, -12.7825890, 4.6120701, -17.3759937, 17.3811169
29: -5.5544381, 11.8401661, -5.5769930, 11.8608360, -14.8628540, 14.8895683
30: -10.0364122, 6.1996717, -10.0362024, 6.2002907, -13.5177002, 13.5242043
31: -10.9430351, 6.9479666, -10.9554005, 6.9460621, -14.6033516, 14.6226196
32: -24.8894386, -4.5925746, -24.8898544, -4.5846653, -13.2500572, 13.2513161
33: -69.2788467, -40.1377907, -69.2869186, -40.1180954, -16.6032639, 16.5727654
34: -53.7286301, -30.9336758, -53.7320366, -30.9262905, -14.0893364, 14.0936890
35: -47.8104591, -26.0778313, -47.8104706, -26.0726643, -12.9865265, 12.9584312
36: -42.8217926, -19.2910614, -42.8093414, -19.2910442, -15.0620079, 15.0509033
37: -86.6671448, -55.5588112, -86.6654205, -55.5547028, -18.8888321, 18.8832169
38: -52.9013863, -24.3669987, -52.9084473, -24.3469639, -18.3002357, 18.2544785
39: -76.5299454, -44.6459198, -76.5410690, -44.6290894, -16.0397987, 16.0084267
40: -67.2109985, -43.5417175, -67.2242203, -43.5482788, -14.2645645, 14.3049469
41: -55.4054604, -32.9735374, -55.4048157, -32.9792709, -16.6353188, 16.6535072
42: -29.4543934, -9.8879976, -29.4511375, -9.8935251, -17.2212448, 17.2346649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5667268, upper bound: 12.4406172
time: 11.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5667267, upper bound: 12.4813845
time: 11.21 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.0987110, 3.6538835, -12.1159077, 3.6594980, -13.8392296, 13.8421440
1: -3.6519027, 7.3870916, -3.6587753, 7.3901663, -8.4753036, 8.4636230
2: -0.7301437, 13.4262142, -0.7369888, 13.4214535, -13.4198151, 13.4176674
3: -1.1144663, 11.3015060, -1.1211978, 11.3001785, -11.9884872, 12.0037270
4: -11.0827637, 5.4607162, -11.0946817, 5.4760609, -14.6357498, 14.6275673
5: 1.8614669, 17.7427330, 1.8528152, 17.7380962, -15.8766289, 15.8899174
6: -39.8553200, -18.2627144, -39.8866463, -18.2229156, -15.0919495, 15.1100731
7: -3.5434554, 12.2369690, -3.5602105, 12.2428989, -13.5697403, 13.5622444
8: -6.6996727, 8.5609989, -6.7046742, 8.5603018, -12.0751877, 12.0899658
9: -4.7588139, 11.6797085, -4.7803965, 11.6959877, -12.9626427, 12.9645920
10: 1.3354015, 25.7321148, 1.3258481, 25.7364769, -20.8873749, 20.8707657
11: -11.4895468, 4.2844992, -11.4961224, 4.2850485, -15.7745953, 15.7806215
12: -11.8866796, 9.8114395, -11.8856153, 9.8260431, -14.9808121, 14.9701424
13: -18.5599537, 6.6961393, -18.5591106, 6.7124386, -16.6261749, 16.5485306
14: 4.9792662, 36.3838158, 4.9607353, 36.4004059, -26.6926270, 26.6651993
15: -8.6597681, 9.2226591, -8.6941547, 9.2509480, -17.9107170, 17.9168129
16: -16.7110081, 2.5355814, -16.7251358, 2.5283506, -14.7582550, 14.7892456
17: 6.2268548, 30.6315651, 6.2143478, 30.6434956, -17.1645355, 17.1851730
18: -14.3579884, 5.1115551, -14.3732548, 5.1175537, -14.3689613, 14.3664169
19: -20.2598381, -4.3332243, -20.2674141, -4.3266315, -14.5118484, 14.5101929
20: -2.4049234, 11.2153111, -2.4126127, 11.2237263, -12.5943604, 12.5983276
21: -11.0572729, 3.2452893, -11.0617094, 3.2494931, -14.3067665, 14.3069992
22: -3.6792703, 13.0713644, -3.6834402, 13.0898323, -14.8806839, 14.8893623
23: -14.5502005, 0.3072362, -14.5748825, 0.3250012, -14.2726555, 14.2772865
24: -19.9291763, -5.1236868, -19.9323597, -5.1178746, -9.2570496, 9.2473946
25: -5.4386759, 10.8328056, -5.4436908, 10.8471317, -13.7564774, 13.7630348
26: -20.9869881, 1.1584752, -20.9967079, 1.1817675, -19.2575569, 19.2762909
27: -15.9926214, 2.1685195, -16.0039253, 2.1708660, -13.1586952, 13.2112541
28: -12.7656870, 4.6049533, -12.7894373, 4.6227551, -17.3884430, 17.3943901
29: -5.5555100, 11.8415985, -5.5790863, 11.8626862, -14.8656387, 14.9006462
30: -10.0405064, 6.2002144, -10.0424948, 6.2023706, -13.5201416, 13.5294838
31: -10.9458218, 6.9511003, -10.9599476, 6.9495049, -14.6092644, 14.6294212
32: -24.8904686, -4.5790863, -24.9047279, -4.5623531, -13.2508202, 13.2776909
33: -69.2798462, -40.1285324, -69.2941284, -40.1030807, -16.6100388, 16.5745850
34: -53.7291298, -30.9218578, -53.7424011, -30.9061623, -14.0941200, 14.1172180
35: -47.8108940, -26.0718956, -47.8145638, -26.0638313, -12.9951820, 12.9691887
36: -42.8219833, -19.2805004, -42.8162193, -19.2738590, -15.0693970, 15.0679665
37: -86.6681671, -55.5544319, -86.6689072, -55.5475769, -18.8990822, 18.8860321
38: -52.9019089, -24.3523884, -52.9222870, -24.3232536, -18.3007202, 18.2840424
39: -76.5311890, -44.6425247, -76.5431747, -44.6237144, -16.0507927, 16.0127220
40: -67.2118683, -43.5334167, -67.2318268, -43.5343857, -14.2566299, 14.3238754
41: -55.4063759, -32.9612732, -55.4172821, -32.9585800, -16.6346207, 16.6801147
42: -29.4553547, -9.8772240, -29.4618797, -9.8753586, -17.2213974, 17.2549973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677955, upper bound: 12.4781772
time: 7.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5677955, upper bound: 12.5192341
time: 7.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.0978193, 3.6569910, -12.1323681, 3.6689544, -13.8469353, 13.8650055
1: -3.6481040, 7.3894720, -3.6661479, 7.3965731, -8.4758224, 8.4705734
2: -0.7302434, 13.4300261, -0.7613938, 13.4319048, -13.4245453, 13.4464455
3: -1.1140839, 11.3018389, -1.1319207, 11.3044319, -11.9909058, 12.0194244
4: -11.0820599, 5.4629273, -11.1064377, 5.4855690, -14.6359558, 14.6377296
5: 1.8615360, 17.7441311, 1.8325758, 17.7442703, -15.8827343, 15.9115553
6: -39.8548775, -18.2775764, -39.8794289, -18.2436657, -15.1034889, 15.0804062
7: -3.5429363, 12.2421207, -3.5906990, 12.2571411, -13.5855179, 13.6054230
8: -6.6985025, 8.5639381, -6.7215672, 8.5680456, -12.0678940, 12.1120110
9: -4.7490635, 11.6803102, -4.7693090, 11.7004271, -12.9568062, 12.9655228
10: 1.3465781, 25.7324905, 1.3392100, 25.7345886, -20.8722534, 20.8658981
11: -11.4871445, 4.2846866, -11.4960957, 4.2868390, -15.7739830, 15.7807827
12: -11.8867788, 9.8115644, -11.8908844, 9.8522243, -15.0131378, 14.9806557
13: -18.5455990, 6.6945367, -18.5395775, 6.7181792, -16.5696411, 16.5766258
14: 4.9918137, 36.3832321, 4.9703445, 36.3985558, -26.6645508, 26.6760864
15: -8.6513481, 9.2211008, -8.6829338, 9.2496433, -17.9009914, 17.9040337
16: -16.7004585, 2.5425851, -16.7311344, 2.5476229, -14.7723579, 14.7833138
17: 6.2282276, 30.6314831, 6.2089224, 30.6557541, -17.1810150, 17.1981888
18: -14.3578291, 5.1125188, -14.3827343, 5.1229944, -14.3691826, 14.3710213
19: -20.2587547, -4.3366666, -20.2709846, -4.3322597, -14.5136414, 14.5113907
20: -2.4048183, 11.2061520, -2.4117138, 11.2118635, -12.5907478, 12.5918121
21: -11.0591755, 3.2443810, -11.0717583, 3.2622428, -14.3214188, 14.3161392
22: -3.6859796, 13.0674992, -3.7016683, 13.1181316, -14.9180145, 14.8974686
23: -14.5502472, 0.3033977, -14.5808601, 0.3204327, -14.2761612, 14.2719803
24: -19.9275589, -5.1241450, -19.9330025, -5.1159201, -9.2640266, 9.2504616
25: -5.4424558, 10.8323002, -5.4563589, 10.8695221, -13.7868576, 13.7714691
26: -20.9945507, 1.1576376, -21.0163174, 1.2224436, -19.3041000, 19.2862396
27: -15.9921093, 2.1621914, -16.0099144, 2.1632004, -13.1759644, 13.1898575
28: -12.7665501, 4.5988221, -12.7917252, 4.6146274, -17.3811779, 17.3905468
29: -5.5617323, 11.8404408, -5.5953016, 11.8958311, -14.9058151, 14.9047623
30: -10.0394192, 6.1998730, -10.0451984, 6.2162285, -13.5368652, 13.5324898
31: -10.9445095, 6.9486637, -10.9680367, 6.9478216, -14.6081772, 14.6293221
32: -24.8897781, -4.5922613, -24.8958626, -4.5803404, -13.2572784, 13.2526321
33: -69.2792206, -40.1352272, -69.2957230, -40.1087036, -16.6122208, 16.5754967
34: -53.7290459, -30.9294319, -53.7367859, -30.9142513, -14.1016006, 14.0939293
35: -47.8031006, -26.0768719, -47.8007202, -26.0753136, -12.9918633, 12.9602623
36: -42.8135033, -19.2903748, -42.7979202, -19.2899132, -15.0641441, 15.0518188
37: -86.6637878, -55.5573311, -86.6645889, -55.5509338, -18.9070892, 18.8948593
38: -52.9025574, -24.3653069, -52.9138565, -24.3418579, -18.3070831, 18.2614441
39: -76.5234680, -44.6452408, -76.5339813, -44.6277618, -16.0490723, 16.0132942
40: -67.2112198, -43.5314789, -67.2596283, -43.5266800, -14.2760963, 14.2916584
41: -55.4059334, -32.9676132, -55.4186935, -32.9638252, -16.6533661, 16.6436234
42: -29.4546089, -9.8964844, -29.4486942, -9.9050894, -17.2416916, 17.2180176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675580, upper bound: 12.4740339
time: 14.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675580, upper bound: 12.5156778
time: 14.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.0993261, 3.6575212, -12.1347818, 3.6683722, -13.8474350, 13.8682747
1: -3.6521363, 7.3905101, -3.6714015, 7.3989968, -8.4817200, 8.4753208
2: -0.7309480, 13.4314871, -0.7622706, 13.4343796, -13.4312172, 13.4484863
3: -1.1147902, 11.3070049, -1.1340297, 11.3132057, -11.9977188, 12.0276203
4: -11.0832500, 5.4645467, -11.1084032, 5.4865961, -14.6455154, 14.6450310
5: 1.8608904, 17.7477150, 1.8336678, 17.7500877, -15.8891973, 15.9140472
6: -39.8556633, -18.2650967, -39.8913193, -18.2226524, -15.0988693, 15.1047745
7: -3.5439036, 12.2442465, -3.5927145, 12.2608032, -13.5836639, 13.6106529
8: -6.7004395, 8.5648537, -6.7240963, 8.5697689, -12.0807953, 12.1147919
9: -4.7578945, 11.6807566, -4.7831192, 11.7036610, -12.9673653, 12.9718246
10: 1.3305321, 25.7332516, 1.3119392, 25.7448730, -20.8987961, 20.8824387
11: -11.4915533, 4.2848129, -11.5028086, 4.2866654, -15.7782192, 15.7876215
12: -11.8939028, 9.8123150, -11.9025373, 9.8568563, -15.0262718, 14.9861221
13: -18.5522995, 6.6962953, -18.5496044, 6.7233496, -16.5929489, 16.5722809
14: 4.9709301, 36.3839874, 4.9345446, 36.4137878, -26.7037811, 26.6911697
15: -8.6564922, 9.2234221, -8.6918392, 9.2552328, -17.9117241, 17.9152603
16: -16.7126083, 2.5431576, -16.7482071, 2.5452273, -14.7727928, 14.8002319
17: 6.2181420, 30.6320724, 6.1911583, 30.6642551, -17.1886177, 17.2091980
18: -14.3591213, 5.1148024, -14.3837872, 5.1266918, -14.3780289, 14.3731689
19: -20.2613335, -4.3332515, -20.2757263, -4.3276072, -14.5210342, 14.5209465
20: -2.4065557, 11.2149420, -2.4187884, 11.2254553, -12.5995674, 12.6047058
21: -11.0619593, 3.2455568, -11.0761652, 3.2633433, -14.3253021, 14.3217220
22: -3.6869783, 13.0717211, -3.7031388, 13.1226673, -14.9241638, 14.9048729
23: -14.5526752, 0.3071928, -14.5833817, 0.3257565, -14.2846527, 14.2810745
24: -19.9289284, -5.1234984, -19.9352932, -5.1152649, -9.2667847, 9.2539406
25: -5.4464664, 10.8328629, -5.4633083, 10.8721561, -13.7899666, 13.7765236
26: -20.9974785, 1.1592052, -21.0216064, 1.2255301, -19.3099709, 19.2937317
27: -15.9932365, 2.1711550, -16.0198936, 2.1777983, -13.1696091, 13.2103081
28: -12.7683249, 4.6052408, -12.7985516, 4.6252923, -17.3936176, 17.4037933
29: -5.5628214, 11.8418713, -5.5973883, 11.8976555, -14.9085999, 14.9158401
30: -10.0434809, 6.2004375, -10.0514717, 6.2182856, -13.5392914, 13.5377884
31: -10.9473200, 6.9517679, -10.9725685, 6.9512787, -14.6141090, 14.6361160
32: -24.8908634, -4.5787945, -24.9107151, -4.5580149, -13.2580490, 13.2789993
33: -69.2801819, -40.1259842, -69.3029480, -40.0936737, -16.6189728, 16.5772781
34: -53.7295532, -30.9176216, -53.7471581, -30.8941517, -14.1063881, 14.1174698
35: -47.8035126, -26.0709476, -47.8047867, -26.0664692, -13.0005035, 12.9709892
36: -42.8137131, -19.2798119, -42.8047714, -19.2727699, -15.0715027, 15.0689087
37: -86.6648254, -55.5530167, -86.6681519, -55.5438042, -18.9173279, 18.8976593
38: -52.9030380, -24.3507652, -52.9276962, -24.3181591, -18.3075485, 18.2910233
39: -76.5246811, -44.6418839, -76.5361786, -44.6224594, -16.0600510, 16.0175896
40: -67.2120819, -43.5231934, -67.2672272, -43.5127831, -14.2681656, 14.3105888
41: -55.4068298, -32.9553757, -55.4311371, -32.9431458, -16.6526489, 16.6702156
42: -29.4555511, -9.8857136, -29.4594135, -9.8869171, -17.2418556, 17.2383957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5685924, upper bound: 12.5101838
time: 15.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5685924, upper bound: 12.5519297
time: 13.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1202593, 3.6761150, -12.1187944, 3.6732273, -13.8753319, 13.8638611
1: -3.6644459, 7.3962669, -3.6632774, 7.3910575, -8.4794579, 8.4750061
2: -0.7420930, 13.4314709, -0.7423109, 13.4229012, -13.4314384, 13.4328232
3: -1.1279005, 11.3065281, -1.1272027, 11.2963333, -12.0093765, 12.0030632
4: -11.1116657, 5.4836011, -11.1099205, 5.4794011, -14.6507645, 14.6589699
5: 1.8447576, 17.7441483, 1.8420200, 17.7346306, -15.8898735, 15.9021282
6: -39.9285660, -18.2332382, -39.9173355, -18.2435722, -15.1012573, 15.1708908
7: -3.5773189, 12.2515564, -3.5768406, 12.2417812, -13.5856018, 13.5854950
8: -6.7026277, 8.5695953, -6.7031031, 8.5636749, -12.0747261, 12.0973644
9: -4.7825680, 11.7157955, -4.7728577, 11.7128086, -13.0037689, 12.9755516
10: 1.3227010, 25.7413101, 1.3379650, 25.7310905, -20.8957901, 20.9049149
11: -11.5013905, 4.2866049, -11.4963589, 4.2867961, -15.7881870, 15.7829638
12: -11.8987589, 9.8278770, -11.8839846, 9.8234177, -14.9795990, 14.9923553
13: -18.5578079, 6.7236919, -18.5504284, 6.7218895, -16.6137543, 16.5801163
14: 4.9672537, 36.4196396, 4.9928608, 36.4054871, -26.7069702, 26.6642380
15: -8.6907072, 9.2877674, -8.6878405, 9.2836409, -17.9743481, 17.9756088
16: -16.7253799, 2.5350363, -16.7206364, 2.5319123, -14.7947083, 14.7845154
17: 6.2119584, 30.6569176, 6.2300558, 30.6487293, -17.1968460, 17.2011032
18: -14.3939610, 5.1252546, -14.3932381, 5.1188354, -14.4016647, 14.4038353
19: -20.2761593, -4.3247967, -20.2707901, -4.3242655, -14.5331039, 14.5207977
20: -2.4226947, 11.2183990, -2.4159532, 11.2131205, -12.6000214, 12.6038437
21: -11.0748234, 3.2494740, -11.0683384, 3.2492659, -14.3240891, 14.3178120
22: -3.6942093, 13.1080132, -3.6849685, 13.1083851, -14.9180145, 14.9155884
23: -14.5797853, 0.3462481, -14.5773211, 0.3446627, -14.3182182, 14.2929115
24: -19.9327965, -5.1144123, -19.9318314, -5.1135235, -9.2572556, 9.2617607
25: -5.4475126, 10.8596869, -5.4395971, 10.8590336, -13.7809067, 13.7810135
26: -21.0091648, 1.2101865, -20.9945221, 1.2092576, -19.2987404, 19.2866821
27: -16.0087204, 2.1713336, -15.9996243, 2.1627061, -13.1998405, 13.2065277
28: -12.7954016, 4.6401014, -12.7870989, 4.6365242, -17.4319267, 17.4272003
29: -5.5896788, 11.8885031, -5.5793648, 11.8884602, -14.9248810, 14.9205627
30: -10.0455780, 6.2063456, -10.0402813, 6.2049103, -13.5374451, 13.5380592
31: -10.9747791, 6.9494138, -10.9716043, 6.9467568, -14.6353035, 14.6384392
32: -24.9234467, -4.5752211, -24.9090290, -4.5840216, -13.2509880, 13.2832451
33: -69.3121719, -40.1045151, -69.3061523, -40.1123505, -16.6152000, 16.6239128
34: -53.7624893, -30.9115620, -53.7523422, -30.9231834, -14.0855789, 14.1347809
35: -47.8188324, -26.0669861, -47.8151321, -26.0701923, -12.9845047, 12.9752579
36: -42.8215027, -19.2856579, -42.8133316, -19.2894878, -15.0711632, 15.0612717
37: -86.6753693, -55.5469131, -86.6705933, -55.5485229, -18.8956528, 18.9106293
38: -52.9471436, -24.3343239, -52.9338226, -24.3445663, -18.3144073, 18.3128204
39: -76.5583725, -44.6240234, -76.5563431, -44.6254616, -16.0719185, 16.0508423
40: -67.2504044, -43.5342369, -67.2433929, -43.5478516, -14.3029633, 14.3390007
41: -55.4287567, -32.9664764, -55.4167595, -32.9783516, -16.6442299, 16.6718483
42: -29.4681816, -9.8855295, -29.4578972, -9.8924932, -17.2360840, 17.2460785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5668457, upper bound: 12.4562625
time: 6.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5668457, upper bound: 12.4970180
time: 6.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1217680, 3.6766570, -12.1212282, 3.6726236, -13.8758316, 13.8671188
1: -3.6684706, 7.3973141, -3.6685262, 7.3934898, -8.4853592, 8.4797497
2: -0.7427913, 13.4329367, -0.7431918, 13.4254265, -13.4380913, 13.4348602
3: -1.1286083, 11.3116884, -1.1293347, 11.3050966, -12.0162048, 12.0112762
4: -11.1128483, 5.4852123, -11.1118937, 5.4804235, -14.6603470, 14.6662369
5: 1.8440971, 17.7476807, 1.8430910, 17.7404385, -15.8963413, 15.9045897
6: -39.9294090, -18.2207375, -39.9292755, -18.2225742, -15.0966263, 15.1952782
7: -3.5782762, 12.2536383, -3.5788691, 12.2454395, -13.5837479, 13.5907326
8: -6.7045498, 8.5705023, -6.7056394, 8.5654221, -12.0876274, 12.1001396
9: -4.7914071, 11.7162609, -4.7866635, 11.7160425, -13.0143356, 12.9818497
10: 1.3066502, 25.7420578, 1.3106833, 25.7413177, -20.9223251, 20.9214325
11: -11.5057878, 4.2867222, -11.5030842, 4.2866545, -15.7924423, 15.7898064
12: -11.9058895, 9.8286304, -11.8956547, 9.8280277, -14.9927063, 14.9978409
13: -18.5644970, 6.7254515, -18.5604553, 6.7271004, -16.6370010, 16.5758362
14: 4.9463787, 36.4204407, 4.9570427, 36.4207077, -26.7461777, 26.6793365
15: -8.6958561, 9.2900877, -8.6967392, 9.2892799, -17.9851360, 17.9868279
16: -16.7375183, 2.5355749, -16.7378044, 2.5295360, -14.7951584, 14.8014984
17: 6.2018838, 30.6574879, 6.2123232, 30.6572304, -17.2044716, 17.2121124
18: -14.3952541, 5.1275134, -14.3943100, 5.1225901, -14.4104900, 14.4059906
19: -20.2787189, -4.3213902, -20.2754803, -4.3196197, -14.5404778, 14.5303307
20: -2.4244504, 11.2271843, -2.4230180, 11.2267218, -12.6088638, 12.6167297
21: -11.0776196, 3.2506685, -11.0727224, 3.2503829, -14.3280029, 14.3233910
22: -3.6952322, 13.1122770, -3.6864431, 13.1129150, -14.9241638, 14.9230232
23: -14.5822010, 0.3500543, -14.5797930, 0.3499837, -14.3267059, 14.3019409
24: -19.9341583, -5.1137614, -19.9341030, -5.1128559, -9.2600060, 9.2652512
25: -5.4515276, 10.8602467, -5.4465618, 10.8616734, -13.7840271, 13.7860489
26: -21.0121002, 1.2117405, -20.9998226, 1.2123501, -19.3045845, 19.2941589
27: -16.0098705, 2.1802993, -16.0096169, 2.1772752, -13.1934814, 13.2269478
28: -12.7971420, 4.6465034, -12.7939129, 4.6472359, -17.4443779, 17.4404163
29: -5.5907717, 11.8899260, -5.5814371, 11.8903351, -14.9276886, 14.9316406
30: -10.0496311, 6.2068977, -10.0465431, 6.2069602, -13.5398865, 13.5433464
31: -10.9775944, 6.9525466, -10.9761410, 6.9502001, -14.6412201, 14.6452408
32: -24.9245014, -4.5617542, -24.9238968, -4.5617213, -13.2517166, 13.3096313
33: -69.3131256, -40.0952759, -69.3133545, -40.0973434, -16.6219559, 16.6257057
34: -53.7629890, -30.8997345, -53.7627106, -30.9030838, -14.0903931, 14.1582985
35: -47.8192787, -26.0610428, -47.8192177, -26.0613670, -12.9931641, 12.9859924
36: -42.8216515, -19.2751236, -42.8202171, -19.2723484, -15.0784988, 15.0783539
37: -86.6763763, -55.5425262, -86.6741486, -55.5413742, -18.9058914, 18.9134750
38: -52.9476585, -24.3197880, -52.9477234, -24.3208637, -18.3148804, 18.3423615
39: -76.5595703, -44.6206818, -76.5585327, -44.6201859, -16.0829124, 16.0551300
40: -67.2512894, -43.5259247, -67.2509918, -43.5339584, -14.2950554, 14.3579025
41: -55.4297028, -32.9542313, -55.4292221, -32.9576454, -16.6435127, 16.6984749
42: -29.4691448, -9.8747749, -29.4686012, -9.8743048, -17.2362328, 17.2664032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5679130, upper bound: 12.4941422
time: 7.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5679130, upper bound: 12.5350372
time: 8.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.0874081, 3.6467175, -12.1248722, 3.6672547, -13.8385353, 13.8587532
1: -3.6571345, 7.3739243, -3.6767263, 7.3881521, -8.4598236, 8.4707794
2: -0.7155289, 13.4044476, -0.7583789, 13.4229164, -13.4079285, 13.4349632
3: -1.1165926, 11.2888393, -1.1342056, 11.3062305, -12.0019379, 12.0009232
4: -11.0987720, 5.4611864, -11.1222334, 5.4749637, -14.6381989, 14.6545105
5: 1.8635454, 17.7307205, 1.8341703, 17.7437477, -15.8733215, 15.8965502
6: -39.9173927, -18.2476654, -39.9304123, -18.2369347, -15.0752220, 15.1517334
7: -3.5606589, 12.2286911, -3.6022458, 12.2460356, -13.5426102, 13.6128235
8: -6.6734624, 8.5438786, -6.7099094, 8.5641842, -12.0787048, 12.0785236
9: -4.7575860, 11.6908236, -4.7713304, 11.7151232, -12.9779053, 12.9404869
10: 1.3374195, 25.7096596, 1.3153567, 25.7390099, -20.8802109, 20.8891525
11: -11.4931755, 4.2795358, -11.5033531, 4.2850165, -15.7781925, 15.7828884
12: -11.8749809, 9.8057652, -11.8921785, 9.8505726, -15.0076752, 14.9702072
13: -18.5479870, 6.7175837, -18.5456791, 6.7312126, -16.5430260, 16.6069756
14: 5.0277758, 36.3997498, 4.9820499, 36.4311981, -26.6630554, 26.6495667
15: -8.6788054, 9.2898159, -8.6862860, 9.2882509, -17.9670563, 17.9761009
16: -16.7057858, 2.4909391, -16.7421131, 2.5229785, -14.7746201, 14.7408714
17: 6.2791266, 30.6347599, 6.2389684, 30.6758957, -17.1649170, 17.1656952
18: -14.3735638, 5.1092777, -14.3988590, 5.1199584, -14.3838940, 14.3913612
19: -20.2488136, -4.3447680, -20.2723846, -4.3332338, -14.5075150, 14.5057678
20: -2.3906171, 11.1927242, -2.4161305, 11.2111664, -12.5678253, 12.5834312
21: -11.0441151, 3.2129550, -11.0740757, 3.2455854, -14.2897005, 14.2870312
22: -3.6565804, 13.0824747, -3.6815464, 13.1351709, -14.9310989, 14.8802948
23: -14.5648603, 0.3448062, -14.5775490, 0.3478923, -14.3100357, 14.2794037
24: -19.9206581, -5.1333280, -19.9341660, -5.1255121, -9.2339935, 9.2364273
25: -5.4273844, 10.8442402, -5.4505577, 10.8759623, -13.7789612, 13.7446175
26: -20.9593086, 1.1670957, -20.9902267, 1.2432289, -19.3199692, 19.2304382
27: -16.0034199, 2.1717632, -16.0220680, 2.1779242, -13.1998329, 13.1868591
28: -12.7782211, 4.6323390, -12.7913179, 4.6436443, -17.4218655, 17.4236565
29: -5.5495167, 11.8619442, -5.5705576, 11.9203215, -14.9465981, 14.8876686
30: -10.0278778, 6.1894088, -10.0421925, 6.2178020, -13.5377045, 13.5205574
31: -10.9476013, 6.9335070, -10.9800701, 6.9402828, -14.6071129, 14.6137581
32: -24.9225731, -4.5740395, -24.9263191, -4.5643487, -13.2498245, 13.2793770
33: -69.2773972, -40.1639748, -69.3169861, -40.1287041, -16.5640907, 16.5848465
34: -53.7487335, -30.9420109, -53.7658730, -30.9172630, -14.0872345, 14.1028290
35: -47.7913971, -26.1026802, -47.8070717, -26.0892830, -12.9480171, 12.9468956
36: -42.7740860, -19.3289452, -42.7950172, -19.2993164, -15.0303650, 15.0261650
37: -86.6463394, -55.5819969, -86.6656342, -55.5614166, -18.8794212, 18.8738785
38: -52.9017105, -24.3947468, -52.9429054, -24.3609238, -18.2260437, 18.2857666
39: -76.5216217, -44.6842384, -76.5467834, -44.6560745, -16.0323868, 16.0121689
40: -67.2345886, -43.5447159, -67.2798538, -43.5299606, -14.3235359, 14.3038902
41: -55.4246521, -32.9810257, -55.4408340, -32.9608192, -16.6529312, 16.6340752
42: -29.4657459, -9.8863773, -29.4629478, -9.8872795, -17.2402573, 17.2102776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5376607, upper bound: 12.5260235
time: 13.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5376607, upper bound: 12.5677587
time: 11.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.0978756, 3.6678843, -12.1266727, 3.6801274, -13.8637085, 13.8670044
1: -3.6613708, 7.3938675, -3.6770263, 7.4003439, -8.4798965, 8.4759274
2: -0.7279673, 13.4282961, -0.7596291, 13.4370575, -13.4343872, 13.4478912
3: -1.1162747, 11.3058167, -1.1349889, 11.3161907, -12.0112343, 12.0148239
4: -11.1075163, 5.4845810, -11.1229248, 5.4887667, -14.6607971, 14.6668510
5: 1.8598838, 17.7433109, 1.8328385, 17.7510796, -15.8911953, 15.9104729
6: -39.9140816, -18.2510338, -39.9321976, -18.2384567, -15.0707397, 15.1707687
7: -3.5631862, 12.2555943, -3.6029043, 12.2621803, -13.5777588, 13.6227074
8: -6.6804352, 8.5591784, -6.7106309, 8.5735197, -12.0744247, 12.0962372
9: -4.7600975, 11.7022629, -4.7725735, 11.7215662, -12.9926834, 12.9563942
10: 1.3340597, 25.7241936, 1.3133240, 25.7469616, -20.9091339, 20.9055786
11: -11.4981689, 4.2820859, -11.5065899, 4.2857289, -15.7838974, 15.7886753
12: -11.9067163, 9.8167143, -11.9110918, 9.8519049, -15.0235710, 15.0002594
13: -18.5541954, 6.7139282, -18.5497074, 6.7322969, -16.5970535, 16.5831184
14: 4.9919310, 36.3956032, 4.9612188, 36.4317474, -26.7115555, 26.6511993
15: -8.6709270, 9.2715168, -8.6825380, 9.2891083, -17.9600353, 17.9540558
16: -16.7071877, 2.5268147, -16.7434502, 2.5448246, -14.7820129, 14.7759247
17: 6.2246704, 30.6449127, 6.2067933, 30.6765194, -17.2090302, 17.2059517
18: -14.3796444, 5.1178293, -14.4022388, 5.1248946, -14.3956509, 14.4017830
19: -20.2616730, -4.3421698, -20.2797165, -4.3330774, -14.5214996, 14.5222626
20: -2.4081855, 11.1991262, -2.4266403, 11.2123785, -12.5796509, 12.5995255
21: -11.0604763, 3.2207098, -11.0835323, 3.2466717, -14.3071480, 14.3042421
22: -3.6934807, 13.0945349, -3.7040281, 13.1356649, -14.9473801, 14.9218445
23: -14.5776243, 0.3446994, -14.5845385, 0.3482208, -14.3224106, 14.2958984
24: -19.9214573, -5.1387148, -19.9342403, -5.1248970, -9.2407455, 9.2471466
25: -5.4473004, 10.8423128, -5.4620891, 10.8764849, -13.7924652, 13.7727203
26: -21.0130386, 1.1908355, -21.0222473, 1.2441852, -19.3326111, 19.2879486
27: -16.0048065, 2.1746237, -16.0233192, 2.1795850, -13.1967010, 13.2135124
28: -12.7937403, 4.6360474, -12.7997532, 4.6440239, -17.4377632, 17.4358006
29: -5.5952559, 11.8812790, -5.5978112, 11.9206619, -14.9627571, 14.9354897
30: -10.0451097, 6.1987562, -10.0526285, 6.2190394, -13.5457993, 13.5404854
31: -10.9516354, 6.9335008, -10.9820251, 6.9405084, -14.6106720, 14.6345634
32: -24.9130993, -4.5771732, -24.9275322, -4.5663772, -13.2382355, 13.3000793
33: -69.2872925, -40.1407623, -69.3176193, -40.1152802, -16.5797882, 16.5930939
34: -53.7507324, -30.9268265, -53.7667274, -30.9086075, -14.0715714, 14.1354065
35: -47.7944107, -26.1023273, -47.8082237, -26.0879440, -12.9575996, 12.9621735
36: -42.7966690, -19.3219299, -42.8079834, -19.2983837, -15.0361366, 15.0506325
37: -86.6544647, -55.5681190, -86.6696625, -55.5529709, -18.8927994, 18.9045258
38: -52.9161682, -24.3952713, -52.9512596, -24.3601646, -18.2432899, 18.2999039
39: -76.5263443, -44.6686478, -76.5475159, -44.6467896, -16.0383720, 16.0305099
40: -67.2400360, -43.5177231, -67.2811737, -43.5133209, -14.2928772, 14.3384190
41: -55.4177551, -32.9729919, -55.4414597, -32.9561577, -16.6328697, 16.6667519
42: -29.4631996, -9.8876648, -29.4636269, -9.8879051, -17.2338295, 17.2424660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5542391, upper bound: 12.5269607
time: 12.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5542391, upper bound: 12.5686916
time: 7.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.1208725, 3.6797357, -12.1377001, 3.6820941, -13.8835182, 13.8899803
1: -3.6646793, 7.3996897, -3.6758759, 7.3998923, -8.4858742, 8.4866867
2: -0.7428951, 13.4367170, -0.7676013, 13.4358587, -13.4428253, 13.4636269
3: -1.1282183, 11.3120384, -1.1400448, 11.3093643, -12.0186005, 12.0269451
4: -11.1121302, 5.4874091, -11.1236219, 5.4899168, -14.6605682, 14.6764183
5: 1.8441653, 17.7490864, 1.8228760, 17.7466507, -15.9024849, 15.9262104
6: -39.9289398, -18.2356529, -39.9220734, -18.2433109, -15.1082077, 15.1656151
7: -3.5777495, 12.2588177, -3.6093423, 12.2596693, -13.5994873, 13.6338844
8: -6.7033863, 8.5734367, -6.7225285, 8.5731783, -12.0803146, 12.1222057
9: -4.7816858, 11.7168713, -4.7755737, 11.7204790, -13.0084801, 12.9827919
10: 1.3178105, 25.7424068, 1.3240318, 25.7394409, -20.9072418, 20.9165649
11: -11.5033779, 4.2869220, -11.5030289, 4.2884331, -15.7918110, 15.7899513
12: -11.9059982, 9.8287411, -11.9009361, 9.8542223, -15.0250626, 15.0083771
13: -18.5501137, 6.7238798, -18.5409260, 6.7328100, -16.5805435, 16.6038437
14: 4.9589167, 36.4198608, 4.9667263, 36.4188919, -26.7181015, 26.6902161
15: -8.6874580, 9.2885294, -8.6855116, 9.2879066, -17.9753647, 17.9740410
16: -16.7269554, 2.5425744, -16.7437420, 2.5487955, -14.8092461, 14.7954636
17: 6.2032342, 30.6574421, 6.2068267, 30.6694946, -17.2209549, 17.2251129
18: -14.3950958, 5.1284819, -14.4037876, 5.1280107, -14.4107475, 14.4105797
19: -20.2776566, -4.3248081, -20.2791023, -4.3252640, -14.5422707, 14.5315208
20: -2.4243188, 11.2180061, -2.4221573, 11.2148190, -12.6052437, 12.6102371
21: -11.0795040, 3.2497334, -11.0827923, 3.2631283, -14.3426323, 14.3325253
22: -3.7019331, 13.1083632, -3.7046297, 13.1412058, -14.9615059, 14.9310760
23: -14.5822525, 0.3462043, -14.5858021, 0.3454132, -14.3302460, 14.2966881
24: -19.9325714, -5.1142054, -19.9347515, -5.1108994, -9.2669754, 9.2683334
25: -5.4553099, 10.8597355, -5.4592071, 10.8841009, -13.8144150, 13.7944565
26: -21.0196934, 1.2109230, -21.0194359, 1.2530026, -19.3510818, 19.3041077
27: -16.0093727, 2.1739702, -16.0156021, 2.1695938, -13.2107506, 13.2055664
28: -12.7980251, 4.6403980, -12.7961826, 4.6390848, -17.4371109, 17.4365807
29: -5.5970011, 11.8887691, -5.5976448, 11.9234562, -14.9678421, 14.9357681
30: -10.0485992, 6.2065468, -10.0492544, 6.2208366, -13.5565872, 13.5463562
31: -10.9762630, 6.9501038, -10.9842319, 6.9485235, -14.6401558, 14.6451530
32: -24.9238396, -4.5749407, -24.9150047, -4.5796871, -13.2582092, 13.2845688
33: -69.3125610, -40.1018906, -69.3149567, -40.1030006, -16.6241570, 16.6266022
34: -53.7628860, -30.9073391, -53.7571106, -30.9111729, -14.0978470, 14.1350212
35: -47.8115082, -26.0660095, -47.8053932, -26.0728302, -12.9898148, 12.9771004
36: -42.8132324, -19.2849579, -42.8019028, -19.2883415, -15.0732651, 15.0622177
37: -86.6720123, -55.5454369, -86.6698303, -55.5447540, -18.9138870, 18.9222908
38: -52.9482498, -24.3326969, -52.9392357, -24.3394470, -18.3212280, 18.3197784
39: -76.5518646, -44.6233406, -76.5492935, -44.6242142, -16.0811768, 16.0556946
40: -67.2506027, -43.5240326, -67.2787704, -43.5262680, -14.3144989, 14.3257217
41: -55.4292297, -32.9605255, -55.4306717, -32.9628868, -16.6622467, 16.6619835
42: -29.4683933, -9.8940096, -29.4554710, -9.9040308, -17.2565346, 17.2294617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5676761, upper bound: 12.4906921
time: 9.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5676761, upper bound: 12.5321506
time: 9.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.1119118, 3.6591296, -12.1383142, 3.6686471, -13.8588715, 13.8849945
1: -3.6644802, 7.3808384, -3.6808527, 7.3901176, -8.4717026, 8.4862881
2: -0.7311597, 13.4143181, -0.7672280, 13.4242172, -13.4230690, 13.4527664
3: -1.1292334, 11.3002510, -1.1413825, 11.3081732, -12.0160980, 12.0212574
4: -11.1045475, 5.4656219, -11.1249104, 5.4771452, -14.6475296, 14.6713905
5: 1.8471541, 17.7400322, 1.8252754, 17.7451496, -15.8930511, 15.9147568
6: -39.9330788, -18.2197437, -39.9322052, -18.2207527, -15.1080360, 15.1709442
7: -3.5761971, 12.2340088, -3.6107175, 12.2471962, -13.5624695, 13.6292725
8: -6.6983471, 8.5590591, -6.7243443, 8.5655537, -12.0975037, 12.1073055
9: -4.7880030, 11.7058859, -4.7881851, 11.7172527, -13.0042992, 12.9731979
10: 1.3051219, 25.7286720, 1.2987900, 25.7417698, -20.9048157, 20.9166260
11: -11.5028257, 4.2844830, -11.5065002, 4.2875891, -15.7904148, 15.7909832
12: -11.8813801, 9.8185434, -11.8936615, 9.8575020, -15.0222702, 14.9838066
13: -18.5505657, 6.7293100, -18.5468998, 6.7369146, -16.5497894, 16.6233978
14: 4.9738712, 36.4247818, 4.9517059, 36.4335938, -26.7088776, 26.7037125
15: -8.7004471, 9.3091478, -8.6981964, 9.2926226, -17.9930687, 18.0073433
16: -16.7377110, 2.5072684, -16.7595348, 2.5246139, -14.8022766, 14.7774429
17: 6.2476172, 30.6478996, 6.2212462, 30.6773911, -17.1844368, 17.1958504
18: -14.3902874, 5.1222134, -14.4014635, 5.1268139, -14.4078293, 14.4023247
19: -20.2673817, -4.3240080, -20.2764874, -4.3207827, -14.5356445, 14.5245972
20: -2.4085202, 11.2204208, -2.4187582, 11.2272196, -12.6022682, 12.6070366
21: -11.0659561, 3.2431591, -11.0777416, 3.2631540, -14.3291101, 14.3209009
22: -3.6660397, 13.1005478, -3.6836731, 13.1452417, -14.9513626, 14.8969955
23: -14.5719614, 0.3500977, -14.5812826, 0.3504157, -14.3263054, 14.2892647
24: -19.9331398, -5.1081848, -19.9369907, -5.1108618, -9.2629929, 9.2610855
25: -5.4394736, 10.8622389, -5.4546871, 10.8861580, -13.8040199, 13.7713966
26: -20.9689312, 1.1887555, -20.9926491, 1.2551122, -19.3442917, 19.2540436
27: -16.0091305, 2.1800966, -16.0243340, 2.1825235, -13.2075157, 13.1993446
28: -12.7842350, 4.6430931, -12.7945728, 4.6493721, -17.4336071, 17.4376659
29: -5.5523505, 11.8708811, -5.5724812, 11.9249763, -14.9544907, 14.8989944
30: -10.0354290, 6.1977472, -10.0451183, 6.2216754, -13.5509567, 13.5316963
31: -10.9750423, 6.9532042, -10.9868221, 6.9517407, -14.6425209, 14.6311531
32: -24.9343224, -4.5583239, -24.9286861, -4.5553689, -13.2705574, 13.2902412
33: -69.3035965, -40.1159210, -69.3215637, -40.1014061, -16.6150742, 16.6203423
34: -53.7613831, -30.9106827, -53.7666092, -30.8996983, -14.1183052, 14.1259956
35: -47.8088837, -26.0604687, -47.8082695, -26.0653019, -12.9889221, 12.9725761
36: -42.7908516, -19.2814312, -42.7957840, -19.2721062, -15.0748634, 15.0548401
37: -86.6649246, -55.5549240, -86.6693726, -55.5460739, -18.9107666, 18.8944511
38: -52.9343643, -24.3176441, -52.9447708, -24.3164654, -18.3045425, 18.3352051
39: -76.5484085, -44.6355820, -76.5508194, -44.6281357, -16.0861816, 16.0416756
40: -67.2460175, -43.5427017, -67.2850647, -43.5289993, -14.3372345, 14.3101139
41: -55.4371071, -32.9563332, -55.4424896, -32.9468689, -16.6816063, 16.6558762
42: -29.4719124, -9.8819704, -29.4654980, -9.8852377, -17.2631645, 17.2175713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5519331, upper bound: 12.5260421
time: 8.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5519331, upper bound: 12.5677775
time: 13.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.1223755, 3.6802804, -12.1401176, 3.6814981, -13.8840332, 13.8932419
1: -3.6687007, 7.4007316, -3.6811607, 7.4023075, -8.4917831, 8.4914417
2: -0.7435796, 13.4381847, -0.7684562, 13.4383631, -13.4495468, 13.4657059
3: -1.1289208, 11.3172073, -1.1421593, 11.3181381, -12.0254135, 12.0351601
4: -11.1133099, 5.4890079, -11.1255913, 5.4909573, -14.6701126, 14.6836929
5: 1.8435268, 17.7526169, 1.8239579, 17.7524815, -15.9089546, 15.9286594
6: -39.9297752, -18.2231464, -39.9339714, -18.2222900, -15.1035767, 15.1899834
7: -3.5787318, 12.2609158, -3.6113720, 12.2633495, -13.5976257, 13.6391602
8: -6.7052927, 8.5743389, -6.7250609, 8.5748768, -12.0932503, 12.1249771
9: -4.7905235, 11.7173367, -4.7893753, 11.7237320, -13.0190506, 12.9891052
10: 1.3017588, 25.7431698, 1.2967515, 25.7497177, -20.9337540, 20.9330292
11: -11.5078058, 4.2870340, -11.5097618, 4.2882919, -15.7960978, 15.7967958
12: -11.9131250, 9.8294926, -11.9125862, 9.8588457, -15.0381889, 15.0138626
13: -18.5568066, 6.7256413, -18.5509586, 6.7379990, -16.6037979, 16.5995789
14: 4.9380722, 36.4206009, 4.9308920, 36.4340820, -26.7573547, 26.7052536
15: -8.6925869, 9.2908325, -8.6944275, 9.2934980, -17.9860840, 17.9852600
16: -16.7391281, 2.5431457, -16.7608547, 2.5464106, -14.8097115, 14.8124199
17: 6.1931515, 30.6580238, 6.1890564, 30.6779995, -17.2285538, 17.2361374
18: -14.3963861, 5.1307635, -14.4048538, 5.1317244, -14.4195824, 14.4127541
19: -20.2802219, -4.3214245, -20.2838097, -4.3205967, -14.5496521, 14.5410767
20: -2.4260626, 11.2268219, -2.4292369, 11.2284164, -12.6140900, 12.6231346
21: -11.0822897, 3.2509232, -11.0871887, 3.2642379, -14.3465271, 14.3381119
22: -3.7029226, 13.1126080, -3.7061355, 13.1457577, -14.9676590, 14.9384880
23: -14.5846958, 0.3500276, -14.5882950, 0.3507738, -14.3387375, 14.3057365
24: -19.9339046, -5.1135697, -19.9370728, -5.1102219, -9.2697182, 9.2718124
25: -5.4593492, 10.8602905, -5.4661713, 10.8866825, -13.8175430, 13.7994766
26: -21.0226326, 1.2124817, -21.0247135, 1.2560878, -19.3569069, 19.3115997
27: -16.0105228, 2.1829453, -16.0256042, 2.1841731, -13.2043839, 13.2260132
28: -12.7997828, 4.6468163, -12.8030033, 4.6497717, -17.4495544, 17.4498196
29: -5.5980778, 11.8902149, -5.5997610, 11.9253139, -14.9706268, 14.9468155
30: -10.0526428, 6.2071218, -10.0555611, 6.2229147, -13.5590286, 13.5516434
31: -10.9790726, 6.9532223, -10.9887714, 6.9519439, -14.6460686, 14.6519432
32: -24.9248753, -4.5614634, -24.9298782, -4.5573826, -13.2589645, 13.3109550
33: -69.3135300, -40.0926743, -69.3222046, -40.0879745, -16.6309204, 16.6284180
34: -53.7634201, -30.8954983, -53.7674713, -30.8910789, -14.1026192, 14.1585808
35: -47.8119431, -26.0601234, -47.8094406, -26.0639801, -12.9984818, 12.9878502
36: -42.8134422, -19.2744026, -42.8087845, -19.2712250, -15.0806351, 15.0793076
37: -86.6730881, -55.5410919, -86.6733856, -55.5375900, -18.9241409, 18.9251022
38: -52.9488411, -24.3181229, -52.9531479, -24.3157406, -18.3217163, 18.3493576
39: -76.5530777, -44.6200104, -76.5515137, -44.6189270, -16.0921745, 16.0600052
40: -67.2514954, -43.5157471, -67.2863770, -43.5123749, -14.3065948, 14.3446426
41: -55.4301529, -32.9483032, -55.4431648, -32.9421997, -16.6615562, 16.6885796
42: -29.4693584, -9.8832378, -29.4661694, -9.8858566, -17.2567024, 17.2498207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=92, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 918

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5687095, upper bound: 12.5269777
time: 7.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5687095, upper bound: 12.5687088
time: 9.53 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.28 seconds
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5667268, upper bound: 12.4406172
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5667267, upper bound: 12.4813845
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5677955, upper bound: 12.4781772
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5677955, upper bound: 12.5192341
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5675580, upper bound: 12.4740339
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5675580, upper bound: 12.5156778
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5685924, upper bound: 12.5101838
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5685924, upper bound: 12.5519297
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5668457, upper bound: 12.4562625
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5668457, upper bound: 12.4970180
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5679130, upper bound: 12.4941422
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5679130, upper bound: 12.5350372
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5376607, upper bound: 12.5260235
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5376607, upper bound: 12.5677587
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5542391, upper bound: 12.5269607
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5542391, upper bound: 12.5686916
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5676761, upper bound: 12.4906921
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5676761, upper bound: 12.5321506
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5519331, upper bound: 12.5260421
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5519331, upper bound: 12.5677775
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5687095, upper bound: 12.5269777
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 19.28
Output dim: 14, lower bound: -12.5687095, upper bound: 12.5687088

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.0966396, 3.6528497, -12.1121731, 3.6589127, -13.8367538, 13.8366852
1: -3.6471114, 7.3854380, -3.6518359, 7.3864202, -8.4663811, 8.4545403
2: -0.7289669, 13.4241161, -0.7350289, 13.4175119, -13.4110146, 13.4135284
3: -1.1135385, 11.2945032, -1.1185504, 11.2873583, -11.9758530, 11.9925690
4: -11.0808163, 5.4583716, -11.0910625, 5.4734216, -14.6238098, 14.6176605
5: 1.8622847, 17.7383404, 1.8521366, 17.7303352, -15.8680506, 15.8862038
6: -39.8539696, -18.2776413, -39.8734818, -18.2492275, -15.0910454, 15.0821304
7: -3.5419807, 12.2344770, -3.5570641, 12.2383842, -13.5694580, 13.5539284
8: -6.6967325, 8.5596123, -6.6998668, 8.5575218, -12.0595360, 12.0838566
9: -4.7463160, 11.6789351, -4.7585092, 11.6921244, -12.9482918, 12.9507523
10: 1.3570533, 25.7311630, 1.3656240, 25.7257557, -20.8550873, 20.8421249
11: -11.4823322, 4.2839313, -11.4831867, 4.2842050, -15.7665367, 15.7671185
12: -11.8779411, 9.8102674, -11.8703957, 9.8204632, -14.9651833, 14.9606209
13: -18.5517635, 6.6936193, -18.5457077, 6.7056150, -16.6003609, 16.5496597
14: 5.0081358, 36.3827324, 5.0143499, 36.3844757, -26.6445999, 26.6318207
15: -8.6535177, 9.2192507, -8.6827765, 9.2428226, -17.8963394, 17.9020271
16: -16.6944771, 2.5348315, -16.6985741, 2.5301981, -14.7520142, 14.7606773
17: 6.2387967, 30.6306839, 6.2362070, 30.6344318, -17.1532021, 17.1700974
18: -14.3555708, 5.1085672, -14.3696451, 5.1122112, -14.3560600, 14.3585510
19: -20.2561626, -4.3369403, -20.2602615, -4.3319850, -14.4998779, 14.4957542
20: -2.4024131, 11.2058811, -2.4037814, 11.2087307, -12.5835419, 12.5831642
21: -11.0525579, 3.2440128, -11.0530672, 3.2481284, -14.3006859, 14.2970800
22: -3.6776557, 13.0658283, -3.6806240, 13.0823936, -14.8697815, 14.8786392
23: -14.5468655, 0.3024206, -14.5703983, 0.3174925, -14.2606659, 14.2649345
24: -19.9272709, -5.1245279, -19.9288349, -5.1189847, -9.2526627, 9.2416992
25: -5.4335995, 10.8319120, -5.4344330, 10.8437147, -13.7499428, 13.7547989
26: -20.9820347, 1.1565573, -20.9871502, 1.1779475, -19.2469559, 19.2627487
27: -15.9909725, 2.1581941, -15.9927912, 2.1531968, -13.1618576, 13.1879654
28: -12.7633600, 4.5971370, -12.7813158, 4.6090012, -17.3723602, 17.3784523
29: -5.5539951, 11.8398438, -5.5760212, 11.8601589, -14.8602753, 14.8874397
30: -10.0348244, 6.1993608, -10.0327024, 6.1996117, -13.5155182, 13.5205994
31: -10.9414778, 6.9469986, -10.9519691, 6.9439168, -14.5997238, 14.6181030
32: -24.8888397, -4.5948648, -24.8885155, -4.5897646, -13.2436600, 13.2473564
33: -69.2782593, -40.1434860, -69.2855530, -40.1307831, -16.5909958, 16.5664368
34: -53.7282944, -30.9402905, -53.7312508, -30.9407196, -14.0747414, 14.0865936
35: -47.8102722, -26.0811806, -47.8100166, -26.0800724, -12.9792519, 12.9550209
36: -42.8216095, -19.2947884, -42.8088875, -19.2993050, -15.0550919, 15.0475807
37: -86.6665115, -55.5609970, -86.6639709, -55.5596428, -18.8839493, 18.8796997
38: -52.9008789, -24.3717232, -52.9073639, -24.3575935, -18.2905960, 18.2492218
39: -76.5289993, -44.6488876, -76.5389404, -44.6357651, -16.0320129, 16.0027084
40: -67.2105484, -43.5450287, -67.2232056, -43.5556755, -14.2597237, 14.3011436
41: -55.4049339, -32.9781075, -55.4036560, -32.9895248, -16.6275711, 16.6491547
42: -29.4537735, -9.8900166, -29.4497700, -9.8980284, -17.2155304, 17.2311020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5662908, upper bound: 12.4291647
time: 8.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5663000, upper bound: 12.4402004
time: 7.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.0950947, 3.6530991, -12.1108418, 3.6643553, -13.8416786, 13.8366623
1: -3.6438887, 7.3857393, -3.6474195, 7.3900409, -8.4779778, 8.4567070
2: -0.7276857, 13.4243441, -0.7339578, 13.4194775, -13.4130020, 13.4141502
3: -1.1134977, 11.2914057, -1.1186410, 11.2849903, -11.9776726, 11.9985867
4: -11.0770702, 5.4586525, -11.0861616, 5.4744349, -14.6238022, 14.6155167
5: 1.8623018, 17.7372189, 1.8517017, 17.7302666, -15.8679647, 15.8855171
6: -39.8541107, -18.2824917, -39.8807373, -18.2557068, -15.0905304, 15.0864220
7: -3.5414155, 12.2345228, -3.5585320, 12.2416296, -13.5789871, 13.5548019
8: -6.6949701, 8.5597744, -6.6993937, 8.5585957, -12.0612068, 12.0846615
9: -4.7483358, 11.6788979, -4.7662935, 11.7123852, -12.9623108, 12.9551468
10: 1.3528275, 25.7310524, 1.3515949, 25.7449493, -20.8748398, 20.8538437
11: -11.4840221, 4.2833157, -11.4905148, 4.2901497, -15.7741718, 15.7738304
12: -11.8781738, 9.8099613, -11.8760881, 9.8254547, -14.9690628, 14.9656219
13: -18.5521145, 6.6935463, -18.5483437, 6.7162662, -16.6042786, 16.5516129
14: 5.0030088, 36.3827209, 4.9959803, 36.4225311, -26.6802063, 26.6471024
15: -8.6507893, 9.2197456, -8.6800079, 9.2471361, -17.8979263, 17.8997536
16: -16.6961231, 2.5348854, -16.7090607, 2.5430894, -14.7708511, 14.7679100
17: 6.2378883, 30.6307068, 6.2278018, 30.6410847, -17.1511688, 17.1837692
18: -14.3517656, 5.1088686, -14.3659477, 5.1138697, -14.3584270, 14.3614826
19: -20.2548981, -4.3367004, -20.2618542, -4.3308263, -14.4996338, 14.5014000
20: -2.4026103, 11.2047396, -2.4083471, 11.2076569, -12.5832253, 12.5845642
21: -11.0508146, 3.2439313, -11.0550661, 3.2487783, -14.2995930, 14.2989979
22: -3.6775136, 13.0663471, -3.6871765, 13.0865974, -14.8732529, 14.8907623
23: -14.5472593, 0.3024352, -14.5748758, 0.3187644, -14.2627487, 14.2674446
24: -19.9257126, -5.1245732, -19.9275341, -5.1186395, -9.2541008, 9.2432480
25: -5.4332457, 10.8319368, -5.4367199, 10.8451948, -13.7493439, 13.7621841
26: -20.9789009, 1.1566803, -20.9861641, 1.1800497, -19.2477951, 19.2664871
27: -15.9908314, 2.1592183, -16.0027027, 2.1567240, -13.1657104, 13.1894722
28: -12.7635231, 4.5979719, -12.7896729, 4.6123700, -17.3758926, 17.3876457
29: -5.5540586, 11.8395452, -5.5783315, 11.8617764, -14.8611679, 14.8947563
30: -10.0357389, 6.1990695, -10.0378857, 6.2060728, -13.5235481, 13.5252533
31: -10.9412708, 6.9475636, -10.9596834, 6.9459815, -14.6012192, 14.6247292
32: -24.8889179, -4.6013708, -24.8984966, -4.5985775, -13.2434731, 13.2572556
33: -69.2779236, -40.1390800, -69.3171921, -40.1175003, -16.5996780, 16.5998726
34: -53.7282562, -30.9372139, -53.7586861, -30.9297485, -14.0833054, 14.1131363
35: -47.8101997, -26.0793705, -47.8275948, -26.0725155, -12.9862022, 12.9745865
36: -42.8215027, -19.2933998, -42.8293076, -19.2931042, -15.0589256, 15.0640602
37: -86.6662216, -55.5593414, -86.6746216, -55.5541496, -18.8882561, 18.8869781
38: -52.9008331, -24.3697929, -52.9339523, -24.3498917, -18.2953949, 18.2644958
39: -76.5288391, -44.6465912, -76.5554504, -44.6289444, -16.0373688, 16.0189247
40: -67.2104034, -43.5421524, -67.2442627, -43.5486298, -14.2642021, 14.2996101
41: -55.4049606, -32.9755516, -55.4274673, -32.9812660, -16.6306190, 16.6612167
42: -29.4539738, -9.8981323, -29.4559937, -9.9101467, -17.2135277, 17.2351189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5662908, upper bound: 12.4700291
time: 11.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5663000, upper bound: 12.4809688
time: 10.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.0981445, 3.6533937, -12.1145782, 3.6583402, -13.8372650, 13.8399544
1: -3.6511440, 7.3864822, -3.6571059, 7.3888531, -8.4722672, 8.4592686
2: -0.7296714, 13.4255772, -0.7359029, 13.4200172, -13.4177208, 13.4155693
3: -1.1142426, 11.2996788, -1.1206819, 11.2961426, -11.9826698, 12.0007706
4: -11.0819702, 5.4600129, -11.0930367, 5.4744482, -14.6334000, 14.6249275
5: 1.8616557, 17.7418823, 1.8532267, 17.7361603, -15.8745041, 15.8886557
6: -39.8547821, -18.2651272, -39.8853989, -18.2282162, -15.0864029, 15.1065063
7: -3.5429776, 12.2365780, -3.5591087, 12.2420282, -13.5675964, 13.5591660
8: -6.6986370, 8.5605097, -6.7024107, 8.5592337, -12.0724411, 12.0866280
9: -4.7551622, 11.6793804, -4.7722883, 11.6953449, -12.9588470, 12.9570389
10: 1.3409939, 25.7319050, 1.3383660, 25.7360134, -20.8816757, 20.8586197
11: -11.4867496, 4.2840514, -11.4898825, 4.2840376, -15.7707872, 15.7739334
12: -11.8851004, 9.8110180, -11.8820906, 9.8251057, -14.9783134, 14.9661140
13: -18.5584450, 6.6954079, -18.5557308, 6.7108278, -16.6236687, 16.5453300
14: 4.9872532, 36.3834991, 4.9785366, 36.3996964, -26.6838150, 26.6468658
15: -8.6586580, 9.2215481, -8.6916800, 9.2484598, -17.9071178, 17.9132271
16: -16.7066727, 2.5353742, -16.7156487, 2.5278459, -14.7523918, 14.7775955
17: 6.2287035, 30.6312809, 6.2184429, 30.6429214, -17.1608124, 17.1811066
18: -14.3568592, 5.1108375, -14.3707256, 5.1159167, -14.3648987, 14.3607063
19: -20.2587528, -4.3335409, -20.2649784, -4.3273330, -14.5072708, 14.5052986
20: -2.4041715, 11.2146778, -2.4108665, 11.2223148, -12.5923615, 12.5960617
21: -11.0553856, 3.2451887, -11.0574493, 3.2492647, -14.3046503, 14.3026381
22: -3.6786578, 13.0700617, -3.6820986, 13.0869350, -14.8759613, 14.8860512
23: -14.5492983, 0.3062444, -14.5729389, 0.3228238, -14.2691803, 14.2739677
24: -19.9286213, -5.1238751, -19.9311485, -5.1183209, -9.2554207, 9.2451820
25: -5.4376307, 10.8324490, -5.4413891, 10.8463335, -13.7530632, 13.7598228
26: -20.9850292, 1.1581373, -20.9924393, 1.1810100, -19.2528076, 19.2702026
27: -15.9920921, 2.1671317, -16.0027962, 2.1677897, -13.1554909, 13.2084084
28: -12.7651281, 4.6035619, -12.7882004, 4.6196890, -17.3848171, 17.3917618
29: -5.5550947, 11.8412952, -5.5781350, 11.8620005, -14.8630829, 14.8985329
30: -10.0388794, 6.1999149, -10.0389900, 6.2016792, -13.5179482, 13.5258598
31: -10.9443035, 6.9501333, -10.9564886, 6.9473596, -14.6056404, 14.6248970
32: -24.8898792, -4.5813932, -24.9034042, -4.5674515, -13.2444229, 13.2737083
33: -69.2791977, -40.1342850, -69.2927704, -40.1157684, -16.5977592, 16.5682220
34: -53.7288017, -30.9284592, -53.7416191, -30.9206276, -14.0795364, 14.1101112
35: -47.8106842, -26.0752220, -47.8141022, -26.0712318, -12.9878922, 12.9657402
36: -42.8217659, -19.2842159, -42.8157196, -19.2821312, -15.0624313, 15.0646362
37: -86.6675262, -55.5566177, -86.6675797, -55.5525208, -18.8941879, 18.8825188
38: -52.9014206, -24.3572083, -52.9212685, -24.3338947, -18.2911224, 18.2787628
39: -76.5302124, -44.6455612, -76.5411148, -44.6304321, -16.0429955, 16.0070229
40: -67.2114258, -43.5367508, -67.2308350, -43.5417557, -14.2517891, 14.3200722
41: -55.4058685, -32.9658623, -55.4160995, -32.9687996, -16.6268692, 16.6757355
42: -29.4547577, -9.8792667, -29.4604836, -9.8798666, -17.2156792, 17.2514381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673617, upper bound: 12.4665543
time: 11.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673693, upper bound: 12.4777571
time: 20.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.0965843, 3.6536205, -12.1132336, 3.6637700, -13.8421669, 13.8399200
1: -3.6479025, 7.3867826, -3.6526737, 7.3924637, -8.4838638, 8.4614487
2: -0.7283977, 13.4257889, -0.7348393, 13.4220047, -13.4197006, 13.4162025
3: -1.1142068, 11.2965603, -1.1207854, 11.2937765, -11.9844894, 12.0067616
4: -11.0782471, 5.4602847, -11.0881567, 5.4754381, -14.6334076, 14.6228027
5: 1.8616762, 17.7407684, 1.8527880, 17.7360706, -15.8743944, 15.8879805
6: -39.8548851, -18.2699757, -39.8926620, -18.2346840, -15.0858841, 15.1108017
7: -3.5423784, 12.2366066, -3.5605769, 12.2452698, -13.5771255, 13.5600662
8: -6.6968756, 8.5606775, -6.7019563, 8.5603256, -12.0741196, 12.0874500
9: -4.7571716, 11.6793489, -4.7800970, 11.7156096, -12.9728470, 12.9614258
10: 1.3367848, 25.7318077, 1.3243055, 25.7552223, -20.9013977, 20.8703308
11: -11.4884148, 4.2834382, -11.4972334, 4.2900062, -15.7784214, 15.7806721
12: -11.8852968, 9.8107214, -11.8877516, 9.8300791, -14.9821777, 14.9711113
13: -18.5587997, 6.6953316, -18.5583668, 6.7214804, -16.6276474, 16.5472755
14: 4.9821396, 36.3835449, 4.9602394, 36.4377136, -26.7193832, 26.6621170
15: -8.6559238, 9.2220411, -8.6889372, 9.2527905, -17.9087143, 17.9109783
16: -16.7083206, 2.5354474, -16.7261505, 2.5407004, -14.7712402, 14.7848434
17: 6.2277594, 30.6312904, 6.2100587, 30.6495667, -17.1587677, 17.1947670
18: -14.3530512, 5.1111493, -14.3670292, 5.1176014, -14.3672752, 14.3636379
19: -20.2574749, -4.3333158, -20.2665501, -4.3261585, -14.5070076, 14.5109596
20: -2.4043670, 11.2135296, -2.4154193, 11.2212582, -12.5920486, 12.5974808
21: -11.0535927, 3.2451215, -11.0594778, 3.2498894, -14.3034821, 14.3045998
22: -3.6785293, 13.0705986, -3.6886523, 13.0911455, -14.8793793, 14.8981781
23: -14.5497122, 0.3062451, -14.5774174, 0.3240955, -14.2712326, 14.2765350
24: -19.9270859, -5.1239262, -19.9298344, -5.1179495, -9.2568321, 9.2467346
25: -5.4372549, 10.8324757, -5.4436474, 10.8478012, -13.7524986, 13.7672310
26: -20.9818668, 1.1582620, -20.9914551, 1.1831353, -19.2536049, 19.2739639
27: -15.9919767, 2.1681716, -16.0126820, 2.1712646, -13.1593361, 13.2098999
28: -12.7652683, 4.6043758, -12.7964745, 4.6230340, -17.3883018, 17.4008503
29: -5.5551739, 11.8409863, -5.5804052, 11.8636255, -14.8639526, 14.9058685
30: -10.0397816, 6.1996541, -10.0441666, 6.2081537, -13.5259933, 13.5305328
31: -10.9440775, 6.9507051, -10.9641905, 6.9494314, -14.6071396, 14.6315308
32: -24.8899746, -4.5878830, -24.9133339, -4.5762701, -13.2442284, 13.2836304
33: -69.2788544, -40.1298141, -69.3244171, -40.1024551, -16.6064453, 16.6016426
34: -53.7287979, -30.9253407, -53.7689896, -30.9096241, -14.0880890, 14.1366730
35: -47.8106308, -26.0734310, -47.8316345, -26.0636902, -12.9948463, 12.9853096
36: -42.8216705, -19.2828236, -42.8362427, -19.2759533, -15.0662880, 15.0811386
37: -86.6672745, -55.5550232, -86.6782150, -55.5469894, -18.8985138, 18.8898010
38: -52.9013596, -24.3552113, -52.9478569, -24.3261738, -18.2958794, 18.2940292
39: -76.5300140, -44.6432037, -76.5576324, -44.6236305, -16.0483589, 16.0232277
40: -67.2112656, -43.5338211, -67.2518768, -43.5347557, -14.2562599, 14.3185196
41: -55.4058990, -32.9633408, -55.4398880, -32.9605484, -16.6298981, 16.6878052
42: -29.4549580, -9.8873787, -29.4667110, -9.8919573, -17.2136917, 17.2554550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673617, upper bound: 12.5077196
time: 10.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5673693, upper bound: 12.5188159
time: 17.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.0972433, 3.6564655, -12.1310444, 3.6677771, -13.8449631, 13.8628311
1: -3.6473491, 7.3888650, -3.6644688, 7.3952556, -8.4728050, 8.4662266
2: -0.7297517, 13.4293804, -0.7603172, 13.4304609, -13.4224548, 13.4443512
3: -1.1138519, 11.3000269, -1.1313781, 11.3004045, -11.9850616, 12.0164757
4: -11.0812912, 5.4622049, -11.1047840, 5.4839478, -14.6335983, 14.6351204
5: 1.8617082, 17.7432823, 1.8329716, 17.7423477, -15.8806400, 15.9103107
6: -39.8543320, -18.2800522, -39.8781853, -18.2489777, -15.0979805, 15.0768394
7: -3.5424359, 12.2417650, -3.5895591, 12.2562761, -13.5833588, 13.6023560
8: -6.6974792, 8.5634823, -6.7192726, 8.5670118, -12.0651627, 12.1086922
9: -4.7454252, 11.6800194, -4.7612095, 11.6997757, -12.9530296, 12.9579926
10: 1.3521767, 25.7322788, 1.3517065, 25.7341156, -20.8665543, 20.8537598
11: -11.4843674, 4.2842584, -11.4898624, 4.2858515, -15.7702188, 15.7741203
12: -11.8851881, 9.8111305, -11.8873472, 9.8512516, -15.0106239, 14.9766388
13: -18.5440922, 6.6938429, -18.5361958, 6.7165413, -16.5671425, 16.5734100
14: 4.9998617, 36.3829231, 4.9882336, 36.3978310, -26.6557922, 26.6578140
15: -8.6502476, 9.2200003, -8.6804628, 9.2472162, -17.8974648, 17.9004631
16: -16.6960526, 2.5423503, -16.7216301, 2.5471117, -14.7665291, 14.7716827
17: 6.2300758, 30.6312313, 6.2129641, 30.6551914, -17.1773033, 17.1941376
18: -14.3566933, 5.1117969, -14.3802071, 5.1213770, -14.3651447, 14.3653069
19: -20.2576504, -4.3369746, -20.2685184, -4.3329763, -14.5090675, 14.5064964
20: -2.4040248, 11.2055073, -2.4099901, 11.2104454, -12.5887566, 12.5895538
21: -11.0572548, 3.2442665, -11.0675192, 3.2619977, -14.3192520, 14.3117857
22: -3.6853890, 13.0662060, -3.7003026, 13.1152143, -14.9132957, 14.8941498
23: -14.5493565, 0.3024068, -14.5788956, 0.3182328, -14.2726898, 14.2686920
24: -19.9270382, -5.1243310, -19.9317818, -5.1163507, -9.2623825, 9.2482529
25: -5.4414253, 10.8319540, -5.4540377, 10.8687572, -13.7834435, 13.7682953
26: -20.9925747, 1.1573093, -21.0120487, 1.2217033, -19.2993126, 19.2801514
27: -15.9916086, 2.1607904, -16.0087681, 2.1600952, -13.1727409, 13.1870308
28: -12.7659922, 4.5974545, -12.7904320, 4.6115413, -17.3775330, 17.3878860
29: -5.5613170, 11.8401375, -5.5943294, 11.8951416, -14.9032440, 14.9026489
30: -10.0378275, 6.1995716, -10.0416861, 6.2155361, -13.5346642, 13.5288849
31: -10.9429684, 6.9476814, -10.9646101, 6.9456758, -14.6045647, 14.6248016
32: -24.8892174, -4.5945668, -24.8945045, -4.5854278, -13.2509079, 13.2486725
33: -69.2786255, -40.1409073, -69.2943268, -40.1213837, -16.5999603, 16.5691376
34: -53.7286682, -30.9360485, -53.7360001, -30.9286919, -14.0870247, 14.0868378
35: -47.8029137, -26.0802059, -47.8002739, -26.0826893, -12.9845772, 12.9568634
36: -42.8133316, -19.2941017, -42.7974548, -19.2981453, -15.0572052, 15.0485001
37: -86.6631775, -55.5595207, -86.6632080, -55.5558662, -18.9022217, 18.8913231
38: -52.9020691, -24.3701096, -52.9128342, -24.3525219, -18.2974281, 18.2561874
39: -76.5225372, -44.6482735, -76.5319290, -44.6344872, -16.0412750, 16.0076141
40: -67.2107620, -43.5348053, -67.2585907, -43.5340729, -14.2712517, 14.2878494
41: -55.4053764, -32.9722023, -55.4175224, -32.9740486, -16.6455994, 16.6392593
42: -29.4539642, -9.8984833, -29.4473267, -9.9095879, -17.2359619, 17.2144394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671240, upper bound: 12.4620934
time: 21.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671317, upper bound: 12.4736059
time: 8.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.0957088, 3.6567049, -12.1297503, 3.6732087, -13.8498650, 13.8627853
1: -3.6441226, 7.3891506, -3.6600647, 7.3988600, -8.4843941, 8.4684048
2: -0.7284732, 13.4296036, -0.7592492, 13.4324312, -13.4244118, 13.4449768
3: -1.1138228, 11.2969046, -1.1314950, 11.2980270, -11.9869041, 12.0224361
4: -11.0775661, 5.4624734, -11.0998755, 5.4849463, -14.6335678, 14.6329536
5: 1.8617311, 17.7421665, 1.8325357, 17.7422829, -15.8805523, 15.9096308
6: -39.8544884, -18.2849007, -39.8854599, -18.2554455, -15.0974617, 15.0811157
7: -3.5418587, 12.2417994, -3.5910625, 12.2595396, -13.5928726, 13.6032104
8: -6.6957045, 8.5636320, -6.7188311, 8.5680637, -12.0668182, 12.1095390
9: -4.7474446, 11.6799679, -4.7690163, 11.7200508, -12.9670105, 12.9623909
10: 1.3479409, 25.7321701, 1.3376670, 25.7533073, -20.8862686, 20.8654327
11: -11.4860191, 4.2836289, -11.4972134, 4.2917843, -15.7778034, 15.7808418
12: -11.8853903, 9.8108311, -11.8930368, 9.8562403, -15.0145149, 14.9816437
13: -18.5444756, 6.6937456, -18.5388451, 6.7271595, -16.5710754, 16.5753555
14: 4.9947195, 36.3829231, 4.9698763, 36.4359207, -26.6913757, 26.6730270
15: -8.6475029, 9.2204895, -8.6777010, 9.2516375, -17.8991394, 17.8981895
16: -16.6977386, 2.5424595, -16.7321548, 2.5599730, -14.7853622, 14.7789612
17: 6.2291703, 30.6312466, 6.2045674, 30.6618118, -17.1752701, 17.2077713
18: -14.3529015, 5.1121058, -14.3765039, 5.1230445, -14.3675098, 14.3682308
19: -20.2563782, -4.3367414, -20.2700539, -4.3318014, -14.5088234, 14.5121422
20: -2.4042387, 11.2043571, -2.4145334, 11.2093849, -12.5884590, 12.5909615
21: -11.0554810, 3.2441945, -11.0695057, 3.2626348, -14.3181152, 14.3136997
22: -3.6852243, 13.0667200, -3.7068572, 13.1194286, -14.9167976, 14.9062386
23: -14.5497370, 0.3024292, -14.5833702, 0.3195033, -14.2747536, 14.2712097
24: -19.9254818, -5.1243849, -19.9304790, -5.1159935, -9.2638092, 9.2498093
25: -5.4410710, 10.8319788, -5.4563160, 10.8702126, -13.7828712, 13.7756500
26: -20.9894142, 1.1574242, -21.0110321, 1.2238336, -19.3001709, 19.2839203
27: -15.9914646, 2.1618290, -16.0186462, 2.1635966, -13.1766167, 13.1885109
28: -12.7661400, 4.5982590, -12.7987595, 4.6149135, -17.3810539, 17.3970184
29: -5.5613885, 11.8398046, -5.5966320, 11.8967562, -14.9041176, 14.9099541
30: -10.0387383, 6.1992912, -10.0468683, 6.2220192, -13.5426865, 13.5335274
31: -10.9427490, 6.9482532, -10.9723434, 6.9477267, -14.6060486, 14.6314392
32: -24.8893166, -4.6010509, -24.9044685, -4.5942345, -13.2506981, 13.2585869
33: -69.2782898, -40.1364899, -69.3259583, -40.1081123, -16.6086426, 16.6025543
34: -53.7286606, -30.9329109, -53.7633972, -30.9177055, -14.0955925, 14.1133881
35: -47.8027916, -26.0783997, -47.8178406, -26.0751686, -12.9915276, 12.9764290
36: -42.8131676, -19.2926979, -42.8179321, -19.2919846, -15.0610771, 15.0650101
37: -86.6629181, -55.5579109, -86.6738892, -55.5503616, -18.9065170, 18.8985901
38: -52.9020462, -24.3681641, -52.9393806, -24.3447609, -18.3022156, 18.2714386
39: -76.5223465, -44.6459198, -76.5484161, -44.6277161, -16.0466156, 16.0237846
40: -67.2106018, -43.5319290, -67.2796555, -43.5270309, -14.2757187, 14.2863007
41: -55.4054260, -32.9696350, -55.4413757, -32.9658279, -16.6486397, 16.6513252
42: -29.4541969, -9.9066191, -29.4535370, -9.9217186, -17.2339859, 17.2184639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671240, upper bound: 12.5038161
time: 8.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5671317, upper bound: 12.5152519
time: 7.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.0987492, 3.6570101, -12.1334639, 3.6672153, -13.8454514, 13.8660698
1: -3.6513779, 7.3899174, -3.6697245, 7.3976941, -8.4787025, 8.4709625
2: -0.7304592, 13.4308453, -0.7611672, 13.4329596, -13.4291573, 13.4463844
3: -1.1145488, 11.3051920, -1.1335095, 11.3091965, -11.9918976, 12.0246506
4: -11.0824585, 5.4638147, -11.1067562, 5.4849768, -14.6431732, 14.6424065
5: 1.8610582, 17.7468224, 1.8340516, 17.7481613, -15.8871031, 15.9127712
6: -39.8551331, -18.2675171, -39.8901062, -18.2279758, -15.0933266, 15.1011925
7: -3.5433986, 12.2438107, -3.5916035, 12.2599621, -13.5814819, 13.6076164
8: -6.6994023, 8.5643806, -6.7218418, 8.5687284, -12.0780640, 12.1114807
9: -4.7542815, 11.6804619, -4.7750516, 11.7030163, -12.9635620, 12.9642944
10: 1.3361287, 25.7330055, 1.3244166, 25.7443924, -20.8930817, 20.8702850
11: -11.4887772, 4.2843900, -11.4966030, 4.2856922, -15.7744694, 15.7809925
12: -11.8923025, 9.8118763, -11.8990221, 9.8558998, -15.0237617, 14.9821358
13: -18.5507469, 6.6955986, -18.5462399, 6.7217283, -16.5904846, 16.5690842
14: 4.9789896, 36.3836060, 4.9524164, 36.4130821, -26.6950150, 26.6728516
15: -8.6553841, 9.2223110, -8.6893749, 9.2528305, -17.9082146, 17.9116859
16: -16.7082672, 2.5429435, -16.7387028, 2.5447102, -14.7669411, 14.7885857
17: 6.2199860, 30.6318130, 6.1952186, 30.6636696, -17.1849136, 17.2051315
18: -14.3579865, 5.1140709, -14.3812790, 5.1250973, -14.3739815, 14.3674717
19: -20.2602234, -4.3335824, -20.2732487, -4.3283091, -14.5164719, 14.5160370
20: -2.4057913, 11.2143030, -2.4170737, 11.2240305, -12.5975723, 12.6024704
21: -11.0600491, 3.2454457, -11.0719261, 3.2631140, -14.3231630, 14.3173714
22: -3.6863852, 13.0704117, -3.7017732, 13.1197586, -14.9194603, 14.9015427
23: -14.5517988, 0.3062334, -14.5814037, 0.3235679, -14.2811890, 14.2777786
24: -19.9283810, -5.1237001, -19.9340782, -5.1157098, -9.2651329, 9.2517242
25: -5.4454460, 10.8325081, -5.4610152, 10.8713551, -13.7865677, 13.7733078
26: -20.9955139, 1.1588771, -21.0173035, 1.2247651, -19.3051453, 19.2876205
27: -15.9927588, 2.1697445, -16.0187759, 2.1746926, -13.1663857, 13.2074432
28: -12.7677345, 4.6038427, -12.7972765, 4.6222239, -17.3899574, 17.4011192
29: -5.5623813, 11.8415852, -5.5964165, 11.8969831, -14.9060440, 14.9137154
30: -10.0418615, 6.2001414, -10.0479698, 6.2176294, -13.5371017, 13.5341568
31: -10.9457760, 6.9508009, -10.9691372, 6.9491034, -14.6105042, 14.6316109
32: -24.8902664, -4.5810933, -24.9093838, -4.5631351, -13.2516594, 13.2750244
33: -69.2795639, -40.1316681, -69.3015900, -40.1063538, -16.6067390, 16.5709381
34: -53.7291870, -30.9242020, -53.7463570, -30.9085846, -14.0918045, 14.1103554
35: -47.8033104, -26.0742626, -47.8043289, -26.0738831, -12.9932289, 12.9676094
36: -42.8134766, -19.2835426, -42.8042908, -19.2810097, -15.0645523, 15.0656052
37: -86.6641769, -55.5551682, -86.6667862, -55.5486984, -18.9124374, 18.8941269
38: -52.9025764, -24.3555546, -52.9267044, -24.3288021, -18.2979431, 18.2857513
39: -76.5237045, -44.6448822, -76.5340805, -44.6291656, -16.0522614, 16.0118904
40: -67.2116394, -43.5265083, -67.2662125, -43.5201645, -14.2633286, 14.3067856
41: -55.4063339, -32.9599648, -55.4299774, -32.9533272, -16.6449089, 16.6658669
42: -29.4549389, -9.8877325, -29.4580383, -9.8914165, -17.2361221, 17.2347984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681592, upper bound: 12.4982333
time: 13.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5097549
time: 23.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.0972023, 3.6572468, -12.1321449, 3.6726184, -13.8503609, 13.8660469
1: -3.6481254, 7.3901987, -3.6653187, 7.4013104, -8.4902916, 8.4731293
2: -0.7291802, 13.4310608, -0.7601156, 13.4349365, -13.4311218, 13.4470291
3: -1.1145182, 11.3020878, -1.1336054, 11.3068123, -11.9937057, 12.0306396
4: -11.0787249, 5.4640770, -11.1018658, 5.4859982, -14.6431732, 14.6402359
5: 1.8610849, 17.7457275, 1.8336205, 17.7481117, -15.8870268, 15.9121075
6: -39.8552704, -18.2723713, -39.8973694, -18.2343903, -15.0928268, 15.1055031
7: -3.5428357, 12.2438612, -3.5930920, 12.2632008, -13.5910187, 13.6084747
8: -6.6976275, 8.5645390, -6.7213726, 8.5698128, -12.0797348, 12.1122856
9: -4.7562752, 11.6804333, -4.7828298, 11.7232933, -12.9775658, 12.9686890
10: 1.3318920, 25.7329082, 1.3103690, 25.7635593, -20.9128113, 20.8819580
11: -11.4904346, 4.2837548, -11.5039406, 4.2916260, -15.7820606, 15.7876949
12: -11.8925085, 9.8115826, -11.9046965, 9.8608770, -15.0276527, 14.9871140
13: -18.5511017, 6.6954994, -18.5488739, 6.7323813, -16.5944061, 16.5710144
14: 4.9738235, 36.3837013, 4.9340544, 36.4510727, -26.7305908, 26.6880798
15: -8.6526480, 9.2228088, -8.6865902, 9.2572346, -17.9098816, 17.9093990
16: -16.7099075, 2.5430071, -16.7492085, 2.5575688, -14.7857742, 14.7958679
17: 6.2190800, 30.6318436, 6.1867995, 30.6703415, -17.1828766, 17.2187729
18: -14.3541813, 5.1143808, -14.3775597, 5.1267672, -14.3763447, 14.3703918
19: -20.2589588, -4.3333349, -20.2748127, -4.3271494, -14.5162048, 14.5216904
20: -2.4059823, 11.2131615, -2.4216423, 11.2229757, -12.5972786, 12.6038818
21: -11.0582685, 3.2453940, -11.0739326, 3.2637672, -14.3220358, 14.3193264
22: -3.6862426, 13.0709181, -3.7083561, 13.1240015, -14.9229431, 14.9136772
23: -14.5522022, 0.3062544, -14.5858994, 0.3248420, -14.2832642, 14.2802963
24: -19.9268436, -5.1237316, -19.9327793, -5.1153259, -9.2665558, 9.2532921
25: -5.4450750, 10.8325462, -5.4632759, 10.8728218, -13.7859688, 13.7807007
26: -20.9923954, 1.1589718, -21.0163193, 1.2268989, -19.3060036, 19.2913971
27: -15.9925985, 2.1707916, -16.0286522, 2.1781950, -13.1702461, 13.2089539
28: -12.7679205, 4.6046839, -12.8056087, 4.6255813, -17.3935013, 17.4102936
29: -5.5624447, 11.8412895, -5.5987053, 11.8986053, -14.9069023, 14.9210396
30: -10.0428085, 6.1998453, -10.0531483, 6.2240963, -13.5451431, 13.5388184
31: -10.9455481, 6.9513874, -10.9768648, 6.9511919, -14.6119728, 14.6382332
32: -24.8903465, -4.5876064, -24.9193211, -4.5719204, -13.2514496, 13.2849312
33: -69.2792816, -40.1272507, -69.3331757, -40.0930748, -16.6154099, 16.6043549
34: -53.7291832, -30.9211140, -53.7737694, -30.8976173, -14.1003761, 14.1369286
35: -47.8032341, -26.0725040, -47.8218765, -26.0663414, -13.0001869, 12.9871597
36: -42.8133926, -19.2821503, -42.8247681, -19.2748127, -15.0684280, 15.0820923
37: -86.6639328, -55.5535355, -86.6774216, -55.5432053, -18.9167557, 18.9014320
38: -52.9025269, -24.3535881, -52.9532242, -24.3210373, -18.3026733, 18.3010101
39: -76.5235138, -44.6425743, -76.5505753, -44.6223488, -16.0576096, 16.0280914
40: -67.2114716, -43.5236282, -67.2872772, -43.5131416, -14.2677956, 14.3052521
41: -55.4063644, -32.9574051, -55.4538307, -32.9451027, -16.6479263, 16.6779213
42: -29.4551506, -9.8958397, -29.4642639, -9.9035378, -17.2341309, 17.2388115

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681592, upper bound: 12.5399798
time: 17.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5515008
time: 6.46 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.1196642, 3.6756101, -12.1174870, 3.6720448, -13.8733826, 13.8616562
1: -3.6636953, 7.3956947, -3.6615813, 7.3897400, -8.4764137, 8.4706497
2: -0.7416086, 13.4308338, -0.7412485, 13.4214983, -13.4293213, 13.4307098
3: -1.1276591, 11.3047218, -1.1267018, 11.2922993, -12.0035477, 12.0001144
4: -11.1108732, 5.4828801, -11.1082678, 5.4777699, -14.6483994, 14.6563339
5: 1.8449554, 17.7432556, 1.8424273, 17.7326870, -15.8877316, 15.9008284
6: -39.9280090, -18.2356873, -39.9160919, -18.2488804, -15.0957184, 15.1673050
7: -3.5767999, 12.2511578, -3.5757141, 12.2409039, -13.5834274, 13.5824280
8: -6.7016087, 8.5691147, -6.7008429, 8.5626326, -12.0719681, 12.0940247
9: -4.7789364, 11.7154951, -4.7647638, 11.7121735, -12.9999695, 12.9680099
10: 1.3283100, 25.7410927, 1.3504639, 25.7305946, -20.8900681, 20.8927917
11: -11.4985905, 4.2861629, -11.4901333, 4.2858176, -15.7844086, 15.7762966
12: -11.8971577, 9.8274460, -11.8804302, 9.8224421, -14.9771080, 14.9883690
13: -18.5563316, 6.7229609, -18.5470181, 6.7202425, -16.6112289, 16.5769005
14: 4.9752941, 36.4193382, 5.0107365, 36.4047775, -26.6981354, 26.6459503
15: -8.6896172, 9.2866659, -8.6853676, 9.2811842, -17.9708023, 17.9720345
16: -16.7210159, 2.5347810, -16.7111797, 2.5314002, -14.7888680, 14.7728882
17: 6.2138233, 30.6566391, 6.2340965, 30.6481533, -17.1931305, 17.1970177
18: -14.3928185, 5.1245680, -14.3907270, 5.1172571, -14.3975945, 14.3981247
19: -20.2750587, -4.3251119, -20.2683334, -4.3249750, -14.5285187, 14.5158844
20: -2.4219022, 11.2177467, -2.4142010, 11.2116985, -12.5980339, 12.6015739
21: -11.0728989, 3.2493734, -11.0640640, 3.2490308, -14.3219299, 14.3134375
22: -3.6935956, 13.1067123, -3.6836102, 13.1054821, -14.9132690, 14.9122620
23: -14.5789127, 0.3452520, -14.5753345, 0.3424714, -14.3147240, 14.2895851
24: -19.9322815, -5.1146059, -19.9306183, -5.1139569, -9.2556267, 9.2595673
25: -5.4464726, 10.8593483, -5.4372740, 10.8582563, -13.7775154, 13.7778168
26: -21.0071754, 1.2098122, -20.9902496, 1.2085202, -19.2938843, 19.2805710
27: -16.0082111, 2.1699300, -15.9984865, 2.1595988, -13.1966400, 13.2036591
28: -12.7948513, 4.6387062, -12.7858076, 4.6334696, -17.4283218, 17.4245148
29: -5.5892563, 11.8881798, -5.5784230, 11.8877888, -14.9223022, 14.9184608
30: -10.0439758, 6.2060366, -10.0367470, 6.2042227, -13.5352516, 13.5344505
31: -10.9732361, 6.9484234, -10.9681587, 6.9446034, -14.6317062, 14.6339264
32: -24.9228592, -4.5775423, -24.9076691, -4.5891266, -13.2446060, 13.2792702
33: -69.3115387, -40.1101875, -69.3047791, -40.1250763, -16.6029205, 16.6175537
34: -53.7621384, -30.9181652, -53.7515297, -30.9376259, -14.0709991, 14.1276703
35: -47.8186646, -26.0703201, -47.8146820, -26.0775928, -12.9772377, 12.9718552
36: -42.8213043, -19.2894020, -42.8128662, -19.2977295, -15.0641937, 15.0579453
37: -86.6747665, -55.5490952, -86.6691742, -55.5534821, -18.8907585, 18.9071350
38: -52.9466934, -24.3391247, -52.9328003, -24.3551884, -18.3047562, 18.3075409
39: -76.5574417, -44.6270523, -76.5542755, -44.6322250, -16.0641174, 16.0451317
40: -67.2499390, -43.5375671, -67.2423706, -43.5552406, -14.2981262, 14.3352013
41: -55.4282646, -32.9710312, -55.4155960, -32.9885445, -16.6364670, 16.6675186
42: -29.4675694, -9.8875628, -29.4565277, -9.8969688, -17.2303581, 17.2425194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4449258
time: 28.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4558591
time: 8.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1181269, 3.6758425, -12.1161757, 3.6774774, -13.8782845, 13.8616142
1: -3.6604488, 7.3959675, -3.6571734, 7.3933544, -8.4880180, 8.4728222
2: -0.7403238, 13.4310541, -0.7401640, 13.4234629, -13.4313164, 13.4313240
3: -1.1276340, 11.3016176, -1.1267692, 11.2899323, -12.0053711, 12.0060902
4: -11.1071491, 5.4831524, -11.1033726, 5.4787741, -14.6484070, 14.6541672
5: 1.8449430, 17.7421532, 1.8419623, 17.7326279, -15.8876848, 15.9001904
6: -39.9281616, -18.2405510, -39.9233894, -18.2553596, -15.0951996, 15.1715889
7: -3.5762019, 12.2511940, -3.5772097, 12.2441616, -13.5929642, 13.5833168
8: -6.6998343, 8.5692978, -6.7003660, 8.5636921, -12.0736504, 12.0948524
9: -4.7809587, 11.7154493, -4.7725410, 11.7324419, -13.0139580, 12.9724083
10: 1.3240972, 25.7409725, 1.3364215, 25.7498131, -20.9098053, 20.9044800
11: -11.5002499, 4.2855439, -11.4974556, 4.2917666, -15.7920170, 15.7829990
12: -11.8973389, 9.8271360, -11.8861446, 9.8274212, -14.9809799, 14.9933395
13: -18.5566845, 6.7228823, -18.5496864, 6.7309184, -16.6151733, 16.5788345
14: 4.9701233, 36.4194031, 4.9923649, 36.4428139, -26.7337112, 26.6611710
15: -8.6868963, 9.2871752, -8.6826172, 9.2854862, -17.9723816, 17.9697914
16: -16.7226601, 2.5348547, -16.7217083, 2.5442863, -14.8077202, 14.7801132
17: 6.2128944, 30.6566620, 6.2257190, 30.6548119, -17.1911011, 17.2106857
18: -14.3890190, 5.1248474, -14.3870325, 5.1189237, -14.3999748, 14.4010658
19: -20.2737560, -4.3248882, -20.2699261, -4.3238239, -14.5282364, 14.5215302
20: -2.4221115, 11.2166233, -2.4187522, 11.2106342, -12.5977402, 12.6030045
21: -11.0711174, 3.2492990, -11.0661173, 3.2496574, -14.3207750, 14.3154163
22: -3.6934650, 13.1072254, -3.6901579, 13.1096954, -14.9167175, 14.9244041
23: -14.5792885, 0.3452873, -14.5798368, 0.3437634, -14.3168335, 14.2921410
24: -19.9307365, -5.1146507, -19.9292736, -5.1135902, -9.2570457, 9.2611160
25: -5.4460917, 10.8593655, -5.4395819, 10.8597355, -13.7769012, 13.7851830
26: -21.0040150, 1.2099261, -20.9892426, 1.2106249, -19.2947159, 19.2843399
27: -16.0080795, 2.1709938, -16.0083714, 2.1631193, -13.2004700, 13.2051582
28: -12.7949963, 4.6395397, -12.7941151, 4.6368275, -17.4318237, 17.4336548
29: -5.5893197, 11.8878736, -5.5807147, 11.8894024, -14.9231873, 14.9257774
30: -10.0448771, 6.2057419, -10.0419540, 6.2106919, -13.5432816, 13.5390968
31: -10.9730148, 6.9490099, -10.9758663, 6.9466619, -14.6331902, 14.6405296
32: -24.9229546, -4.5840473, -24.9176445, -4.5979433, -13.2443962, 13.2891731
33: -69.3112564, -40.1057739, -69.3363800, -40.1117783, -16.6116066, 16.6509743
34: -53.7621346, -30.9150620, -53.7789497, -30.9266357, -14.0795479, 14.1542244
35: -47.8185883, -26.0685310, -47.8322678, -26.0700474, -12.9841843, 12.9914169
36: -42.8211670, -19.2879658, -42.8332901, -19.2915688, -15.0680428, 15.0744476
37: -86.6745148, -55.5474854, -86.6798630, -55.5480042, -18.8950958, 18.9143753
38: -52.9466438, -24.3371773, -52.9593773, -24.3474426, -18.3095474, 18.3228302
39: -76.5572968, -44.6247406, -76.5707397, -44.6254463, -16.0694809, 16.0613136
40: -67.2497864, -43.5346603, -67.2633972, -43.5482101, -14.3026047, 14.3336639
41: -55.4283066, -32.9684601, -55.4393997, -32.9803352, -16.6395264, 16.6795654
42: -29.4677773, -9.8956566, -29.4627399, -9.9090872, -17.2283440, 17.2464676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4856821
time: 6.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4966142
time: 9.54 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.1211805, 3.6761317, -12.1198959, 3.6714387, -13.8738594, 13.8649178
1: -3.6677351, 7.3967409, -3.6668451, 7.3921819, -8.4823227, 8.4753780
2: -0.7423103, 13.4322920, -0.7421099, 13.4240265, -13.4360275, 13.4327888
3: -1.1283643, 11.3098879, -1.1288037, 11.3010826, -12.0103760, 12.0083103
4: -11.1120453, 5.4844751, -11.1102285, 5.4787865, -14.6579590, 14.6636162
5: 1.8442812, 17.7468147, 1.8434863, 17.7385197, -15.8942385, 15.9033279
6: -39.9288139, -18.2232113, -39.9280014, -18.2278538, -15.0910759, 15.1916885
7: -3.5777860, 12.2532473, -3.5777550, 12.2445831, -13.5815887, 13.5876694
8: -6.7035275, 8.5700207, -6.7033653, 8.5643425, -12.0848885, 12.0968094
9: -4.7877512, 11.7159681, -4.7785702, 11.7153959, -13.0105247, 12.9743118
10: 1.3122458, 25.7418365, 1.3231502, 25.7408524, -20.9166107, 20.9092941
11: -11.5029774, 4.2862654, -11.4968557, 4.2856674, -15.7886448, 15.7831211
12: -11.9042969, 9.8281651, -11.8921022, 9.8270645, -14.9902229, 14.9938583
13: -18.5629749, 6.7247181, -18.5570736, 6.7254825, -16.6345291, 16.5726280
14: 4.9543943, 36.4201355, 4.9749403, 36.4200096, -26.7373657, 26.6609879
15: -8.6947403, 9.2889795, -8.6942616, 9.2868404, -17.9815807, 17.9832420
16: -16.7331867, 2.5353220, -16.7282925, 2.5289931, -14.7893295, 14.7898445
17: 6.2037301, 30.6572552, 6.2163601, 30.6566772, -17.2007637, 17.2080383
18: -14.3941116, 5.1267877, -14.3917809, 5.1209831, -14.4064407, 14.4002876
19: -20.2776241, -4.3217134, -20.2730503, -4.3203225, -14.5359077, 14.5254173
20: -2.4236495, 11.2265530, -2.4212949, 11.2252922, -12.6068764, 12.6144981
21: -11.0756989, 3.2505612, -11.0684719, 3.2501545, -14.3258533, 14.3190327
22: -3.6946239, 13.1109724, -3.6850686, 13.1100216, -14.9194336, 14.9196968
23: -14.5813360, 0.3490875, -14.5778484, 0.3478074, -14.3232307, 14.2986565
24: -19.9336262, -5.1139770, -19.9329147, -5.1132832, -9.2583618, 9.2630272
25: -5.4504986, 10.8598995, -5.4442344, 10.8608627, -13.7806129, 13.7828636
26: -21.0101452, 1.2113862, -20.9955444, 1.2115703, -19.2997437, 19.2880859
27: -16.0093784, 2.1789160, -16.0084991, 2.1741891, -13.1902695, 13.2241020
28: -12.7965889, 4.6451297, -12.7926483, 4.6441569, -17.4407463, 17.4377785
29: -5.5903597, 11.8896151, -5.5805149, 11.8896465, -14.9251099, 14.9295387
30: -10.0480404, 6.2065954, -10.0430374, 6.2063017, -13.5376740, 13.5397377
31: -10.9760427, 6.9515562, -10.9727287, 6.9480462, -14.6376038, 14.6407204
32: -24.9238739, -4.5640535, -24.9225407, -4.5668259, -13.2453499, 13.3056412
33: -69.3125458, -40.1009598, -69.3120117, -40.1100578, -16.6096764, 16.6193466
34: -53.7626495, -30.9063568, -53.7618675, -30.9175243, -14.0758018, 14.1512070
35: -47.8191071, -26.0643959, -47.8187485, -26.0687675, -12.9858627, 12.9826088
36: -42.8214912, -19.2788277, -42.8197174, -19.2806072, -15.0715523, 15.0750160
37: -86.6757736, -55.5447426, -86.6726990, -55.5463181, -18.9010315, 18.9099464
38: -52.9471970, -24.3245487, -52.9466476, -24.3314934, -18.3052521, 18.3370895
39: -76.5585938, -44.6236801, -76.5564575, -44.6268845, -16.0750999, 16.0494232
40: -67.2508392, -43.5292473, -67.2499924, -43.5413589, -14.2902184, 14.3541145
41: -55.4291992, -32.9588051, -55.4280663, -32.9678688, -16.6357689, 16.6940918
42: -29.4685287, -9.8768072, -29.4672356, -9.8787899, -17.2305374, 17.2628555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5674973, upper bound: 12.4826541
time: 10.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675011, upper bound: 12.4937356
time: 36.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.1196194, 3.6763713, -12.1185875, 3.6768990, -13.8787689, 13.8648720
1: -3.6644707, 7.3970375, -3.6624427, 7.3957777, -8.4939156, 8.4775562
2: -0.7410231, 13.4325027, -0.7410459, 13.4259815, -13.4380074, 13.4334068
3: -1.1283379, 11.3067837, -1.1289012, 11.2987022, -12.0121841, 12.0143108
4: -11.1083183, 5.4847641, -11.1053333, 5.4797978, -14.6579819, 14.6614494
5: 1.8443141, 17.7457123, 1.8430548, 17.7384567, -15.8941422, 15.9026575
6: -39.9289665, -18.2280350, -39.9352837, -18.2343216, -15.0905685, 15.1959648
7: -3.5772002, 12.2532978, -3.5792670, 12.2478228, -13.5911179, 13.5885506
8: -6.7017465, 8.5701895, -6.7029076, 8.5654192, -12.0865784, 12.0976486
9: -4.7897906, 11.7159252, -4.7863936, 11.7356853, -13.0245476, 12.9787064
10: 1.3079910, 25.7417221, 1.3091393, 25.7600746, -20.9363480, 20.9209900
11: -11.5046616, 4.2856584, -11.5041838, 4.2916079, -15.7962694, 15.7898426
12: -11.9044933, 9.8279028, -11.8978024, 9.8320484, -14.9941025, 14.9988403
13: -18.5633736, 6.7246566, -18.5597286, 6.7360845, -16.6384888, 16.5745850
14: 4.9492922, 36.4202042, 4.9565830, 36.4580345, -26.7729645, 26.6762314
15: -8.6920147, 9.2894917, -8.6915245, 9.2911129, -17.9831276, 17.9810162
16: -16.7348404, 2.5354328, -16.7388210, 2.5418744, -14.8081665, 14.7970772
17: 6.2028074, 30.6572762, 6.2079592, 30.6633110, -17.1987152, 17.2216988
18: -14.3902979, 5.1270924, -14.3880959, 5.1226525, -14.4088097, 14.4032192
19: -20.2763367, -4.3214960, -20.2746239, -4.3191643, -14.5356369, 14.5310631
20: -2.4238603, 11.2254276, -2.4258554, 11.2242146, -12.6065712, 12.6159134
21: -11.0739193, 3.2504895, -11.0705204, 3.2507865, -14.3247061, 14.3210096
22: -3.6945007, 13.1114607, -3.6916361, 13.1142492, -14.9228783, 14.9318314
23: -14.5817261, 0.3490937, -14.5823221, 0.3490911, -14.3253479, 14.3011932
24: -19.9320602, -5.1140084, -19.9315891, -5.1129265, -9.2597771, 9.2646065
25: -5.4501305, 10.8599186, -5.4465041, 10.8623543, -13.7800293, 13.7902298
26: -21.0069904, 1.2115188, -20.9945736, 1.2136989, -19.3005562, 19.2918243
27: -16.0092335, 2.1799674, -16.0183773, 2.1776719, -13.1940956, 13.2256088
28: -12.7967367, 4.6459417, -12.8009624, 4.6475058, -17.4442425, 17.4469032
29: -5.5904484, 11.8893042, -5.5827928, 11.8912640, -14.9259949, 14.9368515
30: -10.0489559, 6.2063241, -10.0482502, 6.2127519, -13.5457115, 13.5443916
31: -10.9758224, 6.9521437, -10.9804201, 6.9501190, -14.6390991, 14.6473465
32: -24.9240150, -4.5705724, -24.9324951, -4.5756130, -13.2451515, 13.3155403
33: -69.3121643, -40.0965424, -69.3436279, -40.0966949, -16.6183662, 16.6527710
34: -53.7626495, -30.9032364, -53.7892914, -30.9065666, -14.0843430, 14.1777496
35: -47.8190079, -26.0625687, -47.8363342, -26.0612564, -12.9928436, 13.0021477
36: -42.8213730, -19.2774124, -42.8401718, -19.2744446, -15.0754051, 15.0915375
37: -86.6755295, -55.5430870, -86.6834030, -55.5408249, -18.9053268, 18.9171829
38: -52.9471283, -24.3225784, -52.9731903, -24.3237305, -18.3100204, 18.3523865
39: -76.5584564, -44.6213379, -76.5729294, -44.6201172, -16.0804749, 16.0656013
40: -67.2506409, -43.5263596, -67.2710419, -43.5343132, -14.2946854, 14.3525772
41: -55.4292145, -32.9562340, -55.4518433, -32.9596252, -16.6387978, 16.7061844
42: -29.4687614, -9.8849154, -29.4734154, -9.8909092, -17.2285271, 17.2668304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5674973, upper bound: 12.5236515
time: 7.57 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5675011, upper bound: 12.5346328
time: 7.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.0852737, 3.6464355, -12.1222467, 3.6715422, -13.8414841, 13.8565331
1: -3.6531267, 7.3736386, -3.6706314, 7.3904581, -8.4683876, 8.4686031
2: -0.7137769, 13.4040318, -0.7562176, 13.4234581, -13.4078140, 13.4334755
3: -1.1163428, 11.2839127, -1.1337689, 11.2998190, -11.9979248, 12.0039558
4: -11.0942774, 5.4607110, -11.1156845, 5.4743605, -14.6358490, 14.6497307
5: 1.8637419, 17.7287369, 1.8341084, 17.7417603, -15.8716125, 15.8946285
6: -39.9169426, -18.2549267, -39.9364090, -18.2486935, -15.0691681, 15.1524239
7: -3.5595841, 12.2283182, -3.6026115, 12.2484188, -13.5499725, 13.6106186
8: -6.6706791, 8.5435953, -6.7071753, 8.5642176, -12.0776711, 12.0760727
9: -4.7559452, 11.6904955, -4.7710590, 11.7347441, -12.9880981, 12.9373436
10: 1.3388057, 25.7093430, 1.3138123, 25.7577343, -20.8942490, 20.8887329
11: -11.4920673, 4.2784686, -11.5044403, 4.2899623, -15.7820301, 15.7829094
12: -11.8736172, 9.8050098, -11.8943300, 9.8545990, -15.0090752, 14.9711609
13: -18.5468674, 6.7167959, -18.5449772, 6.7402282, -16.5444870, 16.6056900
14: 5.0306911, 36.3995514, 4.9815969, 36.4685135, -26.6898499, 26.6464844
15: -8.6749535, 9.2892361, -8.6810455, 9.2902317, -17.9651852, 17.9702816
16: -16.7030716, 2.4907808, -16.7431068, 2.5353708, -14.7876205, 14.7365913
17: 6.2800117, 30.6345444, 6.2346253, 30.6819839, -17.1591759, 17.1752739
18: -14.3686161, 5.1088457, -14.3926287, 5.1200275, -14.3822060, 14.3885841
19: -20.2464333, -4.3448224, -20.2714462, -4.3327928, -14.5026703, 14.5065155
20: -2.3900390, 11.1909409, -2.4189551, 11.2086639, -12.5655289, 12.5825958
21: -11.0404205, 3.2127805, -11.0718994, 3.2460034, -14.2864237, 14.2846794
22: -3.6558669, 13.0816746, -3.6867621, 13.1364803, -14.9298401, 14.8891068
23: -14.5644007, 0.3438234, -14.5800819, 0.3469934, -14.3086548, 14.2786407
24: -19.9185829, -5.1335683, -19.9316425, -5.1255951, -9.2337761, 9.2357712
25: -5.4259992, 10.8439493, -5.4505310, 10.8766422, -13.7749748, 13.7487984
26: -20.9541759, 1.1668899, -20.9849529, 1.2445805, -19.3160019, 19.2280960
27: -16.0028038, 2.1714363, -16.0308323, 2.1783290, -13.2004547, 13.1854858
28: -12.7778130, 4.6317768, -12.7983551, 4.6439338, -17.4217472, 17.4301319
29: -5.5491505, 11.8613234, -5.5719118, 11.9212418, -14.9449234, 14.8928909
30: -10.0272007, 6.1888199, -10.0438652, 6.2235889, -13.5435295, 13.5216141
31: -10.9458561, 6.9331198, -10.9843655, 6.9402032, -14.6050148, 14.6158752
32: -24.9220619, -4.5828309, -24.9349136, -4.5782375, -13.2432175, 13.2853165
33: -69.2764206, -40.1652832, -69.3472443, -40.1280708, -16.5605202, 16.6119156
34: -53.7483444, -30.9455299, -53.7924805, -30.9207096, -14.0812454, 14.1222954
35: -47.7910957, -26.1042137, -47.8241577, -26.0891190, -12.9477196, 12.9630508
36: -42.7737579, -19.3312302, -42.8149796, -19.3013954, -15.0272980, 15.0393219
37: -86.6454468, -55.5825691, -86.6748657, -55.5608673, -18.8788834, 18.8776054
38: -52.9011650, -24.3976002, -52.9683685, -24.3638287, -18.2212143, 18.2957840
39: -76.5205612, -44.6848869, -76.5611877, -44.6559639, -16.0299568, 16.0226364
40: -67.2339630, -43.5451431, -67.2998657, -43.5303459, -14.3231544, 14.2985191
41: -55.4242287, -32.9830666, -55.4635010, -32.9627953, -16.6482239, 16.6417656
42: -29.4653397, -9.8965178, -29.4677792, -9.9039097, -17.2325516, 17.2106819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5369670, upper bound: 12.5558341
time: 11.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5369722, upper bound: 12.5673364
time: 7.04 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.0957594, 3.6676042, -12.1240387, 3.6844015, -13.8666344, 13.8647690
1: -3.6573596, 7.3935995, -3.6709399, 7.4026537, -8.4884605, 8.4737511
2: -0.7262084, 13.4278717, -0.7574736, 13.4376116, -13.4342880, 13.4464073
3: -1.1160210, 11.3008957, -1.1345624, 11.3098030, -12.0072289, 12.0178566
4: -11.1030302, 5.4841199, -11.1163483, 5.4881525, -14.6584244, 14.6620903
5: 1.8601036, 17.7413483, 1.8328052, 17.7490692, -15.8889656, 15.9085426
6: -39.9136467, -18.2583389, -39.9382172, -18.2502155, -15.0646973, 15.1714821
7: -3.5621157, 12.2552633, -3.6032937, 12.2645531, -13.5851288, 13.6204987
8: -6.6776400, 8.5588884, -6.7079039, 8.5735397, -12.0733643, 12.0937710
9: -4.7584753, 11.7019377, -4.7722816, 11.7411871, -13.0028648, 12.9532547
10: 1.3354130, 25.7238598, 1.3117852, 25.7656784, -20.9231720, 20.9051437
11: -11.4970465, 4.2810249, -11.5077171, 4.2906671, -15.7877140, 15.7887421
12: -11.9053421, 9.8159847, -11.9132633, 9.8559227, -15.0249672, 15.0012474
13: -18.5530586, 6.7131481, -18.5490227, 6.7412996, -16.5984955, 16.5818672
14: 4.9948759, 36.3953362, 4.9607201, 36.4690170, -26.7383118, 26.6481094
15: -8.6671219, 9.2708979, -8.6773043, 9.2910633, -17.9581852, 17.9482021
16: -16.7044983, 2.5266654, -16.7444458, 2.5571904, -14.7949867, 14.7715111
17: 6.2255754, 30.6447010, 6.2024741, 30.6825943, -17.2033043, 17.2155457
18: -14.3747053, 5.1174073, -14.3960123, 5.1249371, -14.3939819, 14.3989944
19: -20.2593098, -4.3422456, -20.2788086, -4.3326063, -14.5166855, 14.5229950
20: -2.4075933, 11.1973419, -2.4294410, 11.2098694, -12.5773468, 12.5987053
21: -11.0567780, 3.2205133, -11.0813255, 3.2470810, -14.3038588, 14.3018389
22: -3.6927354, 13.0937500, -3.7092068, 13.1370058, -14.9461441, 14.9306335
23: -14.5771341, 0.3437765, -14.5870857, 0.3473082, -14.3210449, 14.2951050
24: -19.9193840, -5.1389551, -19.9316998, -5.1249385, -9.2405319, 9.2465019
25: -5.4458790, 10.8419847, -5.4620280, 10.8771753, -13.7884865, 13.7769165
26: -21.0079155, 1.1906202, -21.0169792, 1.2455595, -19.3286362, 19.2856140
27: -16.0041637, 2.1742616, -16.0320778, 2.1799965, -13.1973305, 13.2121696
28: -12.7933121, 4.6354837, -12.8067923, 4.6443291, -17.4376411, 17.4422760
29: -5.5948892, 11.8806591, -5.5991545, 11.9215918, -14.9610481, 14.9406891
30: -10.0444345, 6.1981845, -10.0542936, 6.2248354, -13.5516167, 13.5415421
31: -10.9498825, 6.9331055, -10.9863091, 6.9404364, -14.6085434, 14.6366577
32: -24.9126282, -4.5859680, -24.9361420, -4.5802832, -13.2316399, 13.3060493
33: -69.2863541, -40.1420517, -69.3478699, -40.1146240, -16.5762024, 16.6201668
34: -53.7503624, -30.9303169, -53.7933235, -30.9120483, -14.0655479, 14.1548576
35: -47.7941322, -26.1038322, -47.8252945, -26.0877914, -12.9572601, 12.9783058
36: -42.7964096, -19.3242512, -42.8279533, -19.3004780, -15.0330544, 15.0638123
37: -86.6535492, -55.5686951, -86.6788635, -55.5524673, -18.8922424, 18.9082489
38: -52.9156723, -24.3980770, -52.9767036, -24.3630486, -18.2384224, 18.3099365
39: -76.5252533, -44.6692924, -76.5618973, -44.6466904, -16.0359383, 16.0409775
40: -67.2394028, -43.5181351, -67.3012314, -43.5136719, -14.2925034, 14.3330708
41: -55.4172516, -32.9750214, -55.4641533, -32.9581375, -16.6281967, 16.6744499
42: -29.4627991, -9.8977833, -29.4684639, -9.9045191, -17.2261086, 17.2428818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5535409, upper bound: 12.5567753
time: 7.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5535439, upper bound: 12.5682675
time: 7.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.1202745, 3.6792204, -12.1363754, 3.6809158, -13.8815613, 13.8877792
1: -3.6639333, 7.3991070, -3.6742227, 7.3985806, -8.4828568, 8.4823437
2: -0.7424173, 13.4360771, -0.7665167, 13.4344358, -13.4407616, 13.4615250
3: -1.1279880, 11.3102264, -1.1395165, 11.3053408, -12.0127716, 12.0239868
4: -11.1113434, 5.4867058, -11.1219912, 5.4883070, -14.6581955, 14.6737976
5: 1.8443499, 17.7482033, 1.8232751, 17.7447090, -15.9003592, 15.9249287
6: -39.9283829, -18.2380962, -39.9207916, -18.2486305, -15.1026726, 15.1620102
7: -3.5772445, 12.2584534, -3.6082013, 12.2588062, -13.5973358, 13.6308708
8: -6.7023644, 8.5729790, -6.7202539, 8.5721149, -12.0775871, 12.1188717
9: -4.7780561, 11.7165899, -4.7674694, 11.7198267, -13.0046806, 12.9752541
10: 1.3234310, 25.7422028, 1.3365488, 25.7389297, -20.9015274, 20.9044113
11: -11.5006008, 4.2864761, -11.4968262, 4.2874556, -15.7880564, 15.7833023
12: -11.9043732, 9.8283005, -11.8973742, 9.8532352, -15.0225639, 15.0043640
13: -18.5486069, 6.7231808, -18.5375595, 6.7311683, -16.5780487, 16.6006432
14: 4.9669590, 36.4195747, 4.9846096, 36.4181519, -26.7093353, 26.6719055
15: -8.6863213, 9.2874126, -8.6830482, 9.2854633, -17.9717846, 17.9704609
16: -16.7226028, 2.5423336, -16.7342205, 2.5482397, -14.8033943, 14.7838326
17: 6.2050657, 30.6572113, 6.2109165, 30.6689072, -17.2172356, 17.2210464
18: -14.3939676, 5.1277804, -14.4012661, 5.1264133, -14.4067001, 14.4048710
19: -20.2765369, -4.3251386, -20.2766457, -4.3259745, -14.5377121, 14.5266037
20: -2.4235468, 11.2173882, -2.4204192, 11.2134094, -12.6032524, 12.6079788
21: -11.0775909, 3.2496276, -11.0785332, 3.2628829, -14.3404741, 14.3281612
22: -3.7013133, 13.1070776, -3.7032974, 13.1383200, -14.9567909, 14.9277763
23: -14.5813694, 0.3452294, -14.5838470, 0.3432317, -14.3267479, 14.2933731
24: -19.9320164, -5.1144142, -19.9335690, -5.1113214, -9.2653465, 9.2661209
25: -5.4542913, 10.8593674, -5.4568901, 10.8833122, -13.8110085, 13.7912750
26: -21.0177116, 1.2105718, -21.0151520, 1.2522886, -19.3462639, 19.2980194
27: -16.0088596, 2.1725726, -16.0144787, 2.1665154, -13.2075424, 13.2027016
28: -12.7974682, 4.6390090, -12.7949123, 4.6360159, -17.4334831, 17.4339218
29: -5.5965848, 11.8884373, -5.5966883, 11.9227524, -14.9652405, 14.9336243
30: -10.0470085, 6.2062387, -10.0457249, 6.2201920, -13.5544052, 13.5427551
31: -10.9747105, 6.9491291, -10.9807949, 6.9463468, -14.6365433, 14.6406288
32: -24.9232197, -4.5772219, -24.9136772, -4.5847740, -13.2518272, 13.2806053
33: -69.3119431, -40.1076202, -69.3136063, -40.1157074, -16.6119080, 16.6202736
34: -53.7625427, -30.9139366, -53.7562904, -30.9256020, -14.0832481, 14.1279259
35: -47.8113098, -26.0693512, -47.8049164, -26.0802231, -12.9825745, 12.9737091
36: -42.8130302, -19.2887173, -42.8014412, -19.2966080, -15.0663376, 15.0589104
37: -86.6714630, -55.5476837, -86.6684113, -55.5496902, -18.9090424, 18.9187737
38: -52.9478073, -24.3374710, -52.9382553, -24.3500214, -18.3115997, 18.3145142
39: -76.5509491, -44.6264038, -76.5472565, -44.6309624, -16.0733871, 16.0500069
40: -67.2501678, -43.5273361, -67.2777939, -43.5336571, -14.3096619, 14.3219032
41: -55.4287224, -32.9651184, -55.4295197, -32.9731331, -16.6544952, 16.6576195
42: -29.4677715, -9.8960295, -29.4540596, -9.9085531, -17.2508202, 17.2258606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5672610, upper bound: 12.4788643
time: 12.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5672643, upper bound: 12.4902819
time: 15.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.1187553, 3.6794758, -12.1350746, 3.6863551, -13.8864708, 13.8877449
1: -3.6606898, 7.3994036, -3.6698141, 7.4021864, -8.4944496, 8.4845181
2: -0.7411139, 13.4363155, -0.7654463, 13.4364128, -13.4427071, 13.4621658
3: -1.1279562, 11.3071232, -1.1396242, 11.3029661, -12.0145950, 12.0299644
4: -11.1076221, 5.4869580, -11.1170807, 5.4893188, -14.6582031, 14.6716499
5: 1.8443770, 17.7470913, 1.8228364, 17.7446423, -15.9002647, 15.9242554
6: -39.9285164, -18.2429657, -39.9280853, -18.2550812, -15.1021538, 15.1663017
7: -3.5766850, 12.2584696, -3.6097150, 12.2620745, -13.6068649, 13.6317215
8: -6.7005949, 8.5731544, -6.7197962, 8.5732059, -12.0792770, 12.1197166
9: -4.7800617, 11.7165356, -4.7752895, 11.7400951, -13.0186768, 12.9796295
10: 1.3192234, 25.7420883, 1.3225174, 25.7581635, -20.9212341, 20.9161224
11: -11.5022593, 4.2858562, -11.5041571, 4.2934012, -15.7956600, 15.7900133
12: -11.9045763, 9.8280182, -11.9030952, 9.8582115, -15.0264397, 15.0093689
13: -18.5489883, 6.7230530, -18.5401669, 6.7418141, -16.5819626, 16.6025887
14: 4.9618378, 36.4195633, 4.9662266, 36.4561462, -26.7448959, 26.6871033
15: -8.6836157, 9.2879219, -8.6803093, 9.2898598, -17.9734764, 17.9682312
16: -16.7242374, 2.5423989, -16.7447701, 2.5611293, -14.8222351, 14.7911263
17: 6.2041988, 30.6572056, 6.2025008, 30.6755791, -17.2152138, 17.2346992
18: -14.3901482, 5.1280813, -14.3975811, 5.1280870, -14.4090557, 14.4078102
19: -20.2752724, -4.3249078, -20.2781620, -4.3248148, -14.5374451, 14.5322342
20: -2.4237535, 11.2162256, -2.4249675, 11.2123337, -12.6029663, 12.6093941
21: -11.0758018, 3.2495532, -11.0805740, 3.2635331, -14.3393345, 14.3301277
22: -3.7011859, 13.1075897, -3.7098424, 13.1425190, -14.9602356, 14.9398842
23: -14.5817738, 0.3452725, -14.5883360, 0.3445139, -14.3288231, 14.2958794
24: -19.9304886, -5.1144624, -19.9322548, -5.1109657, -9.2667694, 9.2676926
25: -5.4539161, 10.8594074, -5.4591923, 10.8847752, -13.8104019, 13.7986794
26: -21.0145798, 1.2107151, -21.0141487, 1.2544067, -19.3470917, 19.3017578
27: -16.0087433, 2.1736226, -16.0243664, 2.1700027, -13.2113914, 13.2042084
28: -12.7975922, 4.6398425, -12.8032417, 4.6393681, -17.4369602, 17.4430847
29: -5.5966463, 11.8881550, -5.5989885, 11.9243793, -14.9661217, 14.9409828
30: -10.0478802, 6.2059488, -10.0509377, 6.2266364, -13.5624352, 13.5474052
31: -10.9744930, 6.9497004, -10.9885464, 6.9484291, -14.6380157, 14.6472626
32: -24.9233227, -4.5837297, -24.9236336, -4.5936117, -13.2516212, 13.2905273
33: -69.3115997, -40.1032143, -69.3452454, -40.1023560, -16.6205711, 16.6537094
34: -53.7625542, -30.9108467, -53.7837105, -30.9146404, -14.0918236, 14.1544991
35: -47.8111839, -26.0675583, -47.8224716, -26.0726986, -12.9895248, 12.9932518
36: -42.8129425, -19.2873230, -42.8218994, -19.2904263, -15.0701904, 15.0754051
37: -86.6711655, -55.5459976, -86.6790771, -55.5441666, -18.9133453, 18.9260101
38: -52.9478073, -24.3355064, -52.9647675, -24.3423424, -18.3163605, 18.3297806
39: -76.5507584, -44.6240807, -76.5637512, -44.6241226, -16.0787392, 16.0661812
40: -67.2500000, -43.5244370, -67.2988129, -43.5266037, -14.3141403, 14.3203735
41: -55.4287643, -32.9625549, -55.4533234, -32.9648781, -16.6575241, 16.6696510
42: -29.4679928, -9.9041595, -29.4602757, -9.9206676, -17.2488174, 17.2298813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5672610, upper bound: 12.5204256
time: 7.12 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5672643, upper bound: 12.5317417
time: 19.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.1097565, 3.6588340, -12.1356831, 3.6729021, -13.8618164, 13.8827667
1: -3.6604772, 7.3805137, -3.6747568, 7.3924389, -8.4802666, 8.4841061
2: -0.7293923, 13.4139137, -0.7650672, 13.4247437, -13.4229774, 13.4512825
3: -1.1289752, 11.2953291, -1.1409478, 11.3017864, -12.0120850, 12.0242672
4: -11.1000566, 5.4651752, -11.1183586, 5.4765425, -14.6451721, 14.6666069
5: 1.8473692, 17.7380600, 1.8252287, 17.7431507, -15.8913651, 15.9128313
6: -39.9326363, -18.2270107, -39.9382210, -18.2325058, -15.1020203, 15.1716232
7: -3.5751164, 12.2336521, -3.6111047, 12.2495975, -13.5698395, 13.6270638
8: -6.6955442, 8.5587311, -6.7216363, 8.5655947, -12.0964661, 12.1048012
9: -4.7863817, 11.7055626, -4.7878919, 11.7368584, -13.0144958, 12.9700546
10: 1.3064880, 25.7283058, 1.2972631, 25.7604752, -20.9188309, 20.9162445
11: -11.5017014, 4.2834220, -11.5076408, 4.2925234, -15.7942247, 15.7910633
12: -11.8799725, 9.8178015, -11.8957968, 9.8615255, -15.0236549, 14.9847832
13: -18.5494289, 6.7285233, -18.5461674, 6.7459412, -16.5512543, 16.6221390
14: 4.9768314, 36.4245071, 4.9511995, 36.4709129, -26.7356567, 26.7005997
15: -8.6966143, 9.3085566, -8.6929684, 9.2946072, -17.9912224, 18.0015259
16: -16.7350216, 2.5070987, -16.7605305, 2.5369146, -14.8152542, 14.7731628
17: 6.2485218, 30.6476860, 6.2169223, 30.6834793, -17.1786957, 17.2054443
18: -14.3853436, 5.1217794, -14.3952675, 5.1268759, -14.4061546, 14.3995438
19: -20.2649765, -4.3240762, -20.2755699, -4.3203111, -14.5308151, 14.5253372
20: -2.4079249, 11.2186403, -2.4215539, 11.2247477, -12.5999641, 12.6062088
21: -11.0622635, 3.2429771, -11.0755234, 3.2635701, -14.3258333, 14.3185005
22: -3.6653225, 13.0997629, -3.6888590, 13.1465778, -14.9501305, 14.9058037
23: -14.5714655, 0.3491383, -14.5838432, 0.3495169, -14.3249283, 14.2884903
24: -19.9310665, -5.1084199, -19.9344749, -5.1109276, -9.2627907, 9.2604370
25: -5.4380674, 10.8619099, -5.4546623, 10.8868618, -13.8000412, 13.7756042
26: -20.9638252, 1.1885498, -20.9874268, 1.2564886, -19.3402748, 19.2517090
27: -16.0084763, 2.1797414, -16.0330696, 2.1829147, -13.2081528, 13.1979752
28: -12.7838478, 4.6425400, -12.8016081, 4.6496620, -17.4335098, 17.4441490
29: -5.5519943, 11.8702526, -5.5738196, 11.9258795, -14.9527702, 14.9042130
30: -10.0347252, 6.1971731, -10.0467958, 6.2274818, -13.5567818, 13.5327454
31: -10.9732971, 6.9528403, -10.9911270, 6.9516683, -14.6403999, 14.6332588
32: -24.9338303, -4.5671339, -24.9372997, -4.5692797, -13.2639580, 13.2961731
33: -69.3026733, -40.1172295, -69.3518143, -40.1007767, -16.6114807, 16.6473846
34: -53.7610703, -30.9141731, -53.7932320, -30.9031696, -14.1122856, 14.1454430
35: -47.8085938, -26.0619640, -47.8253555, -26.0651588, -12.9886093, 12.9887238
36: -42.7905159, -19.2837372, -42.8157997, -19.2742004, -15.0717926, 15.0679893
37: -86.6640244, -55.5555267, -86.6786194, -55.5455093, -18.9102325, 18.8981972
38: -52.9338341, -24.3204517, -52.9703140, -24.3193703, -18.2996750, 18.3451996
39: -76.5472794, -44.6362305, -76.5652161, -44.6280746, -16.0837555, 16.0521507
40: -67.2454224, -43.5431137, -67.3050919, -43.5293732, -14.3368683, 14.3047543
41: -55.4366531, -32.9583588, -55.4651566, -32.9488640, -16.6768761, 16.6635818
42: -29.4715042, -9.8921337, -29.4703007, -9.9018688, -17.2554474, 17.2179909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -12.5515192, upper bound: 12.5558543
time: 7.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5515242, upper bound: 12.5673657
time: 9.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.1217737, 3.6797733, -12.1387882, 3.6803410, -13.8820648, 13.8910599
1: -3.6679621, 7.4001713, -3.6794753, 7.4010205, -8.4887581, 8.4870834
2: -0.7431009, 13.4375553, -0.7673851, 13.4369516, -13.4474411, 13.4636154
3: -1.1286856, 11.3154058, -1.1416385, 11.3141193, -12.0195847, 12.0321846
4: -11.1125259, 5.4882879, -11.1239357, 5.4893293, -14.6677551, 14.6810684
5: 1.8436975, 17.7517605, 1.8243427, 17.7505379, -15.9068403, 15.9274178
6: -39.9291878, -18.2255764, -39.9327316, -18.2275887, -15.0980492, 15.1863976
7: -3.5782115, 12.2605190, -3.6102514, 12.2625065, -13.5954666, 13.6361084
8: -6.7042847, 8.5738955, -6.7227964, 8.5738277, -12.0905228, 12.1216583
9: -4.7868576, 11.7170630, -4.7813029, 11.7230844, -13.0152588, 12.9815598
10: 1.3073821, 25.7429276, 1.3092246, 25.7492409, -20.9280624, 20.9209442
11: -11.5050049, 4.2866035, -11.5035419, 4.2872987, -15.7923031, 15.7901459
12: -11.9115028, 9.8290424, -11.9090290, 9.8578796, -15.0356750, 15.0098572
13: -18.5552979, 6.7249260, -18.5475922, 6.7363696, -16.6013336, 16.5963783
14: 4.9460878, 36.4202805, 4.9487963, 36.4333687, -26.7486115, 26.6869812
15: -8.6914711, 9.2897444, -8.6919632, 9.2910738, -17.9825439, 17.9817085
16: -16.7347775, 2.5428965, -16.7513390, 2.5458579, -14.8038483, 14.8007927
17: 6.1949792, 30.6577835, 6.1931548, 30.6774349, -17.2248497, 17.2320595
18: -14.3952656, 5.1300344, -14.4023399, 5.1301413, -14.4155140, 14.4070435
19: -20.2791271, -4.3217473, -20.2813702, -4.3213058, -14.5451012, 14.5361633
20: -2.4252977, 11.2261639, -2.4275179, 11.2269936, -12.6121025, 12.6208878
21: -11.0803862, 3.2508090, -11.0829458, 3.2640157, -14.3444023, 14.3337545
22: -3.7023323, 13.1112967, -3.7047758, 13.1428633, -14.9629211, 14.9351959
23: -14.5838184, 0.3490529, -14.5863228, 0.3485532, -14.3352470, 14.3024406
24: -19.9333763, -5.1137791, -19.9358387, -5.1106758, -9.2680969, 9.2696037
25: -5.4582949, 10.8599205, -5.4638758, 10.8859024, -13.8141251, 13.7963219
26: -21.0206738, 1.2121565, -21.0204411, 1.2553422, -19.3521194, 19.3054810
27: -16.0100155, 2.1815319, -16.0244598, 2.1810856, -13.2011833, 13.2231483
28: -12.7992077, 4.6454339, -12.8017569, 4.6466942, -17.4459019, 17.4471912
29: -5.5976663, 11.8898983, -5.5988054, 11.9246397, -14.9680519, 14.9446983
30: -10.0510550, 6.2068272, -10.0520382, 6.2222471, -13.5568275, 13.5480385
31: -10.9775467, 6.9522476, -10.9853439, 6.9497976, -14.6424751, 14.6474419
32: -24.9242859, -4.5637550, -24.9285507, -4.5624814, -13.2525864, 13.3069878
33: -69.3129120, -40.0983582, -69.3208084, -40.1006088, -16.6186676, 16.6220627
34: -53.7630653, -30.9020958, -53.7666512, -30.9054794, -14.0880470, 14.1514740
35: -47.8117218, -26.0634232, -47.8090248, -26.0713959, -12.9912186, 12.9844551
36: -42.8132210, -19.2781219, -42.8082924, -19.2794228, -15.0737228, 15.0759735
37: -86.6724319, -55.5432663, -86.6719513, -55.5425415, -18.9192657, 18.9215851
38: -52.9483528, -24.3228874, -52.9521103, -24.3263702, -18.3121109, 18.3440933
39: -76.5521393, -44.6230164, -76.5493774, -44.6256065, -16.0843658, 16.0542984
40: -67.2510071, -43.5190506, -67.2853851, -43.5197449, -14.3017654, 14.3408318
41: -55.4296188, -32.9528770, -55.4419708, -32.9523849, -16.6537819, 16.6842270
42: -29.4687347, -9.8852825, -29.4647694, -9.8903885, -17.2509689, 17.2462311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5150487
time: 13.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5265653
time: 25.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.1202583, 3.6799970, -12.1374855, 3.6857853, -13.8869629, 13.8910332
1: -3.6647120, 7.4004431, -3.6750646, 7.4046011, -8.5003357, 8.4892502
2: -0.7418248, 13.4377823, -0.7663115, 13.4389172, -13.4494171, 13.4642258
3: -1.1286469, 11.3122883, -1.1417476, 11.3117352, -12.0214157, 12.0381660
4: -11.1088066, 5.4885669, -11.1190434, 5.4903402, -14.6677551, 14.6789246
5: 1.8437328, 17.7506523, 1.8239222, 17.7504654, -15.9067326, 15.9267302
6: -39.9293327, -18.2304382, -39.9399872, -18.2340584, -15.0975342, 15.1906738
7: -3.5776613, 12.2605515, -3.6117539, 12.2657547, -13.6050110, 13.6369629
8: -6.7025003, 8.5740547, -6.7223368, 8.5749178, -12.0921745, 12.1225071
9: -4.7888694, 11.7170029, -4.7891049, 11.7433281, -13.0292549, 12.9859734
10: 1.3031263, 25.7428246, 1.2952294, 25.7684345, -20.9477921, 20.9326324
11: -11.5066662, 4.2859778, -11.5108805, 4.2932553, -15.7999210, 15.7968578
12: -11.9117031, 9.8287554, -11.9147482, 9.8628635, -15.0395775, 15.0148544
13: -18.5556374, 6.7247982, -18.5502129, 6.7470312, -16.6052856, 16.5983124
14: 4.9409637, 36.4203911, 4.9303989, 36.4713707, -26.7841263, 26.7021713
15: -8.6887283, 9.2902317, -8.6891899, 9.2954588, -17.9841881, 17.9794216
16: -16.7364235, 2.5429473, -16.7618675, 2.5587547, -14.8226929, 14.8080864
17: 6.1941109, 30.6578083, 6.1847529, 30.6840515, -17.2228012, 17.2457085
18: -14.3914213, 5.1303315, -14.3986340, 5.1318140, -14.4179039, 14.4099636
19: -20.2778492, -4.3215008, -20.2829266, -4.3201394, -14.5448227, 14.5417976
20: -2.4255202, 11.2250433, -2.4320655, 11.2259274, -12.6118088, 12.6223106
21: -11.0785999, 3.2507582, -11.0849829, 3.2646532, -14.3432531, 14.3357410
22: -3.7022145, 13.1118202, -3.7113256, 13.1470890, -14.9663773, 14.9473038
23: -14.5841951, 0.3490791, -14.5908241, 0.3498526, -14.3373566, 14.3049774
24: -19.9318371, -5.1138167, -19.9345169, -5.1102948, -9.2695084, 9.2711563
25: -5.4579182, 10.8599539, -5.4661465, 10.8873644, -13.8135300, 13.8037033
26: -21.0175056, 1.2122653, -21.0194244, 1.2574582, -19.3529205, 19.3092499
27: -16.0098572, 2.1825743, -16.0343590, 2.1846113, -13.2050018, 13.2246513
28: -12.7993908, 4.6462283, -12.8100634, 4.6500711, -17.4494629, 17.4562912
29: -5.5977345, 11.8896074, -5.6011038, 11.9262295, -14.9689178, 14.9520302
30: -10.0519705, 6.2065296, -10.0572376, 6.2287040, -13.5648766, 13.5526924
31: -10.9773226, 6.9528418, -10.9930477, 6.9518709, -14.6439400, 14.6540565
32: -24.9243965, -4.5702529, -24.9384956, -4.5712876, -13.2523613, 13.3168869
33: -69.3125916, -40.0939789, -69.3524475, -40.0873489, -16.6273499, 16.6554718
34: -53.7630424, -30.8990021, -53.7940712, -30.8945274, -14.0966339, 14.1780357
35: -47.8116302, -26.0616398, -47.8265228, -26.0638771, -12.9981537, 13.0040016
36: -42.8130913, -19.2767410, -42.8287544, -19.2732697, -15.0775528, 15.0925064
37: -86.6721878, -55.5416565, -86.6826324, -55.5369835, -18.9236069, 18.9288559
38: -52.9482803, -24.3209457, -52.9786606, -24.3185806, -18.3168564, 18.3593521
39: -76.5519638, -44.6206741, -76.5659409, -44.6187973, -16.0897293, 16.0704880
40: -67.2508545, -43.5161514, -67.3064346, -43.5127106, -14.3062248, 14.3392906
41: -55.4297028, -32.9503479, -55.4657516, -32.9441910, -16.6568298, 16.6962509
42: -29.4689350, -9.8933830, -29.4709835, -9.9024954, -17.2489662, 17.2502098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=91, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 977
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 675
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1291
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1286
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1285
type: A, layer: 1, pos: 1292
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1294
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1295
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 977

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5567951
time: 10.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5682967
time: 15.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 28.28 seconds
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5662908, upper bound: 12.4291647
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5663000, upper bound: 12.4402004
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5662908, upper bound: 12.4700291
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5663000, upper bound: 12.4809688
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5673617, upper bound: 12.4665543
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5673693, upper bound: 12.4777571
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5673617, upper bound: 12.5077196
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5673693, upper bound: 12.5188159
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5671240, upper bound: 12.4620934
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5671317, upper bound: 12.4736059
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5671240, upper bound: 12.5038161
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5671317, upper bound: 12.5152519
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5681592, upper bound: 12.4982333
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5097549
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5681592, upper bound: 12.5399798
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5515008
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4449258
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4558591
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4856821
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4966142
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5674973, upper bound: 12.4826541
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5675011, upper bound: 12.4937356
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5674973, upper bound: 12.5236515
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5675011, upper bound: 12.5346328
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5369670, upper bound: 12.5558341
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5369722, upper bound: 12.5673364
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5535409, upper bound: 12.5567753
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5535439, upper bound: 12.5682675
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5672610, upper bound: 12.4788643
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5672643, upper bound: 12.4902819
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5672610, upper bound: 12.5204256
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5672643, upper bound: 12.5317417
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5515192, upper bound: 12.5558543
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5515242, upper bound: 12.5673657
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5150487
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5265653
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5567951
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 28.28
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5682967

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.0886564, 3.6438189, -12.1108799, 3.6538365, -13.8236504, 13.8263474
1: -3.6432047, 7.3777871, -3.6515601, 7.3822851, -8.4579582, 8.4466400
2: -0.7231992, 13.4162865, -0.7344572, 13.4130449, -13.4013519, 13.4052048
3: -1.1073748, 11.2847576, -1.1179965, 11.2818232, -11.9645729, 11.9823475
4: -11.0766773, 5.4557972, -11.0894833, 5.4721103, -14.6136017, 14.6124496
5: 1.8737440, 17.7265263, 1.8530521, 17.7234268, -15.8496828, 15.8734741
6: -39.8425980, -18.2863388, -39.8670120, -18.2497520, -15.0790443, 15.0664864
7: -3.5333257, 12.2251482, -3.5561717, 12.2329655, -13.5563889, 13.5444145
8: -6.6841168, 8.5431252, -6.6990852, 8.5478897, -12.0369186, 12.0663528
9: -4.7404451, 11.6717319, -4.7576056, 11.6881962, -12.9367599, 12.9422760
10: 1.3703852, 25.7164574, 1.3678355, 25.7175522, -20.8327026, 20.8249512
11: -11.4782076, 4.2821741, -11.4814606, 4.2834663, -15.7616739, 15.7636347
12: -11.8708382, 9.8018341, -11.8662701, 9.8196421, -14.9577141, 14.9495316
13: -18.5444565, 6.6821575, -18.5414219, 6.7036858, -16.5904617, 16.5374069
14: 5.0321140, 36.3582687, 5.0179529, 36.3701935, -26.6076584, 26.6044159
15: -8.6469860, 9.2088480, -8.6815424, 9.2374296, -17.8844147, 17.8903904
16: -16.6836948, 2.5230312, -16.6971893, 2.5235486, -14.7339211, 14.7468681
17: 6.2540631, 30.6160088, 6.2382851, 30.6258163, -17.1289406, 17.1529312
18: -14.3444176, 5.1069312, -14.3636389, 5.1115270, -14.3435555, 14.3496037
19: -20.2439575, -4.3421307, -20.2538033, -4.3320966, -14.4882431, 14.4851952
20: -2.3915753, 11.2000656, -2.3979487, 11.2084656, -12.5730209, 12.5724030
21: -11.0402212, 3.2405782, -11.0465345, 3.2478147, -14.2880363, 14.2871132
22: -3.6669183, 13.0607891, -3.6748037, 13.0818329, -14.8587341, 14.8687553
23: -14.5419540, 0.2997317, -14.5682373, 0.3165474, -14.2527657, 14.2560234
24: -19.9198895, -5.1290259, -19.9252110, -5.1193457, -9.2446289, 9.2319298
25: -5.4272213, 10.8295822, -5.4317131, 10.8431425, -13.7427177, 13.7479782
26: -20.9700489, 1.1499140, -20.9806023, 1.1772470, -19.2340393, 19.2488480
27: -15.9865055, 2.1567788, -15.9908266, 2.1527483, -13.1559334, 13.1835022
28: -12.7559509, 4.5939455, -12.7775860, 4.6081462, -17.3640976, 17.3715324
29: -5.5503941, 11.8377800, -5.5743971, 11.8594656, -14.8556824, 14.8834305
30: -10.0305767, 6.1935234, -10.0312061, 6.1967907, -13.5079613, 13.5122299
31: -10.9265461, 6.9459815, -10.9448357, 6.9438095, -14.5852432, 14.6094170
32: -24.8770370, -4.6048875, -24.8817825, -4.5901699, -13.2313766, 13.2299690
33: -69.2511292, -40.1684952, -69.2705383, -40.1319351, -16.5638885, 16.5279846
34: -53.7127304, -30.9564133, -53.7220001, -30.9415226, -14.0585709, 14.0614357
35: -47.7895699, -26.1027336, -47.7978325, -26.0808449, -12.9579468, 12.9218521
36: -42.8006592, -19.3195114, -42.7963142, -19.3001709, -15.0331726, 15.0096054
37: -86.6466827, -55.5794907, -86.6526947, -55.5611572, -18.8634033, 18.8529510
38: -52.8697014, -24.4036541, -52.8889313, -24.3585758, -18.2581940, 18.1973343
39: -76.4999084, -44.6770935, -76.5221786, -44.6367111, -16.0017929, 15.9575768
40: -67.1999664, -43.5452156, -67.2180176, -43.5562439, -14.2461662, 14.2943764
41: -55.3907394, -32.9933929, -55.3952789, -32.9904213, -16.6125488, 16.6253014
42: -29.4504356, -9.8926964, -29.4481812, -9.8989410, -17.2087250, 17.2196846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646801, upper bound: 12.3937736
time: 6.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4276022
time: 11.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.0963860, 3.6523056, -12.1120243, 3.6585782, -13.8362198, 13.8333969
1: -3.6470838, 7.3847942, -3.6518059, 7.3860512, -8.4659576, 8.4535103
2: -0.7287858, 13.4235172, -0.7349327, 13.4171524, -13.4104424, 13.4108047
3: -1.1133758, 11.2935839, -1.1184523, 11.2868032, -11.9751091, 11.9879074
4: -11.0803518, 5.4573398, -11.0908089, 5.4727836, -14.6263885, 14.6144447
5: 1.8624439, 17.7374706, 1.8522358, 17.7298145, -15.8673706, 15.8852348
6: -39.8533745, -18.2779121, -39.8731041, -18.2493973, -15.0849304, 15.0814400
7: -3.5418265, 12.2339268, -3.5569813, 12.2380152, -13.5685043, 13.5508575
8: -6.6965151, 8.5586033, -6.6997275, 8.5568981, -12.0587082, 12.0765743
9: -4.7457623, 11.6782560, -4.7581577, 11.6917286, -12.9477768, 12.9499092
10: 1.3583031, 25.7304649, 1.3664069, 25.7253017, -20.8535156, 20.8314285
11: -11.4809895, 4.2835960, -11.4823742, 4.2839909, -15.7649803, 15.7659702
12: -11.8772230, 9.8100939, -11.8699760, 9.8203583, -14.9609070, 14.9599953
13: -18.5511990, 6.6932597, -18.5453129, 6.7054024, -16.5960617, 16.5464096
14: 5.0097628, 36.3815117, 5.0153217, 36.3837166, -26.6421661, 26.6243210
15: -8.6532516, 9.2180109, -8.6826267, 9.2420845, -17.8953362, 17.9006386
16: -16.6936989, 2.5336833, -16.6980820, 2.5295105, -14.7507019, 14.7540321
17: 6.2397304, 30.6300507, 6.2367787, 30.6340485, -17.1517487, 17.1578674
18: -14.3546124, 5.1079106, -14.3690805, 5.1118212, -14.3545723, 14.3573055
19: -20.2552757, -4.3369737, -20.2597256, -4.3320050, -14.4954910, 14.4945984
20: -2.4015281, 11.2057972, -2.4032221, 11.2086887, -12.5794563, 12.5819244
21: -11.0514412, 3.2438591, -11.0523891, 3.2480369, -14.2994785, 14.2962484
22: -3.6770244, 13.0655699, -3.6802516, 13.0822306, -14.8647346, 14.8777924
23: -14.5452948, 0.3021455, -14.5694256, 0.3172994, -14.2579231, 14.2658577
24: -19.9265442, -5.1247139, -19.9283829, -5.1190925, -9.2510376, 9.2412262
25: -5.4326487, 10.8317928, -5.4338293, 10.8436432, -13.7486038, 13.7542496
26: -20.9809303, 1.1562707, -20.9864197, 1.1777730, -19.2431221, 19.2617645
27: -15.9902773, 2.1578138, -15.9923801, 2.1529908, -13.1612701, 13.1866570
28: -12.7622948, 4.5970163, -12.7806587, 4.6089530, -17.3712482, 17.3776741
29: -5.5531874, 11.8394241, -5.5755348, 11.8598938, -14.8592377, 14.8866768
30: -10.0343742, 6.1984019, -10.0324097, 6.1990223, -13.5138092, 13.5192871
31: -10.9404812, 6.9469175, -10.9513454, 6.9438496, -14.5968437, 14.6168633
32: -24.8881454, -4.5949745, -24.8880997, -4.5898190, -13.2350311, 13.2467995
33: -69.2772522, -40.1439476, -69.2849197, -40.1310463, -16.5694695, 16.5653381
34: -53.7275314, -30.9406185, -53.7308121, -30.9409218, -14.0636559, 14.0857620
35: -47.8094864, -26.0815315, -47.8095322, -26.0802498, -12.9604492, 12.9541702
36: -42.8208313, -19.2950630, -42.8084030, -19.2994728, -15.0339012, 15.0467834
37: -86.6655502, -55.5615273, -86.6633911, -55.5599747, -18.8680916, 18.8786774
38: -52.8992500, -24.3722801, -52.9063644, -24.3579369, -18.2646904, 18.2478714
39: -76.5276947, -44.6492081, -76.5381088, -44.6359634, -16.0017166, 16.0014114
40: -67.2085800, -43.5452652, -67.2220459, -43.5558205, -14.2578773, 14.3004990
41: -55.4042587, -32.9783897, -55.4032440, -32.9896736, -16.6192017, 16.6484222
42: -29.4521656, -9.8902378, -29.4487610, -9.8981667, -17.2124748, 17.2328568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4050795
time: 18.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4386391
time: 6.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.0871229, 3.6440823, -12.1095638, 3.6592762, -13.8285561, 13.8263054
1: -3.6399589, 7.3780670, -3.6471252, 7.3858747, -8.4695511, 8.4488258
2: -0.7219295, 13.4165010, -0.7333791, 13.4150190, -13.4033356, 13.4058189
3: -1.1073527, 11.2816534, -1.1180885, 11.2794514, -11.9663773, 11.9883308
4: -11.0729656, 5.4560661, -11.0845699, 5.4731092, -14.6135635, 14.6102905
5: 1.8737602, 17.7254219, 1.8525996, 17.7233543, -15.8495941, 15.8728218
6: -39.8427200, -18.2911892, -39.8742676, -18.2562237, -15.0785408, 15.0707893
7: -3.5327303, 12.2251835, -3.5576599, 12.2361717, -13.5659027, 13.5452957
8: -6.6823502, 8.5432768, -6.6986408, 8.5489731, -12.0386047, 12.0671825
9: -4.7424612, 11.6716824, -4.7654152, 11.7084799, -12.9507675, 12.9466820
10: 1.3661580, 25.7163239, 1.3537765, 25.7367535, -20.8524246, 20.8366776
11: -11.4798889, 4.2815704, -11.4887915, 4.2894125, -15.7693014, 15.7703619
12: -11.8710442, 9.8015432, -11.8719444, 9.8246326, -14.9616051, 14.9545212
13: -18.5448399, 6.6820612, -18.5440235, 6.7142801, -16.5944061, 16.5393906
14: 5.0269709, 36.3582916, 4.9995804, 36.4082260, -26.6431885, 26.6195908
15: -8.6442480, 9.2093706, -8.6787682, 9.2417679, -17.8860168, 17.8881378
16: -16.6853371, 2.5231121, -16.7076836, 2.5364330, -14.7527580, 14.7541275
17: 6.2531576, 30.6160469, 6.2298470, 30.6324692, -17.1269226, 17.1666031
18: -14.3406010, 5.1072388, -14.3599281, 5.1131849, -14.3459358, 14.3525295
19: -20.2426739, -4.3418760, -20.2553902, -4.3309369, -14.4879837, 14.4907990
20: -2.3917761, 11.1989183, -2.4024839, 11.2073736, -12.5727310, 12.5738106
21: -11.0384426, 3.2404985, -11.0485315, 3.2484522, -14.2868948, 14.2890301
22: -3.6667695, 13.0612755, -3.6813488, 13.0860386, -14.8621750, 14.8808937
23: -14.5423679, 0.2997456, -14.5727348, 0.3178272, -14.2548447, 14.2585564
24: -19.9183540, -5.1290574, -19.9238949, -5.1189942, -9.2460670, 9.2334785
25: -5.4268723, 10.8295975, -5.4339714, 10.8446007, -13.7421455, 13.7553482
26: -20.9669266, 1.1500292, -20.9795742, 1.1793399, -19.2348938, 19.2525711
27: -15.9863806, 2.1578240, -16.0007019, 2.1562529, -13.1598015, 13.1850014
28: -12.7560930, 4.5947642, -12.7859173, 4.6115170, -17.3676109, 17.3806820
29: -5.5504732, 11.8374662, -5.5767093, 11.8610334, -14.8565826, 14.8907776
30: -10.0314922, 6.1932364, -10.0363903, 6.2032890, -13.5159836, 13.5168953
31: -10.9263325, 6.9465370, -10.9525070, 6.9458742, -14.5867233, 14.6160431
32: -24.8771477, -4.6114030, -24.8917408, -4.5989685, -13.2311592, 13.2398911
33: -69.2507935, -40.1640816, -69.3021851, -40.1186523, -16.5725632, 16.5613861
34: -53.7127686, -30.9532795, -53.7494125, -30.9305573, -14.0671387, 14.0879669
35: -47.7894821, -26.1009293, -47.8154068, -26.0732956, -12.9648972, 12.9413872
36: -42.8005753, -19.3180771, -42.8168373, -19.2940254, -15.0370102, 15.0260696
37: -86.6464081, -55.5778160, -86.6633301, -55.5556793, -18.8677063, 18.8602524
38: -52.8696327, -24.4016800, -52.9154968, -24.3508530, -18.2629700, 18.2125854
39: -76.4997635, -44.6747398, -76.5386658, -44.6298904, -16.0071640, 15.9737587
40: -67.1998291, -43.5423355, -67.2390594, -43.5492287, -14.2506218, 14.2928295
41: -55.3907852, -32.9908333, -55.4190979, -32.9821625, -16.6155853, 16.6374054
42: -29.4506645, -9.9008131, -29.4544067, -9.9110622, -17.2066841, 17.2236633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4351207
time: 6.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4684815
time: 13.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.0948544, 3.6525500, -12.1107025, 3.6640098, -13.8411293, 13.8333740
1: -3.6438444, 7.3850937, -3.6474102, 7.3896441, -8.4775505, 8.4556675
2: -0.7275097, 13.4237261, -0.7338561, 13.4191351, -13.4124222, 13.4114037
3: -1.1133349, 11.2904854, -1.1185410, 11.2844200, -11.9769363, 11.9938850
4: -11.0766153, 5.4576015, -11.0858955, 5.4737902, -14.6263809, 14.6123047
5: 1.8624763, 17.7364006, 1.8518076, 17.7297516, -15.8672752, 15.8845930
6: -39.8534851, -18.2827625, -39.8803558, -18.2558784, -15.0844154, 15.0857353
7: -3.5412381, 12.2339630, -3.5584717, 12.2412758, -13.5780487, 13.5517044
8: -6.6947374, 8.5587616, -6.6992750, 8.5579720, -12.0603867, 12.0773945
9: -4.7477574, 11.6782141, -4.7659588, 11.7119894, -12.9617996, 12.9542961
10: 1.3540673, 25.7303391, 1.3523326, 25.7445221, -20.8732147, 20.8431473
11: -11.4826660, 4.2829728, -11.4896851, 4.2899389, -15.7726049, 15.7726574
12: -11.8774385, 9.8098030, -11.8756676, 9.8253422, -14.9647713, 14.9649849
13: -18.5515518, 6.6931772, -18.5479774, 6.7160254, -16.6000099, 16.5483589
14: 5.0046387, 36.3815269, 4.9969730, 36.4217911, -26.6777344, 26.6395645
15: -8.6505566, 9.2184925, -8.6798611, 9.2463808, -17.8969383, 17.8983536
16: -16.6953278, 2.5337608, -16.7085953, 2.5424094, -14.7695351, 14.7612877
17: 6.2388349, 30.6300812, 6.2283707, 30.6407013, -17.1497154, 17.1715355
18: -14.3508186, 5.1082196, -14.3654013, 5.1134806, -14.3569584, 14.3602276
19: -20.2540150, -4.3367400, -20.2613182, -4.3308535, -14.4952202, 14.5002556
20: -2.4017103, 11.2046547, -2.4077706, 11.2076082, -12.5791626, 12.5833168
21: -11.0496845, 3.2437885, -11.0543690, 3.2486906, -14.2983751, 14.2981577
22: -3.6768966, 13.0660973, -3.6868141, 13.0864277, -14.8681717, 14.8899193
23: -14.5457058, 0.3021734, -14.5739079, 0.3185949, -14.2600098, 14.2683601
24: -19.9250259, -5.1247444, -19.9270439, -5.1187320, -9.2524643, 9.2428055
25: -5.4322696, 10.8318157, -5.4361019, 10.8451309, -13.7479820, 13.7616272
26: -20.9777718, 1.1563873, -20.9854164, 1.1798739, -19.2439919, 19.2655182
27: -15.9901533, 2.1588516, -16.0022736, 2.1565108, -13.1651230, 13.1881599
28: -12.7624369, 4.5978708, -12.7889767, 4.6122832, -17.3747196, 17.3868484
29: -5.5532436, 11.8391037, -5.5778427, 11.8615389, -14.8600883, 14.8940010
30: -10.0352955, 6.1981230, -10.0376101, 6.2054987, -13.5218391, 13.5239487
31: -10.9402676, 6.9474893, -10.9590530, 6.9459224, -14.5983391, 14.6235008
32: -24.8882599, -4.6014833, -24.8980732, -4.5986366, -13.2348328, 13.2566910
33: -69.2768860, -40.1395340, -69.3165588, -40.1177673, -16.5781288, 16.5987663
34: -53.7275581, -30.9375305, -53.7582321, -30.9299698, -14.0722046, 14.1122818
35: -47.8094177, -26.0797062, -47.8271408, -26.0727520, -12.9673882, 12.9737282
36: -42.8207359, -19.2936630, -42.8288727, -19.2933197, -15.0377197, 15.0632744
37: -86.6652527, -55.5599213, -86.6740570, -55.5544739, -18.8724060, 18.8859749
38: -52.8992004, -24.3703403, -52.9329529, -24.3502216, -18.2694473, 18.2631683
39: -76.5274734, -44.6469193, -76.5546417, -44.6291580, -16.0070648, 16.0176125
40: -67.2084427, -43.5423660, -67.2430954, -43.5487518, -14.2623482, 14.2989502
41: -55.4043350, -32.9758530, -55.4270554, -32.9814148, -16.6222534, 16.6604996
42: -29.4523773, -9.8983698, -29.4550056, -9.9102783, -17.2104912, 17.2368317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4462526
time: 26.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4794225
time: 21.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.0901489, 3.6443386, -12.1132984, 3.6532478, -13.8241234, 13.8295898
1: -3.6472311, 7.3788414, -3.6568165, 7.3847237, -8.4638557, 8.4513893
2: -0.7239020, 13.4177332, -0.7353423, 13.4155540, -13.4080276, 13.4072380
3: -1.1080734, 11.2899160, -1.1201286, 11.2906199, -11.9713974, 11.9905624
4: -11.0778236, 5.4573975, -11.0914383, 5.4731464, -14.6231842, 14.6197433
5: 1.8731003, 17.7300701, 1.8541284, 17.7292290, -15.8561287, 15.8759422
6: -39.8433762, -18.2738342, -39.8789291, -18.2287273, -15.0744133, 15.0908470
7: -3.5342615, 12.2272377, -3.5582099, 12.2366133, -13.5545120, 13.5496368
8: -6.6860290, 8.5440292, -6.7016296, 8.5496111, -12.0498123, 12.0691376
9: -4.7492571, 11.6721992, -4.7714357, 11.6914310, -12.9472809, 12.9485741
10: 1.3543348, 25.7171783, 1.3405437, 25.7277946, -20.8592758, 20.8414459
11: -11.4826012, 4.2822857, -11.4881945, 4.2833133, -15.7659149, 15.7704802
12: -11.8779631, 9.8025904, -11.8778992, 9.8242569, -14.9708481, 14.9550209
13: -18.5511837, 6.6839161, -18.5514565, 6.7088804, -16.6138229, 16.5330734
14: 5.0112476, 36.3590202, 4.9821463, 36.3854332, -26.6468201, 26.6194611
15: -8.6521215, 9.2111578, -8.6904354, 9.2430935, -17.8952141, 17.9015923
16: -16.6958656, 2.5235736, -16.7142487, 2.5211744, -14.7342758, 14.7638016
17: 6.2440014, 30.6165886, 6.2205396, 30.6343269, -17.1365623, 17.1639557
18: -14.3457031, 5.1092196, -14.3646975, 5.1152449, -14.3523846, 14.3517647
19: -20.2465076, -4.3387237, -20.2585258, -4.3274441, -14.4956360, 14.4947052
20: -2.3933256, 11.2088461, -2.4050314, 11.2220335, -12.5818253, 12.5852928
21: -11.0429926, 3.2417445, -11.0509472, 3.2489381, -14.2919312, 14.2926922
22: -3.6679192, 13.0649881, -3.6762915, 13.0863857, -14.8648872, 14.8761787
23: -14.5444002, 0.3035400, -14.5707779, 0.3218908, -14.2612686, 14.2650948
24: -19.9212437, -5.1283665, -19.9275093, -5.1186924, -9.2473755, 9.2353973
25: -5.4312305, 10.8301258, -5.4386473, 10.8457508, -13.7458572, 13.7530098
26: -20.9730206, 1.1514711, -20.9858475, 1.1803062, -19.2398949, 19.2563019
27: -15.9876471, 2.1657443, -16.0008106, 2.1673646, -13.1495781, 13.2039337
28: -12.7577171, 4.6003408, -12.7844448, 4.6188164, -17.3765335, 17.3847847
29: -5.5514927, 11.8392248, -5.5764971, 11.8612747, -14.8584785, 14.8945389
30: -10.0346298, 6.1940975, -10.0374794, 6.1988621, -13.5103951, 13.5175095
31: -10.9293613, 6.9491000, -10.9493618, 6.9472599, -14.5911713, 14.6161995
32: -24.8780518, -4.5914226, -24.8966446, -4.5678616, -13.2321129, 13.2563400
33: -69.2520447, -40.1593094, -69.2777786, -40.1168976, -16.5706520, 16.5297585
34: -53.7132378, -30.9445953, -53.7323761, -30.9214115, -14.0633621, 14.0849533
35: -47.7900238, -26.0967884, -47.8019142, -26.0720520, -12.9665909, 12.9325638
36: -42.8008537, -19.3089657, -42.8031998, -19.2830544, -15.0405273, 15.0266533
37: -86.6476974, -55.5751152, -86.6562347, -55.5539703, -18.8736420, 18.8557816
38: -52.8701973, -24.3891220, -52.9028435, -24.3348579, -18.2586594, 18.2268677
39: -76.5010986, -44.6737099, -76.5243835, -44.6313553, -16.0128059, 15.9618378
40: -67.2008514, -43.5368767, -67.2256317, -43.5423203, -14.2382202, 14.3133049
41: -55.3916473, -32.9811745, -55.4077454, -32.9696884, -16.6118660, 16.6519051
42: -29.4513969, -9.8819389, -29.4589157, -9.8807678, -17.2088966, 17.2400131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4302670
time: 10.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4650409
time: 12.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.0978937, 3.6528187, -12.1144371, 3.6579926, -13.8367348, 13.8366814
1: -3.6511164, 7.3858376, -3.6570845, 7.3884912, -8.4718399, 8.4582462
2: -0.7295026, 13.4249668, -0.7357905, 13.4196644, -13.4171333, 13.4128418
3: -1.1140665, 11.2987633, -1.1205772, 11.2955770, -11.9819336, 11.9961166
4: -11.0815248, 5.4589462, -11.0927715, 5.4738178, -14.6360168, 14.6217422
5: 1.8618150, 17.7410526, 1.8533096, 17.7356224, -15.8738079, 15.8877430
6: -39.8541718, -18.2653961, -39.8850098, -18.2283669, -15.0802956, 15.1058235
7: -3.5428183, 12.2359991, -3.5590048, 12.2417040, -13.5666504, 13.5560837
8: -6.6984134, 8.5594778, -6.7022614, 8.5586243, -12.0715866, 12.0793419
9: -4.7545891, 11.6787157, -4.7719808, 11.6949615, -12.9583206, 12.9561958
10: 1.3422680, 25.7311878, 1.3391471, 25.7355766, -20.8800659, 20.8479385
11: -11.4853792, 4.2836957, -11.4890804, 4.2838397, -15.7692184, 15.7727757
12: -11.8843527, 9.8108387, -11.8816624, 9.8249922, -14.9740257, 14.9655037
13: -18.5578651, 6.6950531, -18.5553379, 6.7106152, -16.6193886, 16.5420990
14: 4.9888744, 36.3822746, 4.9795494, 36.3989716, -26.6813660, 26.6393890
15: -8.6583691, 9.2203264, -8.6915331, 9.2477283, -17.9060974, 17.9118595
16: -16.7058754, 2.5342247, -16.7151375, 2.5271595, -14.7510796, 14.7709618
17: 6.2296534, 30.6306877, 6.2190237, 30.6425552, -17.1593628, 17.1688766
18: -14.3559074, 5.1101837, -14.3701525, 5.1155338, -14.3634148, 14.3594570
19: -20.2578526, -4.3335729, -20.2644520, -4.3273687, -14.5028839, 14.5041580
20: -2.4032676, 11.2145882, -2.4102936, 11.2222767, -12.5882797, 12.5948219
21: -11.0542526, 3.2450495, -11.0567722, 3.2491698, -14.3034229, 14.3018217
22: -3.6780291, 13.0698128, -3.6817403, 13.0867634, -14.8708878, 14.8851967
23: -14.5477457, 0.3059826, -14.5719204, 0.3226418, -14.2664490, 14.2749290
24: -19.9279099, -5.1240592, -19.9306602, -5.1184282, -9.2537766, 9.2447319
25: -5.4366789, 10.8323555, -5.4407802, 10.8462620, -13.7516975, 13.7592773
26: -20.9838810, 1.1578290, -20.9917068, 1.1808486, -19.2489853, 19.2692490
27: -15.9914427, 2.1667738, -16.0023727, 2.1675773, -13.1549225, 13.2070999
28: -12.7640333, 4.6034384, -12.7875118, 4.6196074, -17.3836403, 17.3909492
29: -5.5542636, 11.8408852, -5.5776019, 11.8617668, -14.8620491, 14.8977509
30: -10.0384312, 6.1989460, -10.0387068, 6.2010722, -13.5162468, 13.5245705
31: -10.9432735, 6.9500365, -10.9558945, 6.9473000, -14.6027718, 14.6236572
32: -24.8892002, -4.5814734, -24.9029732, -4.5675039, -13.2357864, 13.2731438
33: -69.2781830, -40.1346741, -69.2921448, -40.1160202, -16.5762291, 16.5671310
34: -53.7280426, -30.9287949, -53.7411919, -30.9208088, -14.0684624, 14.1092834
35: -47.8099213, -26.0755692, -47.8136597, -26.0714550, -12.9690895, 12.9648972
36: -42.8209801, -19.2845192, -42.8152695, -19.2823448, -15.0412750, 15.0638428
37: -86.6665115, -55.5571899, -86.6669464, -55.5528221, -18.8783569, 18.8815041
38: -52.8997955, -24.3577194, -52.9202690, -24.3342209, -18.2651825, 18.2774811
39: -76.5288544, -44.6458664, -76.5403137, -44.6306534, -16.0127182, 16.0056992
40: -67.2094574, -43.5369606, -67.2296524, -43.5419159, -14.2499504, 14.3194160
41: -55.4052162, -32.9661293, -55.4156914, -32.9689407, -16.6184845, 16.6750221
42: -29.4531021, -9.8794880, -29.4594803, -9.8799715, -17.2126312, 17.2531738

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4417634
time: 18.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4762515
time: 23.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.0886354, 3.6445785, -12.1119795, 3.6586888, -13.8290482, 13.8295593
1: -3.6439943, 7.3791285, -3.6523952, 7.3883204, -8.4754410, 8.4535637
2: -0.7226265, 13.4179630, -0.7342246, 13.4175510, -13.4100304, 13.4078827
3: -1.1080565, 11.2868023, -1.1202095, 11.2882404, -11.9732056, 11.9965401
4: -11.0741081, 5.4576731, -11.0865374, 5.4741306, -14.6231537, 14.6175346
5: 1.8731518, 17.7289772, 1.8536777, 17.7291851, -15.8560333, 15.8752995
6: -39.8435173, -18.2786579, -39.8862152, -18.2351856, -15.0739021, 15.0951500
7: -3.5337095, 12.2272806, -3.5596969, 12.2398558, -13.5640488, 13.5505104
8: -6.6842670, 8.5441971, -6.7011557, 8.5507107, -12.0515060, 12.0699577
9: -4.7512922, 11.6721497, -4.7792249, 11.7117100, -12.9612961, 12.9529762
10: 1.3500996, 25.7170658, 1.3265085, 25.7470074, -20.8789444, 20.8531342
11: -11.4842749, 4.2816792, -11.4955025, 4.2892642, -15.7735386, 15.7771816
12: -11.8781500, 9.8023014, -11.8835812, 9.8292627, -14.9747200, 14.9599953
13: -18.5515060, 6.6838484, -18.5540752, 6.7195082, -16.6177597, 16.5350189
14: 5.0061083, 36.3590813, 4.9637766, 36.4234238, -26.6824493, 26.6346970
15: -8.6493969, 9.2116642, -8.6876841, 9.2474251, -17.8968220, 17.8993492
16: -16.6974907, 2.5236676, -16.7247505, 2.5340378, -14.7531357, 14.7710304
17: 6.2430949, 30.6166420, 6.2121158, 30.6409645, -17.1345406, 17.1776047
18: -14.3418941, 5.1095042, -14.3610096, 5.1168995, -14.3547859, 14.3546829
19: -20.2452469, -4.3384924, -20.2601051, -4.3262925, -14.4953804, 14.5003624
20: -2.3935177, 11.2077103, -2.4095845, 11.2209578, -12.5815392, 12.5867081
21: -11.0412054, 3.2416923, -11.0529671, 3.2495654, -14.2907705, 14.2946596
22: -3.6678138, 13.0655174, -3.6828113, 13.0906057, -14.8683243, 14.8883133
23: -14.5448122, 0.3035769, -14.5752554, 0.3231602, -14.2633247, 14.2675972
24: -19.9197330, -5.1284056, -19.9261951, -5.1183209, -9.2488022, 9.2369728
25: -5.4308844, 10.8301649, -5.4409142, 10.8472347, -13.7452507, 13.7604141
26: -20.9698734, 1.1515973, -20.9848099, 1.1824081, -19.2407150, 19.2600861
27: -15.9875221, 2.1667907, -16.0107040, 2.1708245, -13.1534386, 13.2054520
28: -12.7578716, 4.6011686, -12.7927837, 4.6221833, -17.3800545, 17.3939514
29: -5.5515299, 11.8389044, -5.5788040, 11.8629055, -14.8593369, 14.9018478
30: -10.0355291, 6.1938124, -10.0426693, 6.2053318, -13.5184364, 13.5221825
31: -10.9291449, 6.9496717, -10.9570465, 6.9493389, -14.5926361, 14.6228218
32: -24.8781681, -4.5979195, -24.9066200, -4.5766602, -13.2319107, 13.2662468
33: -69.2517395, -40.1548767, -69.3093948, -40.1036263, -16.5793381, 16.5632057
34: -53.7132645, -30.9414616, -53.7597580, -30.9104385, -14.0719185, 14.1115189
35: -47.7899017, -26.0950012, -47.8194199, -26.0645180, -12.9735413, 12.9521370
36: -42.8007507, -19.3075581, -42.8237076, -19.2768402, -15.0443840, 15.0431442
37: -86.6474304, -55.5735016, -86.6669159, -55.5485077, -18.8779526, 18.8630600
38: -52.8702049, -24.3871269, -52.9293289, -24.3271179, -18.2634659, 18.2421265
39: -76.5009079, -44.6713524, -76.5408707, -44.6245804, -16.0181694, 15.9780464
40: -67.2006607, -43.5339851, -67.2466736, -43.5353012, -14.2426949, 14.3117676
41: -55.3917542, -32.9786224, -55.4315453, -32.9614487, -16.6148987, 16.6639938
42: -29.4515743, -9.8900442, -29.4651375, -9.8928804, -17.2068710, 17.2440262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4719453
time: 25.51 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658443, upper bound: 12.5062244
time: 6.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.0963612, 3.6530585, -12.1131067, 3.6634417, -13.8416252, 13.8366585
1: -3.6478710, 7.3861523, -3.6526592, 7.3920708, -8.4834442, 8.4604034
2: -0.7282094, 13.4251947, -0.7347224, 13.4216356, -13.4190826, 13.4134827
3: -1.1140453, 11.2956448, -1.1206673, 11.2932110, -11.9837379, 12.0020962
4: -11.0778179, 5.4592218, -11.0878725, 5.4747992, -14.6359787, 14.6195946
5: 1.8618431, 17.7399368, 1.8528633, 17.7355843, -15.8737411, 15.8870735
6: -39.8543091, -18.2702675, -39.8922882, -18.2348480, -15.0797691, 15.1100998
7: -3.5422094, 12.2360401, -3.5605102, 12.2449150, -13.5761795, 13.5569534
8: -6.6966529, 8.5596972, -6.7017889, 8.5596924, -12.0732918, 12.0801640
9: -4.7566156, 11.6786613, -4.7797737, 11.7152061, -12.9723320, 12.9605789
10: 1.3380241, 25.7310829, 1.3250823, 25.7547779, -20.8997879, 20.8596344
11: -11.4870701, 4.2830715, -11.4964180, 4.2897897, -15.7768593, 15.7794895
12: -11.8845692, 9.8105412, -11.8873205, 9.8299751, -14.9779053, 14.9704742
13: -18.5582809, 6.6949530, -18.5580006, 6.7212534, -16.6233521, 16.5440292
14: 4.9837828, 36.3823700, 4.9612074, 36.4369736, -26.7169342, 26.6546021
15: -8.6556721, 9.2208385, -8.6887722, 9.2520275, -17.9076996, 17.9096107
16: -16.7075062, 2.5343285, -16.7256622, 2.5400143, -14.7699280, 14.7781868
17: 6.2287269, 30.6306705, 6.2106428, 30.6491985, -17.1573563, 17.1825333
18: -14.3521070, 5.1105061, -14.3664494, 5.1172104, -14.3657913, 14.3623810
19: -20.2565918, -4.3333492, -20.2660141, -4.3261938, -14.5026245, 14.5097847
20: -2.4034595, 11.2134686, -2.4148695, 11.2211847, -12.5879745, 12.5962257
21: -11.0524635, 3.2449942, -11.0587940, 3.2497962, -14.3022594, 14.3037882
22: -3.6779311, 13.0703173, -3.6882882, 13.0909901, -14.8743172, 14.8973618
23: -14.5481434, 0.3059976, -14.5764494, 0.3239326, -14.2685318, 14.2774353
24: -19.9263954, -5.1240907, -19.9293365, -5.1180654, -9.2551994, 9.2462730
25: -5.4363017, 10.8323765, -5.4430752, 10.8477592, -13.7510910, 13.7666550
26: -20.9807053, 1.1579523, -20.9906998, 1.1829665, -19.2497978, 19.2729950
27: -15.9913139, 2.1678286, -16.0122719, 2.1710596, -13.1587601, 13.2085915
28: -12.7642097, 4.6042676, -12.7958117, 4.6229391, -17.3871498, 17.4000797
29: -5.5543318, 11.8405495, -5.5799198, 11.8633804, -14.8629150, 14.9050674
30: -10.0393238, 6.1986847, -10.0438843, 6.2075624, -13.5242844, 13.5292168
31: -10.9430761, 6.9506230, -10.9635820, 6.9493799, -14.6042442, 14.6302795
32: -24.8893051, -4.5880084, -24.9129391, -4.5763245, -13.2355766, 13.2830582
33: -69.2778625, -40.1302795, -69.3238068, -40.1026917, -16.5848923, 16.6005440
34: -53.7280502, -30.9257011, -53.7685738, -30.9098396, -14.0769882, 14.1358223
35: -47.8098373, -26.0737801, -47.8311806, -26.0639553, -12.9760399, 12.9844666
36: -42.8209076, -19.2830944, -42.8357391, -19.2761555, -15.0451088, 15.0803452
37: -86.6662979, -55.5555420, -86.6775970, -55.5472984, -18.8826485, 18.8887634
38: -52.8997421, -24.3557549, -52.9468384, -24.3265190, -18.2699738, 18.2927094
39: -76.5286865, -44.6435585, -76.5567932, -44.6238632, -16.0180855, 16.0218925
40: -67.2093048, -43.5340500, -67.2507172, -43.5348434, -14.2544174, 14.3178711
41: -55.4052505, -32.9635696, -55.4395332, -32.9606857, -16.6215134, 16.6870880
42: -29.4533081, -9.8875713, -29.4657097, -9.8921280, -17.2106438, 17.2571945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4833536
time: 7.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5658523, upper bound: 12.5173194
time: 20.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.0892639, 3.6474304, -12.1297779, 3.6627100, -13.8318329, 13.8524551
1: -3.6434460, 7.3812051, -3.6641688, 7.3911157, -8.4643822, 8.4583397
2: -0.7240003, 13.4215584, -0.7597293, 13.4259949, -13.4127655, 13.4360161
3: -1.1076989, 11.2902565, -1.1308403, 11.2948513, -11.9737930, 12.0062103
4: -11.0771713, 5.4595790, -11.1031876, 5.4826431, -14.6233521, 14.6298981
5: 1.8731513, 17.7314644, 1.8338804, 17.7354336, -15.8622818, 15.8975840
6: -39.8429718, -18.2887440, -39.8717117, -18.2494812, -15.0859871, 15.0611725
7: -3.5337658, 12.2324152, -3.5886650, 12.2508469, -13.5702591, 13.5928078
8: -6.6848640, 8.5469828, -6.7185049, 8.5573654, -12.0425110, 12.0912094
9: -4.7395649, 11.6727924, -4.7603283, 11.6958580, -12.9414825, 12.9495277
10: 1.3655109, 25.7175636, 1.3538938, 25.7258987, -20.8441467, 20.8365402
11: -11.4802284, 4.2825074, -11.4881449, 4.2850971, -15.7653255, 15.7706528
12: -11.8780499, 9.8026876, -11.8831692, 9.8504400, -15.0031815, 14.9655190
13: -18.5368156, 6.6823401, -18.5319195, 6.7145081, -16.5572777, 16.5611610
14: 5.0237856, 36.3584557, 4.9918108, 36.3836021, -26.6188354, 26.6303482
15: -8.6437159, 9.2096262, -8.6792154, 9.2418489, -17.8855648, 17.8888416
16: -16.6852837, 2.5305912, -16.7202530, 2.5404384, -14.7484207, 14.7578850
17: 6.2453756, 30.6165543, 6.2150750, 30.6465797, -17.1530724, 17.1769714
18: -14.3455582, 5.1101723, -14.3741808, 5.1206684, -14.3526382, 14.3563538
19: -20.2454147, -4.3421583, -20.2620487, -4.3330803, -14.4974365, 14.4959221
20: -2.3931935, 11.1996975, -2.4041207, 11.2101650, -12.5782318, 12.5787849
21: -11.0448809, 3.2408233, -11.0609798, 3.2616813, -14.3065624, 14.3018036
22: -3.6746237, 13.0611343, -3.6944635, 13.1146746, -14.9022675, 14.8842812
23: -14.5444365, 0.2997091, -14.5767326, 0.3172979, -14.2647552, 14.2597961
24: -19.9196434, -5.1288223, -19.9281597, -5.1167455, -9.2543526, 9.2384758
25: -5.4350481, 10.8296223, -5.4513168, 10.8681698, -13.7762299, 13.7614136
26: -20.9805794, 1.1506567, -21.0054626, 1.2209785, -19.2864189, 19.2662354
27: -15.9871445, 2.1594112, -16.0067787, 2.1596775, -13.1668510, 13.1825447
28: -12.7585659, 4.5942173, -12.7866917, 4.6106977, -17.3692627, 17.3809090
29: -5.5576878, 11.8380604, -5.5926981, 11.8943920, -14.8986588, 14.8986130
30: -10.0335770, 6.1937318, -10.0401859, 6.2127242, -13.5271225, 13.5205193
31: -10.9280386, 6.9466481, -10.9574938, 6.9455686, -14.5900726, 14.6161118
32: -24.8774109, -4.6046047, -24.8877525, -4.5858450, -13.2385902, 13.2313118
33: -69.2514648, -40.1659698, -69.2793579, -40.1225891, -16.5728607, 16.5306892
34: -53.7131805, -30.9521618, -53.7267380, -30.9294968, -14.0708542, 14.0616989
35: -47.7822037, -26.1017494, -47.7881012, -26.0835037, -12.9632759, 12.9236755
36: -42.7923737, -19.3188705, -42.7849503, -19.2990532, -15.0353203, 15.0105209
37: -86.6433487, -55.5779953, -86.6518860, -55.5573883, -18.8816719, 18.8645859
38: -52.8708076, -24.4019928, -52.8943787, -24.3534431, -18.2649879, 18.2043076
39: -76.4934387, -44.6764069, -76.5151672, -44.6354446, -16.0110741, 15.9624100
40: -67.2002106, -43.5350075, -67.2533722, -43.5346680, -14.2576981, 14.2810802
41: -55.3911591, -32.9874878, -55.4091644, -32.9749298, -16.6305847, 16.6154175
42: -29.4506340, -9.9011555, -29.4457493, -9.9104996, -17.2291412, 17.2030334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4252845
time: 39.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4605187
time: 8.58 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.0969963, 3.6559124, -12.1309080, 3.6674662, -13.8444138, 13.8595467
1: -3.6473141, 7.3882141, -3.6644511, 7.3948674, -8.4723778, 8.4651966
2: -0.7295770, 13.4287720, -0.7602127, 13.4300947, -13.4218788, 13.4416389
3: -1.1136951, 11.2990980, -1.1312931, 11.2998314, -11.9843292, 12.0117989
4: -11.0808468, 5.4611330, -11.1045189, 5.4833002, -14.6361618, 14.6319122
5: 1.8618817, 17.7424278, 1.8330755, 17.7418137, -15.8799324, 15.9093523
6: -39.8536987, -18.2803230, -39.8778076, -18.2491188, -15.0918694, 15.0761490
7: -3.5422754, 12.2411757, -3.5894547, 12.2559366, -13.5824051, 13.5992432
8: -6.6972556, 8.5624771, -6.7191701, 8.5663872, -12.0642929, 12.1014137
9: -4.7448697, 11.6793566, -4.7608719, 11.6993885, -12.9524918, 12.9571533
10: 1.3534474, 25.7315674, 1.3524652, 25.7336864, -20.8649445, 20.8430862
11: -11.4829998, 4.2838907, -11.4890480, 4.2856236, -15.7686234, 15.7729387
12: -11.8844719, 9.8109341, -11.8869171, 9.8511314, -15.0063553, 14.9760170
13: -18.5434990, 6.6934953, -18.5358162, 6.7163010, -16.5628738, 16.5701485
14: 5.0014505, 36.3816528, 4.9891911, 36.3971024, -26.6533661, 26.6502609
15: -8.6499882, 9.2187805, -8.6803150, 9.2464848, -17.8964729, 17.8990955
16: -16.6952648, 2.5412529, -16.7211418, 2.5464168, -14.7652206, 14.7650490
17: 6.2310028, 30.6305962, 6.2135620, 30.6548309, -17.1758804, 17.1819000
18: -14.3557701, 5.1111388, -14.3796291, 5.1209884, -14.3636513, 14.3640594
19: -20.2567501, -4.3370228, -20.2679749, -4.3329892, -14.5046959, 14.5053635
20: -2.4031587, 11.2054243, -2.4094064, 11.2103996, -12.5846825, 12.5883179
21: -11.0561399, 3.2441056, -11.0668125, 3.2619085, -14.3180485, 14.3109179
22: -3.6847322, 13.0659256, -3.6999400, 13.1150751, -14.9082489, 14.8932915
23: -14.5477886, 0.3021388, -14.5778971, 0.3180575, -14.2699356, 14.2696190
24: -19.9263382, -5.1245236, -19.9313145, -5.1164579, -9.2607574, 9.2477951
25: -5.4404774, 10.8318195, -5.4534492, 10.8686638, -13.7820816, 13.7676849
26: -20.9914303, 1.1569991, -21.0112801, 1.2215314, -19.2955360, 19.2791901
27: -15.9909210, 2.1604238, -16.0083694, 2.1599040, -13.1721840, 13.1857033
28: -12.7649088, 4.5973186, -12.7897511, 4.6114788, -17.3763885, 17.3870697
29: -5.5604782, 11.8396978, -5.5938301, 11.8948917, -14.9022026, 14.9018555
30: -10.0373821, 6.1986198, -10.0414200, 6.2149787, -13.5329590, 13.5275726
31: -10.9419727, 6.9475999, -10.9640026, 6.9456244, -14.6016731, 14.6235580
32: -24.8885403, -4.5946684, -24.8941269, -4.5855188, -13.2422523, 13.2481232
33: -69.2776489, -40.1413727, -69.2937317, -40.1216660, -16.5784302, 16.5680542
34: -53.7279510, -30.9364090, -53.7355804, -30.9289017, -14.0759239, 14.0859833
35: -47.8021431, -26.0805550, -47.7998161, -26.0828972, -12.9657593, 12.9560318
36: -42.8125076, -19.2943745, -42.7969704, -19.2983303, -15.0360603, 15.0477486
37: -86.6622009, -55.5600662, -86.6626434, -55.5561752, -18.8863449, 18.8903122
38: -52.9004250, -24.3706532, -52.9118042, -24.3527737, -18.2715034, 18.2548676
39: -76.5212097, -44.6485977, -76.5311432, -44.6347198, -16.0109787, 16.0062675
40: -67.2088318, -43.5350418, -67.2574387, -43.5342102, -14.2694054, 14.2871857
41: -55.4047089, -32.9724846, -55.4171638, -32.9742012, -16.6371994, 16.6385307
42: -29.4523697, -9.8987265, -29.4463329, -9.9097338, -17.2329369, 17.2162132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655697, upper bound: 12.4368035
time: 6.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655697, upper bound: 12.4720355
time: 5.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.0877390, 3.6476817, -12.1284523, 3.6681414, -13.8367386, 13.8524323
1: -3.6402006, 7.3815036, -3.6597617, 7.3947363, -8.4759941, 8.4605293
2: -0.7227238, 13.4217682, -0.7586621, 13.4279661, -13.4147530, 13.4366608
3: -1.1076547, 11.2871742, -1.1309329, 11.2924938, -11.9756317, 12.0121956
4: -11.0734501, 5.4598651, -11.0982714, 5.4836512, -14.6233292, 14.6277542
5: 1.8731976, 17.7303391, 1.8334579, 17.7353611, -15.8621635, 15.8968811
6: -39.8430862, -18.2935772, -39.8790054, -18.2559624, -15.0854797, 15.0655098
7: -3.5332029, 12.2324524, -3.5901604, 12.2541037, -13.5797958, 13.5937119
8: -6.6830997, 8.5471497, -6.7180519, 8.5584621, -12.0442200, 12.0920525
9: -4.7415671, 11.6727619, -4.7681432, 11.7161293, -12.9554825, 12.9539146
10: 1.3612723, 25.7174110, 1.3398647, 25.7450943, -20.8638687, 20.8482666
11: -11.4819088, 4.2818971, -11.4954824, 4.2910500, -15.7729588, 15.7773800
12: -11.8782501, 9.8024044, -11.8888826, 9.8554230, -15.0070724, 14.9705238
13: -18.5371838, 6.6822767, -18.5345993, 6.7251797, -16.5612030, 16.5630875
14: 5.0186939, 36.3584747, 4.9734316, 36.4216042, -26.6544037, 26.6455765
15: -8.6409950, 9.2101040, -8.6764317, 9.2462740, -17.8872681, 17.8865356
16: -16.6869545, 2.5306911, -16.7307739, 2.5532961, -14.7672577, 14.7651749
17: 6.2444386, 30.6165695, 6.2066364, 30.6532249, -17.1510429, 17.1906357
18: -14.3417320, 5.1104794, -14.3704720, 5.1223564, -14.3550224, 14.3592854
19: -20.2441635, -4.3419256, -20.2635880, -4.3319120, -14.4971619, 14.5015602
20: -2.3933911, 11.1985617, -2.4086771, 11.2091007, -12.5779419, 12.5802155
21: -11.0431309, 3.2407517, -11.0629940, 3.2623122, -14.3054428, 14.3037453
22: -3.6744854, 13.0616302, -3.7010238, 13.1188955, -14.9057541, 14.8963890
23: -14.5448465, 0.2997313, -14.5812311, 0.3185918, -14.2668495, 14.2623177
24: -19.9181442, -5.1288767, -19.9268379, -5.1163497, -9.2557716, 9.2400513
25: -5.4346733, 10.8296537, -5.4536018, 10.8696327, -13.7756500, 13.7687988
26: -20.9774437, 1.1507773, -21.0044651, 1.2231326, -19.2872849, 19.2700043
27: -15.9870205, 2.1604493, -16.0166779, 2.1631644, -13.1706963, 13.1840477
28: -12.7587271, 4.5950446, -12.7950230, 4.6140652, -17.3727913, 17.3900681
29: -5.5577679, 11.8377619, -5.5950289, 11.8959980, -14.8995285, 14.9059525
30: -10.0345039, 6.1934605, -10.0453711, 6.2191997, -13.5351410, 13.5251732
31: -10.9278193, 6.9472194, -10.9651871, 6.9476271, -14.5915527, 14.6227570
32: -24.8775063, -4.6111183, -24.8977394, -4.5946436, -13.2383957, 13.2412109
33: -69.2511520, -40.1615028, -69.3110123, -40.1092491, -16.5815353, 16.5641098
34: -53.7131729, -30.9490643, -53.7541656, -30.9185314, -14.0794182, 14.0882416
35: -47.7820892, -26.0999908, -47.8056259, -26.0759697, -12.9702148, 12.9432449
36: -42.7922592, -19.3174515, -42.8054199, -19.2928810, -15.0391464, 15.0270157
37: -86.6430435, -55.5763855, -86.6625595, -55.5518532, -18.8859482, 18.8718491
38: -52.8707542, -24.4000492, -52.9209137, -24.3457031, -18.2697830, 18.2195892
39: -76.4932480, -44.6741142, -76.5316696, -44.6285896, -16.0164452, 15.9786110
40: -67.2000580, -43.5320740, -67.2744598, -43.5276566, -14.2621651, 14.2795620
41: -55.3912430, -32.9849319, -55.4329987, -32.9667244, -16.6336136, 16.6274796
42: -29.4508553, -9.9092789, -29.4519672, -9.9226112, -17.2271309, 17.2070389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=90, inp2_unstable=91, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=95, inp2_unstable=95, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=25, inp2_unstable=25, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 977
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 675
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1291
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 1286
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1285
type: B, layer: 1, pos: 1292
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1294
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1295
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 947

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 952

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4670394
time: 14.23 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -12.5655609, upper bound: 12.5022532
time: 13.51 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 30.07 seconds
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646801, upper bound: 12.3937736
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4276022
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4050795
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4386391
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4351207
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646801, upper bound: 12.4684815
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4462526
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5646894, upper bound: 12.4794225
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4302670
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4650409
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4417634
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4762515
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658443, upper bound: 12.4719453
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658443, upper bound: 12.5062244
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658523, upper bound: 12.4833536
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5658523, upper bound: 12.5173194
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4252845
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4605187
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655697, upper bound: 12.4368035
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655697, upper bound: 12.4720355
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655609, upper bound: 12.4670394
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 30.07
Output dim: 14, lower bound: -12.5655609, upper bound: 12.5022532
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5671317, upper bound: 12.5152519
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5681592, upper bound: 12.4982333
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5097549
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5681592, upper bound: 12.5399798
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5681669, upper bound: 12.5515008
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4449258
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4558591
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5664303, upper bound: 12.4856821
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5664349, upper bound: 12.4966142
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5674973, upper bound: 12.4826541
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5675011, upper bound: 12.4937356
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5674973, upper bound: 12.5236515
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5675011, upper bound: 12.5346328
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5369722, upper bound: 12.5673364
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5535439, upper bound: 12.5682675
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5672610, upper bound: 12.4788643
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5672643, upper bound: 12.4902819
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5672610, upper bound: 12.5204256
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5672643, upper bound: 12.5317417
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5515242, upper bound: 12.5673657
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5150487
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5265653
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5682939, upper bound: 12.5567951
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 30.07
Output dim: 14, lower bound: -12.5682974, upper bound: 12.5682967

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 24.57 + 1779.26 = 1803.83 seconds
