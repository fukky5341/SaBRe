## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 7200 seconds
Split limit: 100
Threshold: 45.0013539996


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=212, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-28.1366806, 17.0328903, -28.1366806, 17.0328903, -44.3955650, 44.3955612)
1: (-13.6887369, 17.0437737, -13.6887369, 17.0437737, -30.7325096, 30.7325096)
2: (-14.1001348, 21.6170921, -14.1001348, 21.6170921, -35.4909210, 35.4909248)
3: (-12.9170971, 23.4013729, -12.9170971, 23.4013729, -36.3184700, 36.3184700)
4: (-21.5850239, 18.4471970, -21.5850239, 18.4471970, -40.0322189, 40.0322189)
5: (-12.0244370, 22.8004417, -12.0244370, 22.8004417, -34.8248787, 34.8248787)
6: (-50.6833725, -3.5547500, -50.6833725, -3.5547500, -40.5662613, 40.5662537)
7: (-16.4142704, 18.4128113, -16.4142704, 18.4128113, -34.8270798, 34.8270798)
8: (-18.3477325, 21.2852058, -18.3477325, 21.2852058, -39.6329384, 39.6329384)
9: (-16.7399044, 23.2325306, -16.7399044, 23.2325306, -38.6461983, 38.6462021)
10: (-24.3248863, 38.4713745, -24.3248863, 38.4713745, -61.8275909, 61.8275909)
11: (-24.7690277, 17.6132698, -24.7690277, 17.6132698, -42.3822975, 42.3822975)
12: (-28.6516190, 20.1307869, -28.6516190, 20.1307869, -46.9375648, 46.9375610)
13: (-32.9485664, 28.7780704, -32.9485664, 28.7780704, -61.7266388, 61.7266388)
14: (-23.5383434, 39.1661148, -23.5383434, 39.1661148, -60.1073761, 60.1073761)
15: (-18.9612942, 25.8496895, -18.9612942, 25.8496895, -44.8109818, 44.8109818)
16: (-32.7486305, 19.8674545, -32.7486305, 19.8674545, -52.6160851, 52.6160851)
17: (-17.7939606, 38.4373703, -17.7939606, 38.4373703, -55.2744904, 55.2744942)
18: (-25.7858334, 19.6272316, -25.7858334, 19.6272316, -45.4130630, 45.4130630)
19: (-26.4100552, 12.5144444, -26.4100552, 12.5144444, -38.9244995, 38.9244995)
20: (-21.0849285, 20.4594860, -21.0849285, 20.4594860, -41.5444145, 41.5444145)
21: (-25.6940289, 18.9120598, -25.6940289, 18.9120598, -44.6060867, 44.6060867)
22: (-22.1009121, 24.5458031, -22.1009121, 24.5458031, -46.6467133, 46.6467133)
23: (-21.6938934, 17.5081844, -21.6938934, 17.5081844, -39.2020798, 39.2020798)
24: (-32.1186867, 11.9165916, -32.1186867, 11.9165916, -44.0352783, 44.0352783)
25: (-18.1028080, 25.4411068, -18.1028080, 25.4411068, -43.5439148, 43.5439148)
26: (-29.2344494, 26.9800606, -29.2344494, 26.9800606, -56.2145081, 56.2145081)
27: (-32.1001968, 16.5631962, -32.1001968, 16.5631962, -47.8254776, 47.8254814)
28: (-21.5237122, 21.7201538, -21.5237122, 21.7201538, -43.2438660, 43.2438660)
29: (-23.6920109, 22.2411366, -23.6920109, 22.2411366, -45.9331474, 45.9331474)
30: (-29.6156273, 16.8732967, -29.6156273, 16.8732967, -45.9555359, 45.9555321)
31: (-26.3471394, 19.1131210, -26.3471394, 19.1131210, -45.4602585, 45.4602585)
32: (-42.2200890, 8.5029144, -42.2200890, 8.5029144, -47.5713501, 47.5713501)
33: (-72.3290024, -5.5757275, -72.3290024, -5.5757275, -61.3344040, 61.3344116)
34: (-56.4624290, -5.4563274, -56.4624290, -5.4563274, -43.6207123, 43.6207085)
35: (-50.1139221, 0.0764503, -50.1139221, 0.0764503, -48.2550659, 48.2550697)
36: (-47.7510834, 4.9805908, -47.7510834, 4.9805908, -52.0323868, 52.0323792)
37: (-83.6398163, -17.4274712, -83.6398163, -17.4274712, -58.4293747, 58.4293747)
38: (-58.6122246, 3.2742290, -58.6122246, 3.2742290, -61.3017960, 61.3017883)
39: (-78.9286499, -11.5559139, -78.9286499, -11.5559139, -65.3207550, 65.3207550)
40: (-67.6456757, -18.3084507, -67.6456757, -18.3084507, -41.1823769, 41.1823730)
41: (-55.1725731, -6.8072844, -55.1725731, -6.8072844, -42.2860641, 42.2860641)
42: (-33.9530563, 6.8301487, -33.9530563, 6.8301487, -37.7043839, 37.7043839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.98 + 61.27 = 64.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 14, lower bound: -45.0464004, upper bound: 45.0464004

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1745

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0014336, upper bound: 45.0418528
time: 44.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0418525, upper bound: 45.0418528
time: 52.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 97.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 97.09
Output dim: 14, lower bound: -45.0014336, upper bound: 45.0418528
IS_A2, status: Status.UNKNOWN, split count: 1, time: 97.09
Output dim: 14, lower bound: -45.0418525, upper bound: 45.0418528

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -28.0785866, 17.0276642, -28.1295471, 17.0322666, -44.3390808, 44.3837395
1: -13.6553373, 17.0398369, -13.6846752, 17.0432968, -30.6986351, 30.7245121
2: -14.0668850, 21.6131172, -14.0961142, 21.6166115, -35.4579773, 35.4828110
3: -12.8879223, 23.3946152, -12.9134655, 23.4005547, -36.2884750, 36.3080826
4: -21.5543823, 18.4361305, -21.5812931, 18.4458656, -40.0002480, 40.0174255
5: -11.9876451, 22.7945538, -12.0199585, 22.7997322, -34.7873764, 34.8145142
6: -50.6792603, -3.6119490, -50.6828918, -3.5616026, -40.5553665, 40.5088501
7: -16.3776226, 18.4082813, -16.4098396, 18.4122543, -34.7898788, 34.8181229
8: -18.2852898, 21.2787819, -18.3401451, 21.2844353, -39.5697250, 39.6189270
9: -16.7016907, 23.2272606, -16.7352295, 23.2319145, -38.6070709, 38.6360207
10: -24.2818031, 38.4598541, -24.3195381, 38.4699745, -61.7799072, 61.8097153
11: -24.7623730, 17.5974503, -24.7682171, 17.6113300, -42.3737030, 42.3656693
12: -28.6466389, 20.0931358, -28.6510220, 20.1262245, -46.9228706, 46.9022102
13: -32.9338722, 28.7582970, -32.9467850, 28.7757072, -61.7095795, 61.7050819
14: -23.4645996, 39.1608887, -23.5293465, 39.1654892, -60.0326385, 60.0925331
15: -18.9196854, 25.8414459, -18.9562550, 25.8487034, -44.7683868, 44.7976990
16: -32.7191277, 19.8578072, -32.7451057, 19.8662949, -52.5854225, 52.6029129
17: -17.7515945, 38.4318504, -17.7887554, 38.4367180, -55.2286911, 55.2636490
18: -25.7778053, 19.6155930, -25.7848759, 19.6258163, -45.4036217, 45.4004669
19: -26.4008656, 12.4827328, -26.4089546, 12.5106344, -38.9114990, 38.8916855
20: -21.0752106, 20.4297237, -21.0837555, 20.4558907, -41.5311012, 41.5134811
21: -25.6815605, 18.8699837, -25.6925201, 18.9070263, -44.5885849, 44.5625038
22: -22.0914516, 24.5178509, -22.0997887, 24.5422802, -46.6337318, 46.6176376
23: -21.6835098, 17.4969234, -21.6926441, 17.5068092, -39.1903191, 39.1895676
24: -32.1110268, 11.8960171, -32.1177483, 11.9141388, -44.0251656, 44.0137634
25: -18.0906982, 25.4198856, -18.1013603, 25.4385185, -43.5292168, 43.5212479
26: -29.2231255, 26.9536972, -29.2330818, 26.9768448, -56.1999702, 56.1867790
27: -32.0904770, 16.5418015, -32.0990372, 16.5605164, -47.8121567, 47.7928581
28: -21.5145416, 21.6959553, -21.5226021, 21.7172108, -43.2317505, 43.2185593
29: -23.6858139, 22.2263947, -23.6912479, 22.2393551, -45.9251709, 45.9176407
30: -29.6069965, 16.8462353, -29.6145744, 16.8699226, -45.9436874, 45.9305534
31: -26.3317280, 19.0737343, -26.3452778, 19.1082134, -45.4399414, 45.4190140
32: -42.2140656, 8.4621649, -42.2193604, 8.4979620, -47.5607185, 47.5300865
33: -72.3186569, -5.6475868, -72.3277359, -5.5843124, -61.3142090, 61.2611313
34: -56.4560890, -5.5112753, -56.4616623, -5.4628820, -43.6073303, 43.5666924
35: -50.1060715, 0.0141821, -50.1129951, 0.0690165, -48.2391357, 48.1912537
36: -47.7430801, 4.9164848, -47.7501221, 4.9729519, -52.0164795, 51.9673615
37: -83.6305389, -17.4683533, -83.6387024, -17.4323273, -58.4140930, 58.3864250
38: -58.5986595, 3.1987410, -58.6106415, 3.2651052, -61.2785339, 61.2239609
39: -78.9165878, -11.6259365, -78.9272385, -11.5642557, -65.3004913, 65.2488708
40: -67.6368866, -18.3366928, -67.6446075, -18.3118858, -41.1694107, 41.1548805
41: -55.1686134, -6.8521204, -55.1721077, -6.8127346, -42.2757759, 42.2385483
42: -33.9480934, 6.8120508, -33.9524536, 6.8279161, -37.6971054, 37.6835060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -45.0007745, upper bound: 44.9103827
time: 33.24 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0007745, upper bound: 45.0411942
time: 61.89 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -28.1654778, 17.0936012, -28.1333084, 17.0321808, -44.4194412, 44.4557304
1: -13.6918259, 17.0841217, -13.6869678, 17.0431595, -30.7349854, 30.7710896
2: -14.1023474, 21.6603775, -14.0978718, 21.6165657, -35.4904633, 35.5358620
3: -12.9143963, 23.4332314, -12.9141960, 23.4005089, -36.3149033, 36.3474274
4: -21.5871124, 18.4573479, -21.5820427, 18.4450188, -40.0321312, 40.0393906
5: -12.0228596, 22.8304615, -12.0215054, 22.7997475, -34.8226089, 34.8519669
6: -50.7865219, -3.5527430, -50.6826553, -3.5600739, -40.6737404, 40.5643616
7: -16.4143028, 18.4420853, -16.4114685, 18.4122658, -34.8265686, 34.8535538
8: -18.3440475, 21.3485413, -18.3429565, 21.2845936, -39.6286392, 39.6914978
9: -16.7410851, 23.2847805, -16.7354546, 23.2317429, -38.6440506, 38.6919060
10: -24.3327827, 38.5408516, -24.3205795, 38.4701729, -61.8326645, 61.8987350
11: -24.7967186, 17.6061707, -24.7679520, 17.6087856, -42.4055023, 42.3741226
12: -28.6652908, 20.1401253, -28.6508732, 20.1277390, -46.9399261, 46.9467354
13: -32.9573975, 28.8062325, -32.9459724, 28.7760906, -61.7334900, 61.7522049
14: -23.5632839, 39.2395897, -23.5314522, 39.1654396, -60.1303558, 60.1746597
15: -18.9703102, 25.8946800, -18.9571304, 25.8485527, -44.8188629, 44.8518105
16: -32.7572784, 19.9182510, -32.7449493, 19.8665657, -52.6238441, 52.6632004
17: -17.8073788, 38.5059662, -17.7902374, 38.4365120, -55.2837143, 55.3434525
18: -25.8480034, 19.6272182, -25.7851181, 19.6239719, -45.4719772, 45.4123383
19: -26.4898777, 12.5139284, -26.4089088, 12.5125694, -39.0024490, 38.9228363
20: -21.1653271, 20.4629745, -21.0838242, 20.4579239, -41.6232529, 41.5467987
21: -25.7888298, 18.9123077, -25.6926441, 18.9096336, -44.6984634, 44.6049500
22: -22.1753330, 24.5448380, -22.1000099, 24.5441189, -46.7194519, 46.6448479
23: -21.7197685, 17.5163116, -21.6921272, 17.5071869, -39.2269554, 39.2084389
24: -32.2105293, 11.9124937, -32.1179771, 11.9126625, -44.1231918, 44.0304718
25: -18.1446266, 25.4342117, -18.1016216, 25.4357700, -43.5803986, 43.5358353
26: -29.2752285, 26.9732780, -29.2332897, 26.9739037, -56.2491302, 56.2065659
27: -32.1544495, 16.5602531, -32.0991974, 16.5595226, -47.9020233, 47.8149261
28: -21.5623741, 21.7166328, -21.5226479, 21.7148094, -43.2771835, 43.2392807
29: -23.7537575, 22.2376785, -23.6912727, 22.2384148, -45.9921722, 45.9289513
30: -29.6934776, 16.8781986, -29.6145554, 16.8688736, -46.0245590, 45.9621544
31: -26.4508858, 19.1127148, -26.3453007, 19.1109104, -45.5617981, 45.4580154
32: -42.2887573, 8.5100861, -42.2192574, 8.5000544, -47.6375885, 47.5762863
33: -72.4688416, -5.5806952, -72.3279495, -5.5807915, -61.4680786, 61.3278275
34: -56.5523720, -5.4602680, -56.4616928, -5.4603014, -43.7115250, 43.6161804
35: -50.2278328, 0.0687656, -50.1130257, 0.0722456, -48.3647995, 48.2461090
36: -47.8437653, 4.9755468, -47.7502556, 4.9761000, -52.1223984, 52.0261307
37: -83.7435684, -17.4274349, -83.6384735, -17.4305305, -58.5314026, 58.4264526
38: -58.7573547, 3.2689791, -58.6109695, 3.2686348, -61.4406586, 61.2946167
39: -79.0809021, -11.5609818, -78.9270020, -11.5606241, -65.4699631, 65.3137054
40: -67.7307587, -18.3035908, -67.6445847, -18.3105354, -41.2663422, 41.1862602
41: -55.2303543, -6.8028049, -55.1717911, -6.8106337, -42.3438187, 42.2872581
42: -33.9617767, 6.8320913, -33.9521637, 6.8261852, -37.7112541, 37.7051163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=212, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0411941, upper bound: 44.9103827
time: 28.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0411941, upper bound: 45.0411942
time: 378.03 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 409.06 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 409.06
Output dim: 14, lower bound: -45.0007745, upper bound: 44.9103827
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 409.06
Output dim: 14, lower bound: -45.0007745, upper bound: 45.0411942
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 409.06
Output dim: 14, lower bound: -45.0411941, upper bound: 44.9103827
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 409.06
Output dim: 14, lower bound: -45.0411941, upper bound: 45.0411942

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -28.0784492, 17.0275288, -28.1284485, 17.0311375, -44.3252716, 44.3824959
1: -13.6552725, 17.0396156, -13.6842175, 17.0415039, -30.6967773, 30.7238331
2: -14.0668287, 21.6129131, -14.0956335, 21.6149502, -35.4385490, 35.4821167
3: -12.8878822, 23.3944283, -12.9131317, 23.3990822, -36.2869644, 36.3075600
4: -21.5543098, 18.4359474, -21.5806522, 18.4443474, -39.9986572, 40.0166016
5: -11.9875927, 22.7944050, -12.0196114, 22.7986069, -34.7862015, 34.8140182
6: -50.6790543, -3.6120725, -50.6812248, -3.5626292, -40.5710068, 40.5069389
7: -16.3775597, 18.4076176, -16.4093304, 18.4080353, -34.7855949, 34.8169479
8: -18.2852020, 21.2785187, -18.3395081, 21.2822304, -39.5674324, 39.6180267
9: -16.7015018, 23.2271729, -16.7337685, 23.2312126, -38.6060448, 38.6100998
10: -24.2815666, 38.4597206, -24.3175621, 38.4688721, -61.7785416, 61.7773552
11: -24.7622070, 17.5973320, -24.7668724, 17.6102982, -42.3725052, 42.3642044
12: -28.6463242, 20.0930405, -28.6485596, 20.1255646, -46.9218712, 46.8767204
13: -32.9328575, 28.7581596, -32.9384956, 28.7746658, -61.7075233, 61.6966553
14: -23.4641457, 39.1608467, -23.5255146, 39.1650925, -60.0317535, 60.0531769
15: -18.9194641, 25.8411713, -18.9545345, 25.8464088, -44.7658730, 44.7957077
16: -32.7185593, 19.8576164, -32.7403679, 19.8647480, -52.5833054, 52.5979843
17: -17.7512970, 38.4317818, -17.7863045, 38.4359970, -55.2276459, 55.2406998
18: -25.7775860, 19.6154900, -25.7832031, 19.6249771, -45.4025650, 45.3986931
19: -26.4006577, 12.4826889, -26.4072666, 12.5103073, -38.9109650, 38.8899536
20: -21.0750580, 20.4297009, -21.0825214, 20.4555931, -41.5306511, 41.5122223
21: -25.6813774, 18.8699341, -25.6910763, 18.9066143, -44.5879898, 44.5610123
22: -22.0903397, 24.5178032, -22.0907669, 24.5418816, -46.6322212, 46.6085701
23: -21.6832561, 17.4968777, -21.6905804, 17.5064087, -39.1896667, 39.1874580
24: -32.1109009, 11.8959351, -32.1168671, 11.9135008, -44.0244026, 44.0128021
25: -18.0902634, 25.4198360, -18.0977936, 25.4380970, -43.5283585, 43.5176315
26: -29.2222176, 26.9536362, -29.2257462, 26.9763641, -56.1985817, 56.1793823
27: -32.0903282, 16.5414867, -32.0978546, 16.5578995, -47.7816200, 47.7911797
28: -21.5143700, 21.6959038, -21.5211906, 21.7168198, -43.2311897, 43.2170944
29: -23.6851826, 22.2263565, -23.6872654, 22.2390099, -45.9241943, 45.9136200
30: -29.6068382, 16.8461304, -29.6132774, 16.8690834, -45.9425964, 45.9205284
31: -26.3315887, 19.0736656, -26.3440838, 19.1076660, -45.4392548, 45.4177475
32: -42.2138977, 8.4620686, -42.2179756, 8.4972057, -47.5710716, 47.5279350
33: -72.3185577, -5.6477280, -72.3270340, -5.5855198, -61.3108292, 61.2602158
34: -56.4560165, -5.5113611, -56.4610939, -5.4636364, -43.6054688, 43.5707207
35: -50.1059570, 0.0140200, -50.1121597, 0.0677233, -48.2368851, 48.2137909
36: -47.7424240, 4.9164543, -47.7446556, 4.9725046, -52.0178833, 51.9615021
37: -83.6302795, -17.4684868, -83.6367493, -17.4333076, -58.4551239, 58.3815536
38: -58.5985107, 3.1984892, -58.6092453, 3.2630339, -61.2724152, 61.2217178
39: -78.9165039, -11.6260300, -78.9263382, -11.5650740, -65.2962036, 65.2476120
40: -67.6367569, -18.3374405, -67.6434174, -18.3179646, -41.2153931, 41.1505699
41: -55.1685562, -6.8522806, -55.1715622, -6.8141565, -42.3199005, 42.2350883
42: -33.9480095, 6.8119440, -33.9517746, 6.8270330, -37.7007370, 37.6823349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0378355
time: 50.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0390997
time: 57.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -28.1592789, 17.0742207, -28.0837097, 16.9727859, -44.3525734, 44.3858948
1: -13.6890774, 17.0566635, -13.6484785, 16.9617786, -30.6508560, 30.7051430
2: -14.0980396, 21.6263046, -14.0408869, 21.5167332, -35.3838959, 35.4442215
3: -12.9113464, 23.4055748, -12.8708515, 23.3178711, -36.2292175, 36.2764282
4: -21.5800552, 18.4197388, -21.5162888, 18.3338470, -39.9139023, 39.9360275
5: -12.0198708, 22.8054523, -11.9767942, 22.7266407, -34.7465134, 34.7822456
6: -50.7752762, -3.5595069, -50.6465073, -3.5939898, -40.6148834, 40.5192261
7: -16.4105129, 18.4104004, -16.3608227, 18.3182106, -34.7287216, 34.7712250
8: -18.3374138, 21.3155365, -18.2817535, 21.1877289, -39.5251427, 39.5972900
9: -16.7075310, 23.2764835, -16.6296959, 23.1673851, -38.5443726, 38.5725327
10: -24.2851143, 38.5295029, -24.1738205, 38.3742409, -61.6877289, 61.7326584
11: -24.7735004, 17.6005669, -24.6978626, 17.5666733, -42.3401718, 42.2984314
12: -28.6103573, 20.1319466, -28.4904881, 20.0331936, -46.7907715, 46.7764053
13: -32.9377441, 28.7967453, -32.8812943, 28.7321415, -61.6698837, 61.6780396
14: -23.5091972, 39.2350311, -23.3609581, 39.1062965, -60.0151978, 59.9956856
15: -18.9600430, 25.8782864, -18.9017563, 25.7970467, -44.7570877, 44.7800446
16: -32.7386932, 19.9083920, -32.6804886, 19.8093338, -52.5480270, 52.5888824
17: -17.7717209, 38.5002708, -17.6826820, 38.3950119, -55.2063866, 55.2286491
18: -25.8337383, 19.6123409, -25.7108307, 19.5726280, -45.4063644, 45.3231735
19: -26.4744701, 12.5120621, -26.3584385, 12.4945431, -38.9690132, 38.8704987
20: -21.1437492, 20.4606705, -21.0143890, 20.4270954, -41.5708466, 41.4750595
21: -25.7630348, 18.9091816, -25.6143551, 18.8739471, -44.6369820, 44.5235367
22: -22.1555576, 24.5406113, -22.0370464, 24.5202293, -46.6757889, 46.5776596
23: -21.7078075, 17.5141296, -21.6525497, 17.4871826, -39.1949921, 39.1666794
24: -32.2031403, 11.8975067, -32.0738297, 11.8651218, -44.0682602, 43.9713364
25: -18.1285172, 25.4307156, -18.0511456, 25.4082375, -43.5367546, 43.4818611
26: -29.2462463, 26.9692268, -29.1396694, 26.9147911, -56.1610374, 56.1088943
27: -32.1439972, 16.5451126, -32.0395966, 16.5121384, -47.8485794, 47.7418556
28: -21.5468292, 21.7141037, -21.4715576, 21.6908951, -43.2377243, 43.1856613
29: -23.7285461, 22.2341328, -23.6151409, 22.1980762, -45.9266205, 45.8492737
30: -29.6733818, 16.8726959, -29.5540962, 16.8283043, -45.9619179, 45.8898964
31: -26.4354744, 19.1091442, -26.2924404, 19.0937519, -45.5292282, 45.4015846
32: -42.2644272, 8.5050306, -42.1452217, 8.4577827, -47.5631294, 47.4971008
33: -72.4640961, -5.6027069, -72.2887573, -5.6543903, -61.3843689, 61.2630844
34: -56.5463486, -5.4678049, -56.4320335, -5.4889078, -43.6737900, 43.5709801
35: -50.2217789, 0.0591612, -50.0840607, 0.0398378, -48.3199310, 48.1883621
36: -47.8260040, 4.9722013, -47.6950684, 4.9641275, -52.0857544, 51.9625244
37: -83.7323151, -17.4379215, -83.5977554, -17.4608517, -58.4569550, 58.3658829
38: -58.7435989, 3.2615671, -58.5511932, 3.2434111, -61.3975372, 61.2330246
39: -79.0716248, -11.5674086, -78.8863678, -11.5838661, -65.4336929, 65.2673645
40: -67.7231445, -18.3171062, -67.6185226, -18.3528519, -41.2179337, 41.1609535
41: -55.2229538, -6.8100481, -55.1454659, -6.8382053, -42.2838669, 42.2623253
42: -33.9449234, 6.8253174, -33.9012756, 6.7767086, -37.6403885, 37.6437263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -45.0008288, upper bound: 44.9070331
time: 52.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0390995, upper bound: 44.9082860
time: 49.87 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -28.1653404, 17.0934792, -28.1322250, 17.0310440, -44.4056435, 44.4544754
1: -13.6917753, 17.0839043, -13.6865206, 17.0413609, -30.7331352, 30.7704239
2: -14.1022892, 21.6601772, -14.0974083, 21.6149216, -35.4710426, 35.5351639
3: -12.9143677, 23.4330521, -12.9138660, 23.3990402, -36.3134079, 36.3469162
4: -21.5870323, 18.4571667, -21.5813942, 18.4434929, -40.0305252, 40.0385590
5: -12.0228224, 22.8303261, -12.0211697, 22.7986221, -34.8214455, 34.8514938
6: -50.7863464, -3.5528679, -50.6810112, -3.5610871, -40.6893921, 40.5624542
7: -16.4142418, 18.4414234, -16.4109535, 18.4080238, -34.8222656, 34.8523788
8: -18.3439617, 21.3482819, -18.3423004, 21.2823944, -39.6263580, 39.6905823
9: -16.7409172, 23.2846928, -16.7339745, 23.2310543, -38.6430321, 38.6659698
10: -24.3325386, 38.5407104, -24.3186493, 38.4690781, -61.8312607, 61.8663979
11: -24.7965565, 17.6060486, -24.7666187, 17.6077499, -42.4043045, 42.3726654
12: -28.6649761, 20.1400414, -28.6484070, 20.1270905, -46.9389801, 46.9212151
13: -32.9563866, 28.8060970, -32.9376373, 28.7751236, -61.7315102, 61.7437363
14: -23.5628166, 39.2395515, -23.5276279, 39.1650581, -60.1294594, 60.1353188
15: -18.9701042, 25.8943939, -18.9553986, 25.8462696, -44.8163757, 44.8497925
16: -32.7566910, 19.9180565, -32.7402191, 19.8649921, -52.6216812, 52.6582756
17: -17.8070755, 38.5058975, -17.7877998, 38.4357986, -55.2827072, 55.3204651
18: -25.8477955, 19.6271172, -25.7834568, 19.6231365, -45.4709320, 45.4105759
19: -26.4896717, 12.5138845, -26.4072323, 12.5122452, -39.0019150, 38.9211159
20: -21.1651688, 20.4629269, -21.0826035, 20.4576416, -41.6228104, 41.5455322
21: -25.7886467, 18.9122505, -25.6912041, 18.9092216, -44.6978683, 44.6034546
22: -22.1742229, 24.5447922, -22.0909710, 24.5437031, -46.7179260, 46.6357651
23: -21.7195129, 17.5162621, -21.6900883, 17.5067654, -39.2262802, 39.2063522
24: -32.2104340, 11.9124126, -32.1170883, 11.9120293, -44.1224632, 44.0295029
25: -18.1441860, 25.4341602, -18.0980434, 25.4353371, -43.5795212, 43.5322037
26: -29.2742844, 26.9732246, -29.2259293, 26.9734230, -56.2477074, 56.1991539
27: -32.1543121, 16.5599365, -32.0980301, 16.5569077, -47.8715134, 47.8132248
28: -21.5622063, 21.7165966, -21.5212345, 21.7144184, -43.2766266, 43.2378311
29: -23.7531242, 22.2376404, -23.6872807, 22.2380905, -45.9912148, 45.9249191
30: -29.6933136, 16.8780937, -29.6132221, 16.8680363, -46.0234413, 45.9521255
31: -26.4507294, 19.1126537, -26.3441200, 19.1103535, -45.5610809, 45.4567719
32: -42.2885895, 8.5100069, -42.2178802, 8.4992781, -47.6478958, 47.5741196
33: -72.4687653, -5.5808287, -72.3272552, -5.5820179, -61.4646912, 61.3268890
34: -56.5523033, -5.4603519, -56.4610939, -5.4610224, -43.7096825, 43.6202278
35: -50.2277145, 0.0686083, -50.1122017, 0.0709305, -48.3625488, 48.2686768
36: -47.8430939, 4.9754963, -47.7447701, 4.9756670, -52.1238174, 52.0202713
37: -83.7433014, -17.4275532, -83.6365356, -17.4314823, -58.5723724, 58.4215775
38: -58.7571716, 3.2687244, -58.6096382, 3.2665644, -61.4345169, 61.2923660
39: -79.0808105, -11.5610886, -78.9261398, -11.5614605, -65.4657364, 65.3124542
40: -67.7306061, -18.3043461, -67.6433716, -18.3166466, -41.3123512, 41.1819496
41: -55.2302933, -6.8029671, -55.1712685, -6.8120670, -42.3879509, 42.2837791
42: -33.9616966, 6.8319693, -33.9514885, 6.8253002, -37.7149124, 37.7039566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=211, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0008288, upper bound: 45.0378355
time: 54.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0390995, upper bound: 45.0390997
time: 51.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 108.48 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 108.48
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0378355
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 108.48
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0390997
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 108.48
Output dim: 14, lower bound: -45.0008288, upper bound: 44.9070331
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 108.48
Output dim: 14, lower bound: -45.0390995, upper bound: 44.9082860
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 108.48
Output dim: 14, lower bound: -45.0008288, upper bound: 45.0378355
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 108.48
Output dim: 14, lower bound: -45.0390995, upper bound: 45.0390997

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.9052925, 17.0121479, -28.0719795, 17.0261269, -44.1422348, 44.3098373
1: -13.5510073, 17.0321732, -13.6502562, 17.0390701, -30.5900764, 30.6824303
2: -13.9649658, 21.6056862, -14.0624838, 21.6125984, -35.3316574, 35.4405670
3: -12.8071957, 23.3789539, -12.8868608, 23.3940315, -36.2012253, 36.2658157
4: -21.4364204, 18.4253044, -21.5422478, 18.4408493, -39.8772697, 39.9675522
5: -11.8882751, 22.7812614, -11.9873095, 22.7943115, -34.6825867, 34.7685699
6: -50.6600151, -3.6987658, -50.6749840, -3.5908928, -40.5260849, 40.4152908
7: -16.2597122, 18.3958588, -16.3709755, 18.4042130, -34.6639252, 34.7668343
8: -18.1487255, 21.2657547, -18.2949791, 21.2780704, -39.4267960, 39.5607338
9: -16.5976238, 23.2125435, -16.6999512, 23.2264099, -38.4951096, 38.5599709
10: -24.1838684, 38.4325333, -24.2857189, 38.4599762, -61.6609344, 61.7136650
11: -24.7445049, 17.5383930, -24.7610703, 17.5911179, -42.3356247, 42.2994614
12: -28.6312637, 20.0089417, -28.6436558, 20.0980625, -46.8750496, 46.7897530
13: -32.8359756, 28.7256165, -32.9069290, 28.7640190, -61.5999947, 61.6325455
14: -23.3277130, 39.1485100, -23.4805984, 39.1610603, -59.8889771, 59.9938965
15: -18.8548355, 25.7856140, -18.9334221, 25.8282967, -44.6831322, 44.7190361
16: -32.6007423, 19.8371410, -32.7020531, 19.8580246, -52.4587669, 52.5391922
17: -17.6714115, 38.4200516, -17.7602272, 38.4321632, -55.1407585, 55.2023544
18: -25.7522144, 19.5603142, -25.7749271, 19.6070061, -45.3592224, 45.3352432
19: -26.3816795, 12.4016991, -26.4010620, 12.4839325, -38.8656120, 38.8027611
20: -21.0535355, 20.3441029, -21.0755005, 20.4277229, -41.4812584, 41.4196014
21: -25.6550579, 18.7706165, -25.6824780, 18.8742180, -44.5292740, 44.4530945
22: -22.0665016, 24.4158707, -22.0829792, 24.5086994, -46.5752029, 46.4988480
23: -21.6652393, 17.4175968, -21.6846619, 17.4804420, -39.1456833, 39.1022568
24: -32.0883751, 11.8148241, -32.1094666, 11.8871155, -43.9754906, 43.9242897
25: -18.0642109, 25.3161507, -18.0892658, 25.4043293, -43.4685402, 43.4054184
26: -29.1903076, 26.8183746, -29.2153149, 26.9324150, -56.1227226, 56.0336914
27: -32.0667725, 16.4446373, -32.0901375, 16.5263443, -47.7233582, 47.6743355
28: -21.4941406, 21.5858002, -21.5145798, 21.6809616, -43.1751022, 43.1003799
29: -23.6681786, 22.1384277, -23.6816902, 22.2103920, -45.8785706, 45.8201180
30: -29.5863991, 16.7717724, -29.6065292, 16.8448582, -45.8943901, 45.8349152
31: -26.3002644, 18.9729347, -26.3337994, 19.0748634, -45.3751297, 45.3067322
32: -42.1918411, 8.3823929, -42.2107925, 8.4712315, -47.5213318, 47.4399872
33: -72.2956390, -5.7493868, -72.3195496, -5.6187239, -61.2477493, 61.1434631
34: -56.4386444, -5.6256981, -56.4554214, -5.5008783, -43.5485039, 43.4483643
35: -50.0849609, -0.0885277, -50.1052742, 0.0343142, -48.1795044, 48.0998154
36: -47.7189140, 4.7893066, -47.7370300, 4.9310331, -51.9521561, 51.8249741
37: -83.6036758, -17.5372753, -83.6280899, -17.4558334, -58.4026489, 58.2969971
38: -58.5653915, 3.0517225, -58.5984955, 3.2153015, -61.1884384, 61.0577774
39: -78.8877487, -11.6900177, -78.9169769, -11.5858917, -65.2431488, 65.1686401
40: -67.6052551, -18.3741856, -67.6331024, -18.3300705, -41.1695480, 41.1039734
41: -55.1474915, -6.9415884, -55.1647339, -6.8432570, -42.2698746, 42.1358414
42: -33.9370499, 6.7336636, -33.9481964, 6.8014526, -37.6659164, 37.5992393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0181715
time: 58.17 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0378355
time: 73.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -28.0855331, 17.0813999, -28.1231041, 17.0304565, -44.3302307, 44.4317932
1: -13.6549873, 17.0821857, -13.6811495, 17.0411205, -30.6961079, 30.7633362
2: -14.0644207, 21.6613235, -14.0925570, 21.6145649, -35.4342499, 35.5280037
3: -12.8856640, 23.4288712, -12.9106560, 23.3984108, -36.2840729, 36.3395271
4: -21.5549316, 18.4585781, -21.5771980, 18.4437485, -39.9986801, 40.0357742
5: -11.9848690, 22.8432236, -12.0166111, 22.7980347, -34.7829056, 34.8598328
6: -50.6992340, -3.6087976, -50.6804085, -3.5649266, -40.5948792, 40.5071144
7: -16.3723602, 18.4630737, -16.4056664, 18.4074936, -34.7798538, 34.8687401
8: -18.2839184, 21.3034019, -18.3354263, 21.2816124, -39.5655289, 39.6388283
9: -16.7048359, 23.2826099, -16.7306385, 23.2306309, -38.6080246, 38.6649017
10: -24.2852116, 38.5109329, -24.3145485, 38.4678268, -61.7733765, 61.8357468
11: -24.7747154, 17.6012096, -24.7656593, 17.6083851, -42.3831024, 42.3668671
12: -28.6673241, 20.0982399, -28.6479378, 20.1229572, -46.9401779, 46.8805618
13: -32.9395180, 28.7877808, -32.9353981, 28.7733307, -61.7128487, 61.7231789
14: -23.4810677, 39.1944389, -23.5206070, 39.1646919, -60.0484123, 60.0871201
15: -18.9310303, 25.8461266, -18.9525719, 25.8444424, -44.7754745, 44.7986984
16: -32.7237167, 19.9340286, -32.7365417, 19.8640728, -52.5877914, 52.6705704
17: -17.7764740, 38.4768829, -17.7834740, 38.4354172, -55.2485962, 55.2935562
18: -25.8107147, 19.6187439, -25.7820339, 19.6232491, -45.4339638, 45.4007797
19: -26.4456463, 12.4784641, -26.4064178, 12.5079536, -38.9535980, 38.8848801
20: -21.1317463, 20.4293518, -21.0815086, 20.4529686, -41.5847168, 41.5108604
21: -25.7302628, 18.8665180, -25.6898041, 18.9037495, -44.6340103, 44.5563202
22: -22.1628952, 24.5132389, -22.0897465, 24.5387383, -46.7016335, 46.6029854
23: -21.7153740, 17.4975395, -21.6896000, 17.5040417, -39.2194138, 39.1871414
24: -32.1435699, 11.8957701, -32.1156006, 11.9111738, -44.0547447, 44.0113716
25: -18.1457329, 25.4176197, -18.0965805, 25.4350262, -43.5807571, 43.5141983
26: -29.3060379, 26.9501572, -29.2243176, 26.9722595, -56.2782974, 56.1744766
27: -32.1346512, 16.5410194, -32.0966644, 16.5550461, -47.8373108, 47.7804146
28: -21.5688038, 21.6945419, -21.5201225, 21.7135849, -43.2823868, 43.2146645
29: -23.7381592, 22.2252350, -23.6863670, 22.2364082, -45.9745674, 45.9116020
30: -29.6278362, 16.8516273, -29.6117420, 16.8668480, -45.9648895, 45.9226952
31: -26.3774261, 19.0714378, -26.3428001, 19.1047935, -45.4822197, 45.4142380
32: -42.2381287, 8.4628410, -42.2171059, 8.4945354, -47.5919189, 47.5265732
33: -72.3777237, -5.6453428, -72.3261642, -5.5887012, -61.3650970, 61.2590866
34: -56.5222015, -5.5136271, -56.4603271, -5.4669752, -43.6695747, 43.5616646
35: -50.1739960, 0.0117779, -50.1111984, 0.0647287, -48.3018646, 48.2065506
36: -47.8250008, 4.9117632, -47.7437973, 4.9688244, -52.0973129, 51.9557114
37: -83.7013550, -17.4683151, -83.6355743, -17.4355869, -58.5346222, 58.3770485
38: -58.7189407, 3.1982412, -58.6081314, 3.2586594, -61.3954620, 61.2168732
39: -78.9755936, -11.6262703, -78.9251709, -11.5671883, -65.3530960, 65.2443848
40: -67.6588135, -18.3386955, -67.6422424, -18.3209038, -41.2333527, 41.1480217
41: -55.1905594, -6.8530178, -55.1707611, -6.8170500, -42.3448219, 42.2320938
42: -33.9689064, 6.8150625, -33.9513168, 6.8249388, -37.7260895, 37.6818962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1745

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0194201
time: 54.23 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0390997
time: 48.06 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.1663628, 17.1281090, -28.0783882, 16.9720974, -44.3574905, 44.4351921
1: -13.6887913, 17.0992432, -13.6454105, 16.9614010, -30.6501923, 30.7446537
2: -14.0956526, 21.6747246, -14.0378036, 21.5163383, -35.3796005, 35.4901199
3: -12.9091377, 23.4400024, -12.8683834, 23.3172073, -36.2263451, 36.3083878
4: -21.5806732, 18.4423370, -21.5128307, 18.3332672, -39.9139404, 39.9551697
5: -12.0171261, 22.8542824, -11.9737892, 22.7260857, -34.7432098, 34.8280716
6: -50.7954483, -3.5562301, -50.6456909, -3.5962539, -40.6387329, 40.5193748
7: -16.4053001, 18.4658527, -16.3571625, 18.3176460, -34.7229462, 34.8230133
8: -18.3361244, 21.3404350, -18.2777214, 21.1870918, -39.5232162, 39.6181564
9: -16.7108421, 23.3319149, -16.6265736, 23.1667938, -38.5463486, 38.6273537
10: -24.2887650, 38.5806808, -24.1707745, 38.3732147, -61.6826401, 61.7910309
11: -24.7860107, 17.6044350, -24.6966324, 17.5647717, -42.3507843, 42.3010674
12: -28.6313534, 20.1371231, -28.4898777, 20.0306168, -46.8090591, 46.7802849
13: -32.9443817, 28.8263397, -32.8782578, 28.7307606, -61.6751404, 61.7045975
14: -23.5261230, 39.2686462, -23.3560448, 39.1058884, -60.0318604, 60.0296707
15: -18.9715366, 25.8832302, -18.8997765, 25.7950630, -44.7666016, 44.7830048
16: -32.7438354, 19.9848328, -32.6766663, 19.8086567, -52.5524902, 52.6614990
17: -17.7969284, 38.5453796, -17.6798553, 38.3944244, -55.2273026, 55.2815018
18: -25.8669186, 19.6156311, -25.7096863, 19.5709076, -45.4378281, 45.3253174
19: -26.5194702, 12.5078382, -26.3575935, 12.4921856, -39.0116577, 38.8654327
20: -21.2004585, 20.4603424, -21.0133667, 20.4244995, -41.6249580, 41.4737091
21: -25.8119507, 18.9057770, -25.6130638, 18.8710880, -44.6830368, 44.5188408
22: -22.2281399, 24.5360146, -22.0360394, 24.5171337, -46.7452736, 46.5720520
23: -21.7399330, 17.5148201, -21.6515465, 17.4848175, -39.2247505, 39.1663666
24: -32.2357941, 11.8973465, -32.0725708, 11.8627853, -44.0985794, 43.9699173
25: -18.1840076, 25.4284801, -18.0499077, 25.4051514, -43.5891571, 43.4783859
26: -29.3300610, 26.9657383, -29.1382561, 26.9107018, -56.2407608, 56.1039963
27: -32.1883469, 16.5446053, -32.0383987, 16.5092831, -47.9042664, 47.7311020
28: -21.6012459, 21.7127419, -21.4705086, 21.6876793, -43.2889252, 43.1832504
29: -23.7815056, 22.2330055, -23.6142750, 22.1954765, -45.9769821, 45.8472824
30: -29.6943836, 16.8781719, -29.5525780, 16.8260403, -45.9842072, 45.8920441
31: -26.4813881, 19.1069279, -26.2911701, 19.0908642, -45.5722504, 45.3980980
32: -42.2887001, 8.5057764, -42.1443481, 8.4551277, -47.5839958, 47.4957314
33: -72.5232849, -5.6003218, -72.2879105, -5.6575527, -61.4386826, 61.2619247
34: -56.6125603, -5.4700575, -56.4312515, -5.4922543, -43.7378998, 43.5619011
35: -50.2897797, 0.0569143, -50.0831299, 0.0368471, -48.3849182, 48.1811180
36: -47.9085999, 4.9674759, -47.6942215, 4.9604378, -52.1651535, 51.9567108
37: -83.8034363, -17.4377213, -83.5966339, -17.4630852, -58.5364380, 58.3613396
38: -58.8640480, 3.2612848, -58.5500298, 3.2390413, -61.5205765, 61.2281418
39: -79.1307144, -11.5676365, -78.8851852, -11.5859642, -65.4906616, 65.2641373
40: -67.7452393, -18.3183708, -67.6173248, -18.3558140, -41.2358818, 41.1584015
41: -55.2449684, -6.8107700, -55.1446381, -6.8410807, -42.3088112, 42.2592697
42: -33.9658012, 6.8284349, -33.9008102, 6.7746124, -37.6657143, 37.6433182

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1780

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0368842, upper bound: 44.8720838
time: 37.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0368864, upper bound: 44.9059527
time: 50.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -27.9920883, 17.0781345, -28.0757446, 17.0260372, -44.2223816, 44.3818283
1: -13.5874195, 17.0764809, -13.6525440, 17.0389328, -30.6263523, 30.7290249
2: -14.0004025, 21.6529579, -14.0642519, 21.6125488, -35.3640938, 35.4936142
3: -12.8336201, 23.4175968, -12.8875618, 23.3939972, -36.2276154, 36.3051605
4: -21.4691467, 18.4464874, -21.5429726, 18.4400101, -39.9091568, 39.9894600
5: -11.9234695, 22.8171692, -11.9888535, 22.7943459, -34.7178154, 34.8060226
6: -50.7672920, -3.6396518, -50.6747742, -3.5893526, -40.6444511, 40.4706726
7: -16.2963676, 18.4296722, -16.3725967, 18.4042053, -34.7005730, 34.8022690
8: -18.2074051, 21.3355141, -18.2977886, 21.2782173, -39.4856224, 39.6333008
9: -16.6369781, 23.2700844, -16.7001610, 23.2262459, -38.5320244, 38.6158829
10: -24.2347679, 38.5135498, -24.2867699, 38.4601593, -61.7136536, 61.8027000
11: -24.7788448, 17.5471268, -24.7608109, 17.5885696, -42.3674164, 42.3079376
12: -28.6499023, 20.0557404, -28.6435051, 20.0995770, -46.8921089, 46.8341980
13: -32.8595505, 28.7734470, -32.9061394, 28.7644787, -61.6240311, 61.6795883
14: -23.4263611, 39.2272072, -23.4827175, 39.1610146, -59.9866104, 60.0760574
15: -18.9052334, 25.8388710, -18.9343033, 25.8281326, -44.7333679, 44.7731743
16: -32.6389160, 19.8976288, -32.7018967, 19.8582916, -52.4972076, 52.5995255
17: -17.7271118, 38.4941711, -17.7617245, 38.4319572, -55.1957130, 55.2821579
18: -25.8224945, 19.5719872, -25.7751713, 19.6051826, -45.4276772, 45.3471603
19: -26.4707336, 12.4328861, -26.4010105, 12.4858742, -38.9566078, 38.8338966
20: -21.1437054, 20.3773270, -21.0755730, 20.4297390, -41.5734444, 41.4528999
21: -25.7624035, 18.8128948, -25.6825809, 18.8768120, -44.6392136, 44.4954758
22: -22.1503811, 24.4428577, -22.0831966, 24.5105286, -46.6609116, 46.5260544
23: -21.7014980, 17.4370308, -21.6841583, 17.4808159, -39.1823120, 39.1211891
24: -32.1879539, 11.8312435, -32.1096840, 11.8856564, -44.0736084, 43.9409256
25: -18.1182137, 25.3304443, -18.0895119, 25.4015903, -43.5198059, 43.4199562
26: -29.2423763, 26.8379135, -29.2155056, 26.9295006, -56.1718750, 56.0534210
27: -32.1308365, 16.4629745, -32.0903244, 16.5253716, -47.8132935, 47.6963692
28: -21.5420227, 21.6064510, -21.5146103, 21.6785679, -43.2205887, 43.1210632
29: -23.7361488, 22.1496773, -23.6816864, 22.2094498, -45.9455986, 45.8313637
30: -29.6729279, 16.8036785, -29.6065140, 16.8438320, -45.9752769, 45.8664818
31: -26.4195118, 19.0118256, -26.3338356, 19.0775299, -45.4970398, 45.3456612
32: -42.2665405, 8.4302092, -42.2106895, 8.4732962, -47.5981445, 47.4860992
33: -72.4458237, -5.6825209, -72.3197174, -5.6152182, -61.4016647, 61.2101364
34: -56.5349617, -5.5747166, -56.4554710, -5.4982653, -43.6527328, 43.4978485
35: -50.2067490, -0.0339594, -50.1053581, 0.0375423, -48.3051910, 48.1547012
36: -47.8196411, 4.8482695, -47.7371521, 4.9342518, -52.0580597, 51.8837357
37: -83.7167206, -17.4963226, -83.6278381, -17.4540176, -58.5199127, 58.3369904
38: -58.7241707, 3.1219568, -58.5988235, 3.2188129, -61.3506622, 61.1284943
39: -79.0520859, -11.6250830, -78.9167480, -11.5822496, -65.4126740, 65.2333984
40: -67.6991425, -18.3411274, -67.6330490, -18.3287468, -41.2665062, 41.1351929
41: -55.2092552, -6.8922691, -55.1644211, -6.8411608, -42.3379326, 42.1845169
42: -33.9506989, 6.7536745, -33.9479065, 6.7997122, -37.6800919, 37.6208458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1780

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9986267, upper bound: 45.0017097
time: 26.25 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9986267, upper bound: 45.0356023
time: 32.44 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.1724091, 17.1473560, -28.1268997, 17.0303688, -44.4105682, 44.5037918
1: -13.6914635, 17.1264954, -13.6834612, 17.0409794, -30.7324429, 30.8099556
2: -14.0998716, 21.7085991, -14.0943260, 21.6145077, -35.4667206, 35.5810776
3: -12.9121408, 23.4674721, -12.9113827, 23.3983707, -36.3105125, 36.3788528
4: -21.5876503, 18.4797554, -21.5779419, 18.4428940, -40.0305443, 40.0576973
5: -12.0200834, 22.8791485, -12.0181675, 22.7980576, -34.8181419, 34.8973160
6: -50.8065338, -3.5496078, -50.6802025, -3.5633826, -40.7132607, 40.5625992
7: -16.4090385, 18.4968872, -16.4072933, 18.4074821, -34.8165207, 34.9041824
8: -18.3426666, 21.3731804, -18.3382435, 21.2817650, -39.6244316, 39.7114258
9: -16.7442341, 23.3401031, -16.7308655, 23.2304535, -38.6449966, 38.7207794
10: -24.3361740, 38.5919266, -24.3155746, 38.4680328, -61.8261566, 61.9247475
11: -24.8090458, 17.6099319, -24.7654018, 17.6058388, -42.4148865, 42.3753357
12: -28.6859550, 20.1452141, -28.6477814, 20.1244736, -46.9572411, 46.9250908
13: -32.9630280, 28.8356667, -32.9346161, 28.7737293, -61.7367554, 61.7702827
14: -23.5797691, 39.2731476, -23.5226860, 39.1646309, -60.1460876, 60.1692619
15: -18.9815750, 25.8993454, -18.9534569, 25.8442841, -44.8258591, 44.8528023
16: -32.7618561, 19.9945030, -32.7363968, 19.8643112, -52.6261673, 52.7308998
17: -17.8322487, 38.5510025, -17.7849579, 38.4352188, -55.3036346, 55.3733177
18: -25.8809719, 19.6304111, -25.7822666, 19.6214180, -45.5023880, 45.4126778
19: -26.5346661, 12.5096703, -26.4063873, 12.5098934, -39.0445595, 38.9160576
20: -21.2218628, 20.4626045, -21.0816002, 20.4550018, -41.6768646, 41.5442047
21: -25.8375587, 18.9088287, -25.6899281, 18.9063663, -44.7439270, 44.5987549
22: -22.2467918, 24.5401917, -22.0899563, 24.5406036, -46.7873955, 46.6301498
23: -21.7516403, 17.5169334, -21.6890888, 17.5044250, -39.2560654, 39.2060242
24: -32.2430840, 11.9122429, -32.1158104, 11.9097004, -44.1527863, 44.0280533
25: -18.1996841, 25.4319305, -18.0968227, 25.4322548, -43.6319389, 43.5287552
26: -29.3581314, 26.9697132, -29.2245083, 26.9693584, -56.3274918, 56.1942215
27: -32.1986656, 16.5594635, -32.0968552, 16.5540695, -47.9272308, 47.8024559
28: -21.6166306, 21.7152328, -21.5201740, 21.7111855, -43.3278160, 43.2354050
29: -23.8060856, 22.2365017, -23.6863956, 22.2354717, -46.0415573, 45.9228973
30: -29.7143211, 16.8835735, -29.6116924, 16.8657856, -46.0457802, 45.9542770
31: -26.4966087, 19.1104012, -26.3428459, 19.1074696, -45.6040802, 45.4532471
32: -42.3128471, 8.5107288, -42.2170219, 8.4966383, -47.6687813, 47.5727692
33: -72.5279236, -5.5784407, -72.3263626, -5.5851774, -61.5190353, 61.3258057
34: -56.6184845, -5.4626160, -56.4603271, -5.4643917, -43.7737770, 43.6111603
35: -50.2957382, 0.0663586, -50.1112442, 0.0679445, -48.4275208, 48.2614670
36: -47.9256744, 4.9707775, -47.7439232, 4.9720039, -52.2032013, 52.0145187
37: -83.8143921, -17.4273758, -83.6353683, -17.4337730, -58.6518784, 58.4170570
38: -58.8776321, 3.2684727, -58.6084824, 3.2621918, -61.5575485, 61.2875290
39: -79.1399231, -11.5613232, -78.9249802, -11.5634975, -65.5225906, 65.3091736
40: -67.7527008, -18.3055725, -67.6421890, -18.3195915, -41.3302917, 41.1793785
41: -55.2523003, -6.8036938, -55.1704407, -6.8149376, -42.4128838, 42.2807693
42: -33.9825592, 6.8351002, -33.9510269, 6.8232088, -37.7402573, 37.7035484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=211, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1780

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0368842, upper bound: 45.0029947
time: 36.30 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0368864, upper bound: 45.0368865
time: 24.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 63.65 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0181715
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9604576, upper bound: 45.0378355
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0194201
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9986786, upper bound: 45.0390997
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -45.0368842, upper bound: 44.8720838
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -45.0368864, upper bound: 44.9059527
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9986267, upper bound: 45.0017097
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -44.9986267, upper bound: 45.0356023
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -45.0368842, upper bound: 45.0029947
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 63.65
Output dim: 14, lower bound: -45.0368864, upper bound: 45.0368865

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -27.9052925, 17.0121479, -28.0210991, 17.0215149, -44.1378250, 44.2608299
1: -13.5510073, 17.0321732, -13.6209469, 17.0355988, -30.5866051, 30.6531200
2: -13.9649658, 21.6056862, -14.0332584, 21.6090927, -35.3279953, 35.4120789
3: -12.8071957, 23.3789539, -12.8613100, 23.3880844, -36.1952820, 36.2402649
4: -21.4364204, 18.4253044, -21.5153351, 18.4311237, -39.8675461, 39.9406395
5: -11.8882751, 22.7812614, -11.9550190, 22.7891312, -34.6774063, 34.7362823
6: -50.6600151, -3.6987658, -50.6713982, -3.6412196, -40.4761391, 40.4118385
7: -16.2597122, 18.3958588, -16.3387585, 18.4002266, -34.6599388, 34.7346191
8: -18.1487255, 21.2657547, -18.2401409, 21.2724304, -39.4211578, 39.5058975
9: -16.5976238, 23.2125435, -16.6663895, 23.2217712, -38.4902039, 38.5261307
10: -24.1838684, 38.4325333, -24.2480087, 38.4498825, -61.6500778, 61.6730423
11: -24.7445049, 17.5383930, -24.7552223, 17.5772438, -42.3217468, 42.2936172
12: -28.6312637, 20.0089417, -28.6392517, 20.0650063, -46.8452835, 46.7805824
13: -32.8359756, 28.7256165, -32.8940201, 28.7466507, -61.5826263, 61.6196365
14: -23.3277130, 39.1485100, -23.4158478, 39.1564865, -59.8838654, 59.9289169
15: -18.8548355, 25.7856140, -18.8969154, 25.8210640, -44.6758995, 44.6825294
16: -32.6007423, 19.8371410, -32.6760902, 19.8495216, -52.4502640, 52.5132294
17: -17.6714115, 38.4200516, -17.7230759, 38.4273262, -55.1360664, 55.1626740
18: -25.7522144, 19.5603142, -25.7678528, 19.5967808, -45.3489952, 45.3281670
19: -26.3816795, 12.4016991, -26.3929787, 12.4560423, -38.8377228, 38.7946777
20: -21.0535355, 20.3441029, -21.0669594, 20.4015694, -41.4551048, 41.4110641
21: -25.6550579, 18.7706165, -25.6714897, 18.8371830, -44.4922409, 44.4421082
22: -22.0665016, 24.4158707, -22.0746269, 24.4842739, -46.5507736, 46.4904976
23: -21.6652393, 17.4175968, -21.6755390, 17.4705391, -39.1357803, 39.0931358
24: -32.0883751, 11.8148241, -32.1027298, 11.8690128, -43.9573898, 43.9175529
25: -18.0642109, 25.3161507, -18.0786133, 25.3857040, -43.4499130, 43.3947639
26: -29.1903076, 26.8183746, -29.2053375, 26.9093170, -56.0996246, 56.0237122
27: -32.0667725, 16.4446373, -32.0815811, 16.5076637, -47.6956940, 47.6659431
28: -21.4941406, 21.5858002, -21.5065269, 21.6596947, -43.1538353, 43.0923271
29: -23.6681786, 22.1384277, -23.6762486, 22.1974297, -45.8656082, 45.8146744
30: -29.5863991, 16.7717724, -29.5989285, 16.8212166, -45.8733826, 45.8270493
31: -26.3002644, 18.9729347, -26.3202477, 19.0404015, -45.3406677, 45.2931824
32: -42.1918411, 8.3823929, -42.2054977, 8.4354362, -47.4856720, 47.4350128
33: -72.2956390, -5.7493868, -72.3104858, -5.6820202, -61.1844177, 61.1331940
34: -56.4386444, -5.6256981, -56.4498520, -5.5492392, -43.5019150, 43.4424248
35: -50.0849609, -0.0885277, -50.0983505, -0.0205145, -48.1241989, 48.0924110
36: -47.7189140, 4.7893066, -47.7299728, 4.8746052, -51.8957520, 51.8177338
37: -83.6036758, -17.5372753, -83.6199265, -17.4918232, -58.3659134, 58.2880020
38: -58.5653915, 3.0517225, -58.5865288, 3.1489296, -61.1213989, 61.0453110
39: -78.8877487, -11.6900177, -78.9063416, -11.6475515, -65.1811752, 65.1582184
40: -67.6052551, -18.3741856, -67.6253815, -18.3548889, -41.1465759, 41.0955009
41: -55.1474915, -6.9415884, -55.1612549, -6.8826370, -42.2285614, 42.1317520
42: -33.9370499, 6.7336636, -33.9438362, 6.7855797, -37.6480751, 37.5950012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1780

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9243115, upper bound: 45.0159039
time: 25.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9581983, upper bound: 45.0159045
time: 53.49 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.9052925, 17.0121479, -28.1079025, 17.0874710, -44.2085495, 44.3428192
1: -13.5510073, 17.0321732, -13.6573763, 17.0798912, -30.6308975, 30.6895485
2: -13.9649658, 21.6056862, -14.0687103, 21.6563644, -35.3800049, 35.4452477
3: -12.8071957, 23.3789539, -12.8877945, 23.4267044, -36.2339020, 36.2667465
4: -21.4364204, 18.4253044, -21.5480366, 18.4523296, -39.8887482, 39.9733429
5: -11.8882751, 22.7812614, -11.9902248, 22.8250427, -34.7133179, 34.7714844
6: -50.6600151, -3.6987658, -50.7786674, -3.5820217, -40.5358772, 40.5280075
7: -16.2597122, 18.3958588, -16.3754368, 18.4340439, -34.6937561, 34.7712936
8: -18.1487255, 21.2657547, -18.2988510, 21.3421669, -39.4908905, 39.5646057
9: -16.5976238, 23.2125435, -16.7058144, 23.2792816, -38.5467949, 38.5643120
10: -24.1838684, 38.4325333, -24.2989578, 38.5308495, -61.7370529, 61.7258530
11: -24.7445049, 17.5383930, -24.7895622, 17.5860004, -42.3305054, 42.3279572
12: -28.6312637, 20.0089417, -28.6579056, 20.1119232, -46.8881340, 46.7975960
13: -32.8359756, 28.7256165, -32.9175148, 28.7945194, -61.6304932, 61.6431313
14: -23.3277130, 39.1485100, -23.5145397, 39.2351761, -59.9642639, 60.0283585
15: -18.8548355, 25.7856140, -18.9474583, 25.8742943, -44.7291298, 44.7330704
16: -32.6007423, 19.8371410, -32.7142181, 19.9099617, -52.5107040, 52.5513611
17: -17.6714115, 38.4200516, -17.7788239, 38.5014381, -55.2102966, 55.2190704
18: -25.7522144, 19.5603142, -25.8380680, 19.6084328, -45.3606491, 45.3983841
19: -26.3816795, 12.4016991, -26.4820023, 12.4872265, -38.8689041, 38.8837013
20: -21.0535355, 20.3441029, -21.1570778, 20.4347935, -41.4883270, 41.5011826
21: -25.6550579, 18.7706165, -25.7787762, 18.8794994, -44.5345573, 44.5493927
22: -22.0665016, 24.4158707, -22.1585026, 24.5112629, -46.5777664, 46.5743713
23: -21.6652393, 17.4175968, -21.7117920, 17.4899540, -39.1551933, 39.1293869
24: -32.0883751, 11.8148241, -32.2022514, 11.8854628, -43.9738388, 44.0170746
25: -18.0642109, 25.3161507, -18.1325397, 25.4000225, -43.4642334, 43.4486923
26: -29.1903076, 26.8183746, -29.2574978, 26.9288540, -56.1191635, 56.0758743
27: -32.0667725, 16.4446373, -32.1456070, 16.5260925, -47.7217941, 47.7362175
28: -21.4941406, 21.5858002, -21.5543613, 21.6803856, -43.1745262, 43.1401596
29: -23.6681786, 22.1384277, -23.7441902, 22.2087040, -45.8768845, 45.8826180
30: -29.5863991, 16.7717724, -29.6854286, 16.8531494, -45.8976440, 45.9114571
31: -26.3002644, 18.9729347, -26.4394321, 19.0793438, -45.3796082, 45.4123688
32: -42.1918411, 8.3823929, -42.2801971, 8.4833355, -47.5330582, 47.5106201
33: -72.2956390, -5.7493868, -72.4606247, -5.6151342, -61.2515564, 61.2835846
34: -56.4386444, -5.6256981, -56.5461464, -5.4982357, -43.5547562, 43.5431480
35: -50.0849609, -0.0885277, -50.2201424, 0.0340605, -48.1796417, 48.2149048
36: -47.7189140, 4.7893066, -47.8306656, 4.9336348, -51.9546661, 51.9204483
37: -83.6036758, -17.5372753, -83.7329178, -17.4509163, -58.4062653, 58.4034233
38: -58.5653915, 3.0517225, -58.7452202, 3.2191563, -61.1921692, 61.2039490
39: -78.8877487, -11.6900177, -79.0706635, -11.5826082, -65.2462997, 65.3241272
40: -67.6052551, -18.3741856, -67.7192535, -18.3217907, -41.1781807, 41.1914864
41: -55.1474915, -6.9415884, -55.2230072, -6.8333273, -42.2793846, 42.1977768
42: -33.9370499, 6.7336636, -33.9574966, 6.8056068, -37.6700974, 37.6098557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1780

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9243115, upper bound: 45.0356004
time: 30.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9581983, upper bound: 45.0356024
time: 23.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.0855331, 17.0813999, -28.0721436, 17.0258522, -44.3258095, 44.3827553
1: -13.6549873, 17.0821857, -13.6518154, 17.0376511, -30.6926384, 30.7340012
2: -14.0644207, 21.6613235, -14.0633268, 21.6110611, -35.4305801, 35.4995193
3: -12.8856640, 23.4288712, -12.8851042, 23.3924599, -36.2781219, 36.3139763
4: -21.5549316, 18.4585781, -21.5503159, 18.4340134, -39.9889450, 40.0088959
5: -11.9848690, 22.8432236, -11.9843235, 22.7928524, -34.7777214, 34.8275452
6: -50.6992340, -3.6087976, -50.6767883, -3.6152706, -40.5449104, 40.5036545
7: -16.3723602, 18.4630737, -16.3734589, 18.4035110, -34.7758713, 34.8365326
8: -18.2839184, 21.3034019, -18.2806053, 21.2759590, -39.5598755, 39.5840073
9: -16.7048359, 23.2826099, -16.6970901, 23.2259789, -38.6031418, 38.6310616
10: -24.2852116, 38.5109329, -24.2768116, 38.4577026, -61.7625351, 61.7951088
11: -24.7747154, 17.6012096, -24.7598038, 17.5945415, -42.3692551, 42.3610153
12: -28.6673241, 20.0982399, -28.6435280, 20.0898705, -46.9103127, 46.8713531
13: -32.9395180, 28.7877808, -32.9225082, 28.7559185, -61.6954346, 61.7102890
14: -23.4810677, 39.1944389, -23.4558601, 39.1601028, -60.0432777, 60.0221214
15: -18.9310303, 25.8461266, -18.9159966, 25.8372002, -44.7682304, 44.7621231
16: -32.7237167, 19.9340286, -32.7105637, 19.8555431, -52.5792618, 52.6445923
17: -17.7764740, 38.4768829, -17.7462959, 38.4305725, -55.2439117, 55.2538872
18: -25.8107147, 19.6187439, -25.7749443, 19.6130180, -45.4237328, 45.3936882
19: -26.4456463, 12.4784641, -26.3983383, 12.4800529, -38.9256973, 38.8768005
20: -21.1317463, 20.4293518, -21.0729523, 20.4268227, -41.5585709, 41.5023041
21: -25.7302628, 18.8665180, -25.6788216, 18.8667068, -44.5969696, 44.5453415
22: -22.1628952, 24.5132389, -22.0814209, 24.5143242, -46.6772194, 46.5946579
23: -21.7153740, 17.4975395, -21.6804581, 17.4941730, -39.2095490, 39.1779976
24: -32.1435699, 11.8957701, -32.1088295, 11.8930492, -44.0366211, 44.0046005
25: -18.1457329, 25.4176197, -18.0859032, 25.4163857, -43.5621185, 43.5035248
26: -29.3060379, 26.9501572, -29.2143230, 26.9491463, -56.2551842, 56.1644821
27: -32.1346512, 16.5410194, -32.0881042, 16.5363388, -47.8096390, 47.7720261
28: -21.5688038, 21.6945419, -21.5120602, 21.6923218, -43.2611237, 43.2066040
29: -23.7381592, 22.2252350, -23.6809502, 22.2234402, -45.9616013, 45.9061852
30: -29.6278362, 16.8516273, -29.6041431, 16.8431549, -45.9438896, 45.9148178
31: -26.3774261, 19.0714378, -26.3292522, 19.0702934, -45.4477196, 45.4006882
32: -42.2381287, 8.4628410, -42.2118225, 8.4587708, -47.5562897, 47.5216064
33: -72.3777237, -5.6453428, -72.3170776, -5.6519594, -61.3018341, 61.2488708
34: -56.5222015, -5.5136271, -56.4547234, -5.5153608, -43.6229935, 43.5557251
35: -50.1739960, 0.0117779, -50.1042709, 0.0098696, -48.2465439, 48.1991653
36: -47.8250008, 4.9117632, -47.7367744, 4.9123850, -52.0409546, 51.9484787
37: -83.7013550, -17.4683151, -83.6274185, -17.4716072, -58.4978943, 58.3680229
38: -58.7189407, 3.1982412, -58.5961914, 3.1923027, -61.3283844, 61.2043991
39: -78.9755936, -11.6262703, -78.9145584, -11.6288385, -65.2910919, 65.2339630
40: -67.6588135, -18.3386955, -67.6345062, -18.3457489, -41.2103386, 41.1395340
41: -55.1905594, -6.8530178, -55.1672630, -6.8564272, -42.3035278, 42.2279968
42: -33.9689064, 6.8150625, -33.9469719, 6.8090525, -37.7082520, 37.6776619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1780

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9625246, upper bound: 45.0171784
time: 52.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9964284, upper bound: 45.0171801
time: 29.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.0855331, 17.0813999, -28.1590500, 17.0918083, -44.3965416, 44.4647713
1: -13.6549873, 17.0821857, -13.6883001, 17.0819435, -30.7369308, 30.7704849
2: -14.0644207, 21.6613235, -14.0987883, 21.6583252, -35.4826050, 35.5327148
3: -12.8856640, 23.4288712, -12.9116001, 23.4310837, -36.3167496, 36.3404694
4: -21.5549316, 18.4585781, -21.5830154, 18.4552193, -40.0101509, 40.0415955
5: -11.9848690, 22.8432236, -12.0195427, 22.8287697, -34.8136368, 34.8627663
6: -50.6992340, -3.6087976, -50.7840729, -3.5560317, -40.6046906, 40.6198349
7: -16.3723602, 18.4630737, -16.4101276, 18.4373169, -34.8096771, 34.8731995
8: -18.2839184, 21.3034019, -18.3393440, 21.3457241, -39.6296425, 39.6427460
9: -16.7048359, 23.2826099, -16.7365150, 23.2834892, -38.6597328, 38.6692619
10: -24.2852116, 38.5109329, -24.3277931, 38.5386887, -61.8495026, 61.8479614
11: -24.7747154, 17.6012096, -24.7941513, 17.6032333, -42.3779488, 42.3953629
12: -28.6673241, 20.0982399, -28.6621780, 20.1368656, -46.9532585, 46.8883896
13: -32.9395180, 28.7877808, -32.9460220, 28.8038654, -61.7433853, 61.7338028
14: -23.4810677, 39.1944389, -23.5545540, 39.2387924, -60.1236763, 60.1215782
15: -18.9310303, 25.8461266, -18.9666195, 25.8904152, -44.8214455, 44.8127441
16: -32.7237167, 19.9340286, -32.7487068, 19.9160118, -52.6397285, 52.6827354
17: -17.7764740, 38.4768829, -17.8020802, 38.5047035, -55.3181458, 55.3102951
18: -25.8107147, 19.6187439, -25.8451729, 19.6246605, -45.4353752, 45.4639168
19: -26.4456463, 12.4784641, -26.4873352, 12.5112457, -38.9568939, 38.9657974
20: -21.1317463, 20.4293518, -21.1630745, 20.4600410, -41.5917892, 41.5924263
21: -25.7302628, 18.8665180, -25.7861176, 18.9090290, -44.6392899, 44.6526337
22: -22.1628952, 24.5132389, -22.1652985, 24.5413055, -46.7042007, 46.6785355
23: -21.7153740, 17.4975395, -21.7167110, 17.5135384, -39.2289124, 39.2142487
24: -32.1435699, 11.8957701, -32.2083740, 11.9095058, -44.0530777, 44.1041451
25: -18.1457329, 25.4176197, -18.1398201, 25.4307137, -43.5764465, 43.5574417
26: -29.3060379, 26.9501572, -29.2664795, 26.9687271, -56.2747650, 56.2166367
27: -32.1346512, 16.5410194, -32.1520729, 16.5548019, -47.8357391, 47.8422966
28: -21.5688038, 21.6945419, -21.5598946, 21.7130165, -43.2818222, 43.2544365
29: -23.7381592, 22.2252350, -23.7488594, 22.2347393, -45.9729004, 45.9740944
30: -29.6278362, 16.8516273, -29.6906281, 16.8751144, -45.9681511, 45.9992027
31: -26.3774261, 19.0714378, -26.4484138, 19.1092548, -45.4866791, 45.5198517
32: -42.2381287, 8.4628410, -42.2865334, 8.5066729, -47.6036911, 47.5971909
33: -72.3777237, -5.6453428, -72.4672699, -5.5851059, -61.3689346, 61.3992310
34: -56.5222015, -5.5136271, -56.5510178, -5.4643507, -43.6758499, 43.6564331
35: -50.1739960, 0.0117779, -50.2260590, 0.0644617, -48.3019943, 48.3216743
36: -47.8250008, 4.9117632, -47.8374138, 4.9714212, -52.0998535, 52.0511780
37: -83.7013550, -17.4683151, -83.7404327, -17.4306889, -58.5382767, 58.4834366
38: -58.7189407, 3.1982412, -58.7548714, 3.2625380, -61.3991776, 61.3630142
39: -78.9755936, -11.6262703, -79.0788956, -11.5638676, -65.3561859, 65.3999100
40: -67.6588135, -18.3386955, -67.7283936, -18.3126411, -41.2420197, 41.2355232
41: -55.1905594, -6.8530178, -55.2290077, -6.8071098, -42.3543625, 42.2940254
42: -33.9689064, 6.8150625, -33.9606285, 6.8290834, -37.7302628, 37.6925240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1780

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9625246, upper bound: 45.0368845
time: 25.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9964284, upper bound: 45.0368866
time: 45.20 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -28.1402817, 17.1249523, -27.9937363, 16.9414978, -44.2959747, 44.3436584
1: -13.6727266, 17.0974064, -13.5955276, 16.9354992, -30.6082268, 30.6929340
2: -14.0806417, 21.6728954, -13.9915733, 21.4965668, -35.3397598, 35.4382629
3: -12.8910875, 23.4364738, -12.8138924, 23.2828426, -36.1739311, 36.2503662
4: -21.5628815, 18.4393578, -21.4560890, 18.3157578, -39.8786392, 39.8954468
5: -12.0001888, 22.8507938, -11.9223881, 22.6925449, -34.6927338, 34.7731819
6: -50.7932281, -3.5772057, -50.6245041, -3.6606541, -40.5715714, 40.4759445
7: -16.3917694, 18.4625854, -16.3150330, 18.2936077, -34.6853790, 34.7776184
8: -18.3182030, 21.3375416, -18.2207165, 21.1732578, -39.4914627, 39.5582581
9: -16.6829510, 23.3283234, -16.5402718, 23.1200409, -38.4709473, 38.5375175
10: -24.2582779, 38.5737343, -24.0777092, 38.3088150, -61.5868225, 61.6910706
11: -24.7828960, 17.5803871, -24.6550045, 17.4902496, -42.2731476, 42.2353897
12: -28.6169338, 20.1274986, -28.4434395, 19.9793911, -46.7435875, 46.7226944
13: -32.9189758, 28.8198395, -32.7989655, 28.6827278, -61.6017036, 61.6188049
14: -23.4851685, 39.2653885, -23.2274284, 39.0586128, -59.9417000, 59.8972816
15: -18.9485054, 25.8793526, -18.8277168, 25.7734509, -44.7219543, 44.7070694
16: -32.7219086, 19.9796562, -32.6036835, 19.7548752, -52.4767838, 52.5833397
17: -17.7684917, 38.5424576, -17.5898209, 38.3526611, -55.1565781, 55.1885414
18: -25.8624878, 19.5999908, -25.6759853, 19.5214500, -45.3839378, 45.2759781
19: -26.5144844, 12.4810915, -26.2970638, 12.4122562, -38.9267426, 38.7781563
20: -21.1958408, 20.4380531, -20.9617348, 20.3572540, -41.5530930, 41.3997879
21: -25.8062286, 18.8785439, -25.5507603, 18.7896023, -44.5958328, 44.4293060
22: -22.2230186, 24.5199680, -21.9803219, 24.4677258, -46.6907425, 46.5002899
23: -21.7357025, 17.4904213, -21.5981350, 17.4103203, -39.1460228, 39.0885544
24: -32.2315025, 11.8704720, -32.0017242, 11.7818842, -44.0133858, 43.8721962
25: -18.1782455, 25.4067345, -17.9961033, 25.3387871, -43.5170326, 43.4028397
26: -29.3237972, 26.9346123, -29.0696945, 26.8177376, -56.1415329, 56.0043068
27: -32.1841660, 16.5182533, -31.9763317, 16.4296837, -47.8200302, 47.6429749
28: -21.5965462, 21.6867180, -21.4170837, 21.6095123, -43.2060585, 43.1038017
29: -23.7775631, 22.2168770, -23.5577869, 22.1458435, -45.9234085, 45.7746658
30: -29.6909027, 16.8552094, -29.5072422, 16.7551689, -45.9088860, 45.8228035
31: -26.4740467, 19.0810032, -26.2310524, 19.0124931, -45.4865417, 45.3120575
32: -42.2854309, 8.4964571, -42.1216049, 8.4230528, -47.5452728, 47.4596062
33: -72.5168762, -5.6243210, -72.2292633, -5.7315302, -61.3559952, 61.1764450
34: -56.6090927, -5.4868832, -56.4040031, -5.5448542, -43.6823425, 43.5168839
35: -50.2851601, 0.0407276, -50.0419655, -0.0122252, -48.3280487, 48.1215439
36: -47.9034653, 4.9418736, -47.6448441, 4.8838644, -52.0833740, 51.8812256
37: -83.7967529, -17.4667301, -83.5177383, -17.5505390, -58.4396057, 58.2507172
38: -58.8581543, 3.2290497, -58.4824333, 3.1384563, -61.4146576, 61.1262665
39: -79.1235428, -11.5891953, -78.8141403, -11.6515236, -65.4162216, 65.1691971
40: -67.7403259, -18.3332405, -67.5752640, -18.4010677, -41.1837692, 41.0970268
41: -55.2419815, -6.8367043, -55.1064072, -6.9205885, -42.2256088, 42.1939850
42: -33.9633484, 6.8113079, -33.8787498, 6.7210102, -37.6091003, 37.5993042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9967231, upper bound: 44.8686127
time: 72.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0354287, upper bound: 44.8691340
time: 26.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -28.1642494, 17.1277504, -28.0720177, 16.9710426, -44.3576622, 44.4274025
1: -13.6876669, 17.0990410, -13.6420059, 16.9608040, -30.6484718, 30.7410469
2: -14.0945635, 21.6745605, -14.0345097, 21.5158482, -35.3804626, 35.4855423
3: -12.9079208, 23.4396439, -12.8646049, 23.3161182, -36.2240372, 36.3042488
4: -21.5794716, 18.4420662, -21.5090904, 18.3324432, -39.9119148, 39.9511566
5: -12.0159473, 22.8538475, -11.9701700, 22.7247829, -34.7407303, 34.8240166
6: -50.7951965, -3.5578456, -50.6450043, -3.6011205, -40.6302795, 40.5169106
7: -16.4043350, 18.4655495, -16.3542290, 18.3167381, -34.7210732, 34.8197784
8: -18.3349304, 21.3401451, -18.2740135, 21.1862144, -39.5211449, 39.6141586
9: -16.7089558, 23.3316174, -16.6208096, 23.1658592, -38.5434189, 38.6150513
10: -24.2866611, 38.5801353, -24.1642761, 38.3714371, -61.6784210, 61.7764511
11: -24.7856827, 17.6027622, -24.6956635, 17.5596523, -42.3453369, 42.2984238
12: -28.6302795, 20.1364212, -28.4865551, 20.0284424, -46.8058167, 46.7709618
13: -32.9426270, 28.8258057, -32.8728523, 28.7291203, -61.6717453, 61.6986580
14: -23.5229645, 39.2684174, -23.3465137, 39.1052017, -60.0276794, 60.0111961
15: -18.9699593, 25.8828926, -18.8950253, 25.7940445, -44.7640038, 44.7779160
16: -32.7422905, 19.9842434, -32.6718636, 19.8068485, -52.5491409, 52.6561050
17: -17.7948132, 38.5451088, -17.6735344, 38.3935547, -55.2243500, 55.2705116
18: -25.8665104, 19.6145401, -25.7084503, 19.5675812, -45.4340897, 45.3229904
19: -26.5190411, 12.5060253, -26.3563099, 12.4866047, -39.0056458, 38.8623352
20: -21.2000351, 20.4587917, -21.0120850, 20.4197845, -41.6198196, 41.4708786
21: -25.8114586, 18.9039001, -25.6115818, 18.8653755, -44.6768341, 44.5154800
22: -22.2276421, 24.5349197, -22.0345573, 24.5137558, -46.7413979, 46.5694771
23: -21.7394924, 17.5131721, -21.6502266, 17.4797745, -39.2192688, 39.1633987
24: -32.2353287, 11.8955555, -32.0711365, 11.8572531, -44.0925827, 43.9666901
25: -18.1835155, 25.4270210, -18.0484524, 25.4006672, -43.5841827, 43.4754715
26: -29.3295231, 26.9635773, -29.1365585, 26.9040051, -56.2335281, 56.1001358
27: -32.1879654, 16.5428772, -32.0372238, 16.5039062, -47.8929062, 47.7281685
28: -21.6008244, 21.7109718, -21.4692078, 21.6822701, -43.2830963, 43.1801796
29: -23.7811184, 22.2318802, -23.6130733, 22.1919594, -45.9730759, 45.8449554
30: -29.6940575, 16.8766098, -29.5515862, 16.8212852, -45.9772987, 45.8894272
31: -26.4807339, 19.1051922, -26.2892036, 19.0855370, -45.5662689, 45.3943939
32: -42.2882500, 8.5050316, -42.1430969, 8.4528770, -47.5803375, 47.4960709
33: -72.5225220, -5.6020298, -72.2855377, -5.6626797, -61.4291458, 61.2576904
34: -56.6119347, -5.4712124, -56.4294357, -5.4957867, -43.7335968, 43.5594559
35: -50.2892570, 0.0557899, -50.0814552, 0.0334206, -48.3787994, 48.1782532
36: -47.9080429, 4.9657993, -47.6925354, 4.9551792, -52.1562195, 51.9532013
37: -83.8028183, -17.4397469, -83.5948029, -17.4692974, -58.5106583, 58.3572884
38: -58.8634834, 3.2591314, -58.5482826, 3.2323751, -61.5046158, 61.2237930
39: -79.1298523, -11.5691509, -78.8825226, -11.5905371, -65.4754715, 65.2599182
40: -67.7447739, -18.3194523, -67.6159286, -18.3590698, -41.2224655, 41.1557007
41: -55.2446709, -6.8125439, -55.1437759, -6.8465147, -42.2919350, 42.2564621
42: -33.9655838, 6.8272057, -33.9001389, 6.7709179, -37.6602821, 37.6454430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9968609, upper bound: 44.9043718
time: 24.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0354351, upper bound: 44.9044683
time: 51.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -27.9660606, 17.0749702, -27.9910889, 16.9954071, -44.1608696, 44.2902946
1: -13.5713787, 17.0746422, -13.6026697, 17.0130463, -30.5844250, 30.6773109
2: -13.9854221, 21.6511288, -14.0180378, 21.5927792, -35.3242645, 35.4417648
3: -12.8156042, 23.4140549, -12.8330860, 23.3596172, -36.1752205, 36.2471390
4: -21.4513702, 18.4435272, -21.4862499, 18.4225025, -39.8738708, 39.9297791
5: -11.9065170, 22.8136902, -11.9374418, 22.7607880, -34.6673050, 34.7511330
6: -50.7650833, -3.6605997, -50.6535988, -3.6537175, -40.5773201, 40.4272804
7: -16.2828445, 18.4264183, -16.3304749, 18.3801537, -34.6629982, 34.7568932
8: -18.1894913, 21.3326321, -18.2408028, 21.2643967, -39.4538879, 39.5734329
9: -16.6091080, 23.2664909, -16.6138954, 23.1794910, -38.4566269, 38.5260735
10: -24.2043037, 38.5065231, -24.1937218, 38.3958130, -61.6178741, 61.7027664
11: -24.7757454, 17.5230541, -24.7191830, 17.5140858, -42.2898331, 42.2422371
12: -28.6354904, 20.0461502, -28.5970650, 20.0483837, -46.8267326, 46.7767029
13: -32.8341446, 28.7668552, -32.8268738, 28.7163811, -61.5505257, 61.5937271
14: -23.3854446, 39.2239609, -23.3541336, 39.1137314, -59.8964500, 59.9436913
15: -18.8822250, 25.8350105, -18.8622551, 25.8065224, -44.6887474, 44.6972656
16: -32.6169510, 19.8924389, -32.6289330, 19.8044586, -52.4214096, 52.5213699
17: -17.6987305, 38.4912376, -17.6717110, 38.3901787, -55.1250076, 55.1891670
18: -25.8180428, 19.5563793, -25.7415199, 19.5557480, -45.3737907, 45.2978973
19: -26.4657459, 12.4061232, -26.3404827, 12.4059353, -38.8716812, 38.7466049
20: -21.1390762, 20.3550758, -21.0239410, 20.3624916, -41.5015678, 41.3790169
21: -25.7566566, 18.7856865, -25.6202774, 18.7953415, -44.5519981, 44.4059639
22: -22.1452694, 24.4267998, -22.0274353, 24.4611244, -46.6063919, 46.4542351
23: -21.6972580, 17.4126587, -21.6307430, 17.4062901, -39.1035461, 39.0434036
24: -32.1836281, 11.8044004, -32.0388565, 11.8047647, -43.9883919, 43.8432579
25: -18.1124325, 25.3087196, -18.0356846, 25.3352032, -43.4476357, 43.3444061
26: -29.2360821, 26.8067989, -29.1469002, 26.8365116, -56.0725937, 55.9536972
27: -32.1266403, 16.4366474, -32.0282593, 16.4457951, -47.7290497, 47.6082382
28: -21.5372887, 21.5804253, -21.4611931, 21.6004105, -43.1376991, 43.0416183
29: -23.7321625, 22.1335564, -23.6252232, 22.1598053, -45.8919678, 45.7587814
30: -29.6694298, 16.7807236, -29.5611706, 16.7729263, -45.8999405, 45.7972641
31: -26.4121380, 18.9859085, -26.2736931, 18.9991760, -45.4113159, 45.2596016
32: -42.2632866, 8.4209137, -42.1879425, 8.4412441, -47.5594482, 47.4499893
33: -72.4394226, -5.7064819, -72.2610779, -5.6891289, -61.3189392, 61.1245956
34: -56.5314751, -5.5914869, -56.4281960, -5.5508747, -43.5971756, 43.4528809
35: -50.2020950, -0.0501575, -50.0641861, -0.0115042, -48.2483292, 48.0951004
36: -47.8144836, 4.8226995, -47.6877632, 4.8576698, -51.9762421, 51.8081970
37: -83.7100525, -17.5253525, -83.5489502, -17.5414410, -58.4231110, 58.2264023
38: -58.7182579, 3.0896597, -58.5312042, 3.1182499, -61.2447281, 61.0265732
39: -79.0449295, -11.6466103, -78.8457184, -11.6478167, -65.3382874, 65.1384506
40: -67.6942444, -18.3559608, -67.5909958, -18.3739815, -41.2143784, 41.0738411
41: -55.2062683, -6.9182215, -55.1261711, -6.9206572, -42.2547417, 42.1192093
42: -33.9482460, 6.7365532, -33.9258690, 6.7461004, -37.6234665, 37.5768471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9584623, upper bound: 44.9982591
time: 26.27 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9971734, upper bound: 44.9987941
time: 26.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.9899750, 17.0777779, -28.0693817, 17.0249710, -44.2225266, 44.3740425
1: -13.5863276, 17.0762730, -13.6491461, 17.0383472, -30.6246758, 30.7254181
2: -13.9993191, 21.6527901, -14.0609865, 21.6120605, -35.3649597, 35.4890404
3: -12.8324003, 23.4172401, -12.8838196, 23.3929176, -36.2253189, 36.3010597
4: -21.4679585, 18.4462147, -21.5392342, 18.4391937, -39.9071503, 39.9854507
5: -11.9223080, 22.8167419, -11.9852362, 22.7930374, -34.7153473, 34.8019791
6: -50.7670517, -3.6412592, -50.6740799, -3.5942259, -40.6360016, 40.4682159
7: -16.2954140, 18.4293709, -16.3696442, 18.4033089, -34.6987228, 34.7990150
8: -18.2061996, 21.3352318, -18.2940788, 21.2773438, -39.4835434, 39.6293106
9: -16.6350956, 23.2697678, -16.6944103, 23.2253265, -38.5290680, 38.6035957
10: -24.2326584, 38.5129547, -24.2802925, 38.4584007, -61.7094879, 61.7881317
11: -24.7785282, 17.5454617, -24.7598324, 17.5834465, -42.3619766, 42.3052940
12: -28.6488247, 20.0550480, -28.6401882, 20.0974159, -46.8889122, 46.8248787
13: -32.8577652, 28.7728767, -32.9007339, 28.7628059, -61.6205711, 61.6736107
14: -23.4232407, 39.2269936, -23.4731712, 39.1603088, -59.9824295, 60.0575867
15: -18.9036713, 25.8385353, -18.9295254, 25.8271179, -44.7307892, 44.7680588
16: -32.6373444, 19.8970394, -32.6971283, 19.8564835, -52.4938278, 52.5941696
17: -17.7250271, 38.4938889, -17.7554073, 38.4310760, -55.1927643, 55.2711411
18: -25.8220959, 19.5709114, -25.7739372, 19.6018448, -45.4239426, 45.3448486
19: -26.4703064, 12.4310665, -26.3997383, 12.4802895, -38.9505959, 38.8308029
20: -21.1432858, 20.3757896, -21.0743008, 20.4250412, -41.5683289, 41.4500885
21: -25.7619152, 18.8110313, -25.6811123, 18.8711090, -44.6330261, 44.4921417
22: -22.1499176, 24.4417706, -22.0817051, 24.5071583, -46.6570740, 46.5234756
23: -21.7010307, 17.4354057, -21.6828060, 17.4757595, -39.1767883, 39.1182098
24: -32.1874809, 11.8294754, -32.1082382, 11.8801241, -44.0676041, 43.9377136
25: -18.1177368, 25.3290005, -18.0880623, 25.3970833, -43.5148201, 43.4170609
26: -29.2418308, 26.8357353, -29.2138119, 26.9228020, -56.1646347, 56.0495453
27: -32.1304550, 16.4612389, -32.0891418, 16.5199814, -47.8019409, 47.6934662
28: -21.5415916, 21.6046810, -21.5133190, 21.6731606, -43.2147522, 43.1180000
29: -23.7357216, 22.1485500, -23.6804848, 22.2059269, -45.9416504, 45.8290329
30: -29.6726017, 16.8021049, -29.6055050, 16.8390656, -45.9683876, 45.8638611
31: -26.4188251, 19.0101128, -26.3318462, 19.0722313, -45.4910583, 45.3419571
32: -42.2661400, 8.4294996, -42.2094536, 8.4710464, -47.5945053, 47.4864693
33: -72.4450684, -5.6841593, -72.3173752, -5.6203690, -61.3921051, 61.2058258
34: -56.5343399, -5.5758553, -56.4536400, -5.5018187, -43.6483994, 43.4954185
35: -50.2061996, -0.0350485, -50.1037102, 0.0341158, -48.2990494, 48.1518288
36: -47.8190765, 4.8465338, -47.7354660, 4.9289618, -52.0491180, 51.8802414
37: -83.7161255, -17.4983559, -83.6260376, -17.4601955, -58.4941483, 58.3329468
38: -58.7235794, 3.1198177, -58.5970573, 3.2121754, -61.3346634, 61.1240997
39: -79.0512161, -11.6265707, -78.9141006, -11.5867777, -65.3975830, 65.2291870
40: -67.6986847, -18.3422089, -67.6316376, -18.3320274, -41.2530670, 41.1325111
41: -55.2089577, -6.8940535, -55.1635551, -6.8465881, -42.3210526, 42.1816826
42: -33.9504967, 6.7524443, -33.9472351, 6.7959976, -37.6746368, 37.6229858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9585996, upper bound: 45.0341061
time: 71.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9971790, upper bound: 45.0341458
time: 47.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.1463242, 17.1442223, -28.0422287, 16.9997711, -44.3490295, 44.4122276
1: -13.6753960, 17.1246567, -13.6335621, 17.0150967, -30.6904926, 30.7582188
2: -14.0848866, 21.7067757, -14.0481062, 21.5947456, -35.4269028, 35.5292206
3: -12.8941278, 23.4639549, -12.8568850, 23.3640099, -36.2581367, 36.3208389
4: -21.5698414, 18.4767895, -21.5211868, 18.4253998, -39.9952393, 39.9979782
5: -12.0031357, 22.8756561, -11.9667616, 22.7645092, -34.7676468, 34.8424187
6: -50.8043060, -3.5705538, -50.6589966, -3.6277556, -40.6461029, 40.5191765
7: -16.3955097, 18.4936295, -16.3651733, 18.3834248, -34.7789345, 34.8588028
8: -18.3247490, 21.3702717, -18.2812347, 21.2679348, -39.5926819, 39.6515045
9: -16.7163620, 23.3365364, -16.6445847, 23.1837044, -38.5696106, 38.6309853
10: -24.3056602, 38.5849342, -24.2225513, 38.4036484, -61.7303314, 61.8248444
11: -24.8059464, 17.5858822, -24.7237740, 17.5313377, -42.3372841, 42.3096542
12: -28.6715393, 20.1355858, -28.6013565, 20.0733070, -46.8918190, 46.8675117
13: -32.9376221, 28.8291607, -32.8553200, 28.7257309, -61.6633530, 61.6844788
14: -23.5388298, 39.2699127, -23.3941154, 39.1173439, -60.0559578, 60.0368996
15: -18.9585724, 25.8954601, -18.8813229, 25.8226719, -44.7812424, 44.7767830
16: -32.7399063, 19.9893494, -32.6634178, 19.8105125, -52.5504189, 52.6527672
17: -17.8038406, 38.5480804, -17.6949348, 38.3934441, -55.2328835, 55.2803192
18: -25.8765144, 19.6147842, -25.7486248, 19.5719795, -45.4484940, 45.3634109
19: -26.5296783, 12.4829082, -26.3458500, 12.4299507, -38.9596291, 38.8287582
20: -21.2172585, 20.4403152, -21.0299644, 20.3877411, -41.6049995, 41.4702797
21: -25.8318481, 18.8815937, -25.6276398, 18.8248482, -44.6566963, 44.5092316
22: -22.2416859, 24.5241413, -22.0342484, 24.4911976, -46.7328835, 46.5583878
23: -21.7474022, 17.4925556, -21.6356487, 17.4299088, -39.1773109, 39.1282043
24: -32.2387848, 11.8853521, -32.0449829, 11.8288164, -44.0676003, 43.9303360
25: -18.1939106, 25.4102020, -18.0430050, 25.3658752, -43.5597839, 43.4532089
26: -29.3518410, 26.9385757, -29.1559105, 26.8763733, -56.2282143, 56.0944862
27: -32.1944695, 16.5330887, -32.0347519, 16.4744549, -47.8429680, 47.7143478
28: -21.6119270, 21.6891975, -21.4667625, 21.6330242, -43.2449493, 43.1559601
29: -23.8021431, 22.2203884, -23.6299229, 22.1858444, -45.9879875, 45.8503113
30: -29.7108440, 16.8605995, -29.5663776, 16.7948837, -45.9704285, 45.8850594
31: -26.4892960, 19.0844936, -26.2827129, 19.0290718, -45.5183678, 45.3672066
32: -42.3096008, 8.5014067, -42.1942711, 8.4645538, -47.6300659, 47.5366020
33: -72.5215302, -5.6024361, -72.2677002, -5.6591215, -61.4363708, 61.2402725
34: -56.6150284, -5.4794273, -56.4330711, -5.5170212, -43.7182198, 43.5661812
35: -50.2910919, 0.0501890, -50.0700760, 0.0188684, -48.3706512, 48.2018700
36: -47.9205360, 4.9451618, -47.6945343, 4.8954144, -52.1214142, 51.9389648
37: -83.8077469, -17.4563770, -83.5564728, -17.5211983, -58.5550461, 58.3064117
38: -58.8717117, 3.2362337, -58.5408554, 3.1616068, -61.4516754, 61.1856766
39: -79.1327744, -11.5828505, -78.8539429, -11.6290712, -65.4481506, 65.2142029
40: -67.7477875, -18.3204269, -67.6001434, -18.3648262, -41.2781448, 41.1180267
41: -55.2493057, -6.8296394, -55.1321907, -6.8944178, -42.3296776, 42.2154465
42: -33.9801102, 6.8179493, -33.9290047, 6.7695761, -37.6836166, 37.6595116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9967231, upper bound: 44.9995526
time: 37.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9184531, upper bound: 45.0000764
time: 71.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.1702919, 17.1470070, -28.1205254, 17.0293083, -44.4106979, 44.4959984
1: -13.6903534, 17.1262989, -13.6800365, 17.0403976, -30.7307510, 30.8063354
2: -14.0987997, 21.7084332, -14.0910540, 21.6140175, -35.4676056, 35.5764847
3: -12.9109306, 23.4671154, -12.9076214, 23.3972912, -36.3082199, 36.3747368
4: -21.5864220, 18.4794865, -21.5741844, 18.4420853, -40.0285072, 40.0536728
5: -12.0189075, 22.8787155, -12.0145578, 22.7967567, -34.8156662, 34.8932724
6: -50.8062706, -3.5512247, -50.6794891, -3.5682354, -40.7048111, 40.5601425
7: -16.4080772, 18.4965801, -16.4043770, 18.4065781, -34.8146553, 34.9009552
8: -18.3414593, 21.3728752, -18.3345413, 21.2808857, -39.6223450, 39.7074165
9: -16.7423630, 23.3397980, -16.7251015, 23.2295265, -38.6420670, 38.7085114
10: -24.3340302, 38.5913620, -24.3091030, 38.4662361, -61.8219452, 61.9102097
11: -24.8087254, 17.6082535, -24.7644272, 17.6007214, -42.4094467, 42.3726807
12: -28.6848564, 20.1444740, -28.6444855, 20.1223297, -46.9540405, 46.9157639
13: -32.9612694, 28.8351307, -32.9292259, 28.7721138, -61.7333832, 61.7643585
14: -23.5766258, 39.2729149, -23.5131702, 39.1639252, -60.1419296, 60.1508141
15: -18.9800148, 25.8990097, -18.9486866, 25.8432636, -44.8232803, 44.8476944
16: -32.7603264, 19.9938965, -32.7315979, 19.8625145, -52.6228409, 52.7254944
17: -17.8301525, 38.5507202, -17.7786255, 38.4343376, -55.3006592, 55.3623314
18: -25.8805485, 19.6293297, -25.7810421, 19.6180801, -45.4986267, 45.4103699
19: -26.5342445, 12.5078411, -26.4051037, 12.5043240, -39.0385666, 38.9129448
20: -21.2214508, 20.4610653, -21.0803032, 20.4502792, -41.6717300, 41.5413666
21: -25.8370762, 18.9069672, -25.6884155, 18.9006348, -44.7377090, 44.5953827
22: -22.2463036, 24.5390816, -22.0884933, 24.5372238, -46.7835274, 46.6275749
23: -21.7511978, 17.5152931, -21.6877518, 17.4993706, -39.2505684, 39.2030449
24: -32.2425995, 11.9104519, -32.1143646, 11.9041748, -44.1467743, 44.0248184
25: -18.1991997, 25.4304619, -18.0953503, 25.4277611, -43.6269608, 43.5258102
26: -29.3575668, 26.9675388, -29.2227974, 26.9626541, -56.3202209, 56.1903381
27: -32.1982841, 16.5576973, -32.0956688, 16.5486717, -47.9158630, 47.7995415
28: -21.6162109, 21.7134590, -21.5188713, 21.7057991, -43.3220100, 43.2323303
29: -23.8057060, 22.2353745, -23.6851864, 22.2319527, -46.0376587, 45.9205627
30: -29.7139969, 16.8820076, -29.6106911, 16.8610535, -46.0388565, 45.9516373
31: -26.4959736, 19.1086731, -26.3408737, 19.1021404, -45.5981140, 45.4495468
32: -42.3124237, 8.5100107, -42.2157440, 8.4943552, -47.6651230, 47.5731049
33: -72.5271606, -5.5801220, -72.3240204, -5.5903234, -61.5095062, 61.3215179
34: -56.6178932, -5.4637928, -56.4585075, -5.4679337, -43.7694626, 43.6087036
35: -50.2951889, 0.0652514, -50.1096153, 0.0645361, -48.4213943, 48.2586060
36: -47.9251175, 4.9690495, -47.7422485, 4.9667406, -52.1942749, 52.0110321
37: -83.8137970, -17.4293938, -83.6335526, -17.4399548, -58.6260986, 58.4129868
38: -58.8770638, 3.2663002, -58.6067047, 3.2555151, -61.5416031, 61.2831726
39: -79.1390305, -11.5628510, -78.9223099, -11.5680847, -65.5074844, 65.3049088
40: -67.7522507, -18.3066635, -67.6407776, -18.3228512, -41.3168564, 41.1767082
41: -55.2520218, -6.8054886, -55.1695633, -6.8203812, -42.3960152, 42.2779579
42: -33.9823456, 6.8338671, -33.9503632, 6.8194637, -37.7347946, 37.7056732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=210, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1764

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9968609, upper bound: 45.0353971
time: 26.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0354351, upper bound: 45.0354353
time: 41.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 70.55 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9243115, upper bound: 45.0159039
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9581983, upper bound: 45.0159045
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9243115, upper bound: 45.0356004
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9581983, upper bound: 45.0356024
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9625246, upper bound: 45.0171784
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9964284, upper bound: 45.0171801
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9625246, upper bound: 45.0368845
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9964284, upper bound: 45.0368866
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9967231, upper bound: 44.8686127
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -45.0354287, upper bound: 44.8691340
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9968609, upper bound: 44.9043718
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -45.0354351, upper bound: 44.9044683
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9584623, upper bound: 44.9982591
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9971734, upper bound: 44.9987941
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9585996, upper bound: 45.0341061
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9971790, upper bound: 45.0341458
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9967231, upper bound: 44.9995526
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9184531, upper bound: 45.0000764
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -44.9968609, upper bound: 45.0353971
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 70.55
Output dim: 14, lower bound: -45.0354351, upper bound: 45.0354353

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -27.8208504, 16.9814949, -27.9950294, 17.0183620, -44.0463943, 44.1993027
1: -13.5011673, 17.0062828, -13.6048794, 17.0337620, -30.5349293, 30.6111622
2: -13.9188061, 21.5858994, -14.0182610, 21.6072540, -35.2761536, 35.3722534
3: -12.7527523, 23.3445759, -12.8433084, 23.3845425, -36.1372948, 36.1878853
4: -21.3797646, 18.4078102, -21.4975510, 18.4281731, -39.8079376, 39.9053612
5: -11.8368664, 22.7476730, -11.9380608, 22.7856255, -34.6224899, 34.6857338
6: -50.6388016, -3.7630744, -50.6691437, -3.6621509, -40.4327240, 40.3447876
7: -16.2176075, 18.3717880, -16.3252563, 18.3969612, -34.6145706, 34.6970444
8: -18.0918312, 21.2519398, -18.2222462, 21.2695274, -39.3613586, 39.4741859
9: -16.5113983, 23.1657524, -16.6385345, 23.2181702, -38.4004593, 38.4507256
10: -24.0908470, 38.3680725, -24.2175407, 38.4428596, -61.5502014, 61.5771561
11: -24.7028618, 17.4639015, -24.7521324, 17.5531998, -42.2560616, 42.2160339
12: -28.5848541, 19.9577332, -28.6248398, 20.0554276, -46.7877426, 46.7152481
13: -32.7567444, 28.6774178, -32.8686142, 28.7401447, -61.4968872, 61.5460320
14: -23.1991768, 39.1012115, -23.3749371, 39.1532249, -59.7515221, 59.8387642
15: -18.7829456, 25.7640343, -18.8739300, 25.8171806, -44.6001282, 44.6379623
16: -32.5277786, 19.7833176, -32.6541061, 19.8443184, -52.3720970, 52.4374237
17: -17.5814209, 38.3782578, -17.6946697, 38.4243469, -55.0431137, 55.0919533
18: -25.7185287, 19.5109100, -25.7633915, 19.5811691, -45.2996979, 45.2742996
19: -26.3211155, 12.3217869, -26.3879585, 12.4292870, -38.7504044, 38.7097473
20: -21.0018539, 20.2768745, -21.0623112, 20.3792801, -41.3811340, 41.3391876
21: -25.5926590, 18.6891594, -25.6657410, 18.8099461, -44.4026031, 44.3549004
22: -22.0107098, 24.3664703, -22.0695133, 24.4682121, -46.4789200, 46.4359818
23: -21.6118279, 17.3430901, -21.6713047, 17.4461708, -39.0579987, 39.0143967
24: -32.0175400, 11.7339554, -32.0983963, 11.8421345, -43.8596725, 43.8323517
25: -18.0103016, 25.2497768, -18.0727997, 25.3639755, -43.3742752, 43.3225784
26: -29.1216679, 26.7254429, -29.1990528, 26.8781643, -55.9998322, 55.9244957
27: -32.0046539, 16.3651123, -32.0773582, 16.4813194, -47.6074982, 47.5817413
28: -21.4407005, 21.5076523, -21.5017872, 21.6336975, -43.0743980, 43.0094376
29: -23.6116772, 22.0887985, -23.6722870, 22.1812973, -45.7929764, 45.7610855
30: -29.5410423, 16.7009354, -29.5954342, 16.7982445, -45.8041649, 45.7517738
31: -26.2400932, 18.8946266, -26.3128929, 19.0144825, -45.2545776, 45.2075195
32: -42.1690903, 8.3503685, -42.2022629, 8.4260998, -47.4495163, 47.3963699
33: -72.2369003, -5.8232613, -72.3040161, -5.7059746, -61.0988770, 61.0505600
34: -56.4113541, -5.6782827, -56.4463768, -5.5660429, -43.4569054, 43.3869019
35: -50.0437622, -0.1375685, -50.0937004, -0.0366917, -48.0645752, 48.0355530
36: -47.6694489, 4.7127237, -47.7248459, 4.8490076, -51.8201370, 51.7359772
37: -83.5247269, -17.6246986, -83.6132202, -17.5208511, -58.2552490, 58.1911583
38: -58.4977341, 2.9511433, -58.5805740, 3.1166754, -61.0194397, 60.9393845
39: -78.8166428, -11.7555704, -78.8991547, -11.6690731, -65.0861206, 65.0838852
40: -67.5632019, -18.4194107, -67.6204453, -18.3697701, -41.0852356, 41.0434227
41: -55.1092072, -7.0210218, -55.1582565, -6.9085531, -42.1632309, 42.0485992
42: -33.9150162, 6.6800966, -33.9414024, 6.7684708, -37.6040688, 37.5384064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9417538, upper bound: 44.9760237
time: 30.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9417538, upper bound: 45.0144944
time: 50.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -27.8989563, 17.0110874, -28.0189781, 17.0211639, -44.1300316, 44.2609940
1: -13.5476131, 17.0315838, -13.6198359, 17.0354004, -30.5830135, 30.6514206
2: -13.9616833, 21.6051941, -14.0321856, 21.6089230, -35.3234024, 35.4129410
3: -12.8034391, 23.3778610, -12.8601151, 23.3877201, -36.1911583, 36.2379761
4: -21.4326687, 18.4244804, -21.5141220, 18.4308548, -39.8635254, 39.9386024
5: -11.8846569, 22.7799397, -11.9538460, 22.7886906, -34.6733475, 34.7337875
6: -50.6593094, -3.7036233, -50.6711540, -3.6428161, -40.4736862, 40.4034042
7: -16.2567940, 18.3949432, -16.3378143, 18.3999214, -34.6567154, 34.7327576
8: -18.1450424, 21.2648735, -18.2389584, 21.2721252, -39.4171677, 39.5038300
9: -16.5918770, 23.2116032, -16.6645355, 23.2214470, -38.4779434, 38.5232048
10: -24.1773834, 38.4307251, -24.2458801, 38.4493065, -61.6355286, 61.6688309
11: -24.7435341, 17.5333023, -24.7548981, 17.5755806, -42.3191147, 42.2882004
12: -28.6279602, 20.0067711, -28.6381683, 20.0642700, -46.8359413, 46.7773666
13: -32.8305626, 28.7239685, -32.8922539, 28.7460995, -61.5766602, 61.6162224
14: -23.3181877, 39.1477928, -23.4127159, 39.1562500, -59.8653717, 59.9247475
15: -18.8500843, 25.7846069, -18.8953571, 25.8207245, -44.6708069, 44.6799622
16: -32.5959663, 19.8353100, -32.6745148, 19.8489151, -52.4448814, 52.5098267
17: -17.6650581, 38.4191628, -17.7209721, 38.4270172, -55.1250458, 55.1597633
18: -25.7509613, 19.5569687, -25.7674236, 19.5956955, -45.3466568, 45.3243942
19: -26.3804054, 12.3961229, -26.3925476, 12.4542303, -38.8346367, 38.7886696
20: -21.0522594, 20.3393955, -21.0665302, 20.4000225, -41.4522820, 41.4059258
21: -25.6535702, 18.7649212, -25.6709957, 18.8353367, -44.4889069, 44.4359169
22: -22.0650005, 24.4125328, -22.0741444, 24.4831886, -46.5481873, 46.4866791
23: -21.6639080, 17.4125481, -21.6750870, 17.4689026, -39.1328125, 39.0876350
24: -32.0869408, 11.8093033, -32.1022186, 11.8672180, -43.9541588, 43.9115219
25: -18.0627422, 25.3116550, -18.0781250, 25.3842621, -43.4470062, 43.3897781
26: -29.1886044, 26.8116913, -29.2047844, 26.9071503, -56.0957565, 56.0164757
27: -32.0655823, 16.4392624, -32.0812073, 16.5059338, -47.6927834, 47.6545868
28: -21.4928589, 21.5803795, -21.5060883, 21.6579552, -43.1508141, 43.0864677
29: -23.6669865, 22.1349258, -23.6758461, 22.1962833, -45.8632698, 45.8107719
30: -29.5853844, 16.7670174, -29.5985832, 16.8196335, -45.8707733, 45.8201675
31: -26.2982979, 18.9676437, -26.3196011, 19.0386829, -45.3369827, 45.2872467
32: -42.1905785, 8.3801432, -42.2050972, 8.4347057, -47.4860306, 47.4313850
33: -72.2932739, -5.7545366, -72.3096695, -5.6836882, -61.1801605, 61.1236801
34: -56.4368401, -5.6292439, -56.4492455, -5.5504122, -43.4994621, 43.4381027
35: -50.0833054, -0.0919561, -50.0978203, -0.0216141, -48.1213608, 48.0862808
36: -47.7172241, 4.7840176, -47.7294312, 4.8728952, -51.8922501, 51.8088074
37: -83.6018372, -17.5434608, -83.6193085, -17.4938374, -58.3618240, 58.2621918
38: -58.5636139, 3.0450687, -58.5859299, 3.1467543, -61.1170197, 61.0293579
39: -78.8850784, -11.6945210, -78.9054413, -11.6490211, -65.1769104, 65.1431351
40: -67.6038284, -18.3774529, -67.6248932, -18.3559875, -41.1438713, 41.0820770
41: -55.1466103, -6.9469948, -55.1609688, -6.8844128, -42.2257500, 42.1149139
42: -33.9364090, 6.7299700, -33.9436264, 6.7843485, -37.6502075, 37.5895462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9417538, upper bound: 44.9761802
time: 53.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9775451, upper bound: 45.0144965
time: 24.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.8208504, 16.9814949, -28.0818596, 17.0843124, -44.1171494, 44.2812309
1: -13.5011673, 17.0062828, -13.6413231, 17.0780621, -30.5792294, 30.6476059
2: -13.9188061, 21.5858994, -14.0537148, 21.6545277, -35.3281784, 35.4054222
3: -12.7527523, 23.3445759, -12.8697720, 23.4231701, -36.1759224, 36.2143478
4: -21.3797646, 18.4078102, -21.5302582, 18.4493637, -39.8291283, 39.9380684
5: -11.8368664, 22.7476730, -11.9732780, 22.8215485, -34.6584167, 34.7209511
6: -50.6388016, -3.7630744, -50.7764397, -3.6029859, -40.4924202, 40.4609680
7: -16.2176075, 18.3717880, -16.3619003, 18.4307823, -34.6483917, 34.7336884
8: -18.0918312, 21.2519398, -18.2809563, 21.3392792, -39.4311104, 39.5328979
9: -16.5113983, 23.1657524, -16.6779289, 23.2756977, -38.4570618, 38.4889069
10: -24.0908470, 38.3680725, -24.2684708, 38.5238724, -61.6371765, 61.6299667
11: -24.7028618, 17.4639015, -24.7864742, 17.5619316, -42.2647934, 42.2503738
12: -28.5848541, 19.9577332, -28.6434784, 20.1023369, -46.8306274, 46.7322769
13: -32.7567444, 28.6774178, -32.8921585, 28.7879601, -61.5447044, 61.5695763
14: -23.1991768, 39.1012115, -23.4736176, 39.2319031, -59.8319511, 59.9381943
15: -18.7829456, 25.7640343, -18.9244423, 25.8704224, -44.6533661, 44.6884766
16: -32.5277786, 19.7833176, -32.6922607, 19.9048004, -52.4325790, 52.4755783
17: -17.5814209, 38.3782578, -17.7504349, 38.4985046, -55.1173820, 55.1483345
18: -25.7185287, 19.5109100, -25.8336449, 19.5928173, -45.3113480, 45.3445549
19: -26.3211155, 12.3217869, -26.4769974, 12.4604692, -38.7815857, 38.7987823
20: -21.0018539, 20.2768745, -21.1524582, 20.4125137, -41.4143677, 41.4293327
21: -25.5926590, 18.6891594, -25.7730503, 18.8522549, -44.4449158, 44.4622116
22: -22.0107098, 24.3664703, -22.1533890, 24.4952145, -46.5059242, 46.5198593
23: -21.6118279, 17.3430901, -21.7075615, 17.4655762, -39.0774040, 39.0506516
24: -32.0175400, 11.7339554, -32.1979561, 11.8585815, -43.8761215, 43.9319115
25: -18.0103016, 25.2497768, -18.1267643, 25.3782997, -43.3886032, 43.3765411
26: -29.1216679, 26.7254429, -29.2511959, 26.8977242, -56.0193939, 55.9766388
27: -32.0046539, 16.3651123, -32.1414032, 16.4997253, -47.6335983, 47.6520195
28: -21.4407005, 21.5076523, -21.5496540, 21.6543674, -43.0950699, 43.0573044
29: -23.6116772, 22.0887985, -23.7402115, 22.1925735, -45.8042526, 45.8290100
30: -29.5410423, 16.7009354, -29.6819496, 16.8301849, -45.8284073, 45.8361816
31: -26.2400932, 18.8946266, -26.4320889, 19.0534172, -45.2935104, 45.3267136
32: -42.1690903, 8.3503685, -42.2769623, 8.4739885, -47.4969025, 47.4719734
33: -72.2369003, -5.8232613, -72.4542084, -5.6391201, -61.1660156, 61.2010040
34: -56.4113541, -5.6782827, -56.5426712, -5.5150309, -43.5097771, 43.4876213
35: -50.0437622, -0.1375685, -50.2154922, 0.0178690, -48.1200409, 48.1580811
36: -47.6694489, 4.7127237, -47.8255424, 4.9080505, -51.8790436, 51.8386917
37: -83.5247269, -17.6246986, -83.7262726, -17.4799080, -58.2956009, 58.3065987
38: -58.4977341, 2.9511433, -58.7392883, 3.1868887, -61.0901947, 61.0980453
39: -78.8166428, -11.7555704, -79.0635376, -11.6041365, -65.1511841, 65.2498169
40: -67.5632019, -18.4194107, -67.7143478, -18.3366642, -41.1168213, 41.1394196
41: -55.1092072, -7.0210218, -55.2200241, -6.8592548, -42.2140427, 42.1146240
42: -33.9150162, 6.6800966, -33.9550476, 6.7884893, -37.6260681, 37.5532417

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9210467, upper bound: 44.9954294
time: 74.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9214209, upper bound: 45.0341410
time: 26.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.8989563, 17.0110874, -28.1058083, 17.0871315, -44.2007751, 44.3429642
1: -13.5476131, 17.0315838, -13.6562853, 17.0796986, -30.6273117, 30.6878700
2: -13.9616833, 21.6051941, -14.0676346, 21.6562042, -35.3754158, 35.4461212
3: -12.8034391, 23.3778610, -12.8865700, 23.4263420, -36.2297821, 36.2644310
4: -21.4326687, 18.4244804, -21.5468369, 18.4520531, -39.8847198, 39.9713173
5: -11.8846569, 22.7799397, -11.9890556, 22.8246117, -34.7092667, 34.7689972
6: -50.6593094, -3.7036233, -50.7784271, -3.5836349, -40.5334167, 40.5195923
7: -16.2567940, 18.3949432, -16.3744774, 18.4337349, -34.6905289, 34.7694206
8: -18.1450424, 21.2648735, -18.2976608, 21.3418770, -39.4869194, 39.5625343
9: -16.5918770, 23.2116032, -16.7039433, 23.2789764, -38.5345421, 38.5613632
10: -24.1773834, 38.4307251, -24.2968426, 38.5302811, -61.7224884, 61.7216454
11: -24.7435341, 17.5333023, -24.7892513, 17.5842991, -42.3278351, 42.3225555
12: -28.6279602, 20.0067711, -28.6568184, 20.1112194, -46.8788223, 46.7943802
13: -32.8305626, 28.7239685, -32.9157791, 28.7939949, -61.6245575, 61.6397476
14: -23.3181877, 39.1477928, -23.5113983, 39.2349396, -59.9457817, 60.0241852
15: -18.8500843, 25.7846069, -18.9459114, 25.8739529, -44.7240372, 44.7305183
16: -32.5959663, 19.8353100, -32.7126541, 19.9093838, -52.5053482, 52.5479660
17: -17.6650581, 38.4191628, -17.7767372, 38.5011597, -55.1992874, 55.2161331
18: -25.7509613, 19.5569687, -25.8376732, 19.6073494, -45.3583107, 45.3946419
19: -26.3804054, 12.3961229, -26.4815578, 12.4854259, -38.8658295, 38.8776817
20: -21.0522594, 20.3393955, -21.1566486, 20.4332504, -41.4855118, 41.4960442
21: -25.6535702, 18.7649212, -25.7782955, 18.8776207, -44.5311890, 44.5432167
22: -22.0650005, 24.4125328, -22.1580181, 24.5101585, -46.5751572, 46.5705490
23: -21.6639080, 17.4125481, -21.7113609, 17.4883251, -39.1522331, 39.1239090
24: -32.0869408, 11.8093033, -32.2017670, 11.8836651, -43.9706039, 44.0110703
25: -18.0627422, 25.3116550, -18.1320839, 25.3985634, -43.4613037, 43.4437408
26: -29.1886044, 26.8116913, -29.2569218, 26.9266872, -56.1152916, 56.0686111
27: -32.0655823, 16.4392624, -32.1452179, 16.5243340, -47.7188759, 47.7248688
28: -21.4928589, 21.5803795, -21.5539360, 21.6786156, -43.1714745, 43.1343155
29: -23.6669865, 22.1349258, -23.7437763, 22.2075500, -45.8745346, 45.8787003
30: -29.5853844, 16.7670174, -29.6851158, 16.8515739, -45.8950081, 45.9045830
31: -26.2982979, 18.9676437, -26.4387741, 19.0776081, -45.3759079, 45.4064178
32: -42.1905785, 8.3801432, -42.2797928, 8.4826012, -47.5334244, 47.5069733
33: -72.2932739, -5.7545366, -72.4598236, -5.6167927, -61.2472687, 61.2740860
34: -56.4368401, -5.6292439, -56.5455475, -5.4993887, -43.5523224, 43.5388222
35: -50.0833054, -0.0919561, -50.2196045, 0.0329475, -48.1767883, 48.2087784
36: -47.7172241, 4.7840176, -47.8301277, 4.9319305, -51.9511871, 51.9115067
37: -83.6018372, -17.5434608, -83.7323074, -17.4529305, -58.4021912, 58.3776436
38: -58.5636139, 3.0450687, -58.7446136, 3.2169991, -61.1877594, 61.1879883
39: -78.8850784, -11.6945210, -79.0697937, -11.5840921, -65.2420502, 65.3091049
40: -67.6038284, -18.3774529, -67.7187805, -18.3228722, -41.1754913, 41.1780586
41: -55.1466103, -6.9469948, -55.2227211, -6.8350840, -42.2765770, 42.1809387
42: -33.9364090, 6.7299700, -33.9572716, 6.8043909, -37.6722183, 37.6043930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9567934, upper bound: 44.9955723
time: 27.31 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9568165, upper bound: 45.0341457
time: 31.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.0009422, 17.0507889, -28.0460815, 17.0226994, -44.2343102, 44.3212357
1: -13.6051378, 17.0562897, -13.6357594, 17.0358162, -30.6409531, 30.6920490
2: -14.0182123, 21.6415520, -14.0483322, 21.6092186, -35.3787460, 35.4596710
3: -12.8311996, 23.3944950, -12.8670835, 23.3889332, -36.2201309, 36.2615776
4: -21.4981766, 18.4410706, -21.5324917, 18.4310608, -39.9292374, 39.9735641
5: -11.9334507, 22.8096695, -11.9673805, 22.7893562, -34.7228088, 34.7770500
6: -50.6780396, -3.6731873, -50.6745834, -3.6362019, -40.5015144, 40.4365540
7: -16.3302269, 18.4390259, -16.3599529, 18.4002323, -34.7304611, 34.7989807
8: -18.2269211, 21.2895775, -18.2626896, 21.2730732, -39.4999924, 39.5522690
9: -16.6185665, 23.2358360, -16.6692162, 23.2223701, -38.5133400, 38.5556679
10: -24.1921692, 38.4465256, -24.2463379, 38.4507217, -61.6626205, 61.6993217
11: -24.7330723, 17.5267067, -24.7567101, 17.5704594, -42.3035316, 42.2834167
12: -28.6208878, 20.0470428, -28.6291161, 20.0802841, -46.8527527, 46.8058968
13: -32.8602219, 28.7397842, -32.8970795, 28.7494125, -61.6096344, 61.6368637
14: -23.3524914, 39.1471901, -23.4149170, 39.1568604, -59.9109344, 59.9319878
15: -18.8590813, 25.8245144, -18.8929863, 25.8333168, -44.6923981, 44.7174988
16: -32.6507416, 19.8802528, -32.6885834, 19.8503494, -52.5010910, 52.5688362
17: -17.6864738, 38.4351196, -17.7178841, 38.4276161, -55.1509171, 55.1831284
18: -25.7770977, 19.5693188, -25.7704773, 19.5973778, -45.3744736, 45.3397980
19: -26.3851147, 12.3985395, -26.3933430, 12.4533052, -38.8384209, 38.7918816
20: -21.0801010, 20.3621235, -21.0683327, 20.4045410, -41.4846420, 41.4304581
21: -25.6679497, 18.7850590, -25.6730995, 18.8394814, -44.5074310, 44.4581604
22: -22.1071587, 24.4638100, -22.0762997, 24.4982796, -46.6054382, 46.5401077
23: -21.6619759, 17.4230118, -21.6762238, 17.4697685, -39.1317444, 39.0992355
24: -32.0727463, 11.8148975, -32.1045151, 11.8661699, -43.9389153, 43.9194107
25: -18.0918884, 25.3512135, -18.0801277, 25.3946476, -43.4865341, 43.4313431
26: -29.2374611, 26.8572063, -29.2080441, 26.9180069, -56.1554680, 56.0652504
27: -32.0725594, 16.4614487, -32.0838890, 16.5099926, -47.7214928, 47.6877899
28: -21.5153999, 21.6163940, -21.5073338, 21.6663036, -43.1817017, 43.1237259
29: -23.6816959, 22.1756153, -23.6769867, 22.2073174, -45.8890152, 45.8526001
30: -29.5824966, 16.7807465, -29.6006470, 16.8202057, -45.8746719, 45.8394890
31: -26.3173313, 18.9931183, -26.3218842, 19.0443802, -45.3617096, 45.3150024
32: -42.2153702, 8.4307823, -42.2085571, 8.4494238, -47.5200996, 47.4828491
33: -72.3190308, -5.7192192, -72.3106689, -5.6759501, -61.2162781, 61.1661606
34: -56.4949493, -5.5662441, -56.4512596, -5.5321703, -43.5779915, 43.5001831
35: -50.1327896, -0.0372715, -50.0996132, -0.0063200, -48.1869431, 48.1422768
36: -47.7756119, 4.8351660, -47.7316055, 4.8867931, -51.9653778, 51.8666916
37: -83.6224518, -17.5557289, -83.6207123, -17.5006027, -58.3872528, 58.2711678
38: -58.6513481, 3.0977125, -58.5902252, 3.1600227, -61.2265015, 61.0984573
39: -78.9044800, -11.6918535, -78.9073792, -11.6503448, -65.1960373, 65.1596069
40: -67.6167297, -18.3839417, -67.6295776, -18.3605995, -41.1489258, 41.0874481
41: -55.1522980, -6.9324942, -55.1642761, -6.8823671, -42.2382050, 42.1448174
42: -33.9468880, 6.7614565, -33.9445343, 6.7919369, -37.6642303, 37.6210403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9800037, upper bound: 44.9773150
time: 44.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9804278, upper bound: 45.0157797
time: 25.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.0791817, 17.0803356, -28.0700417, 17.0254936, -44.3180351, 44.3828659
1: -13.6515722, 17.0815926, -13.6507072, 17.0374451, -30.6890182, 30.7322998
2: -14.0611229, 21.6608505, -14.0622559, 21.6108894, -35.4260063, 35.5003738
3: -12.8818855, 23.4277916, -12.8838921, 23.3921032, -36.2739868, 36.3116837
4: -21.5511875, 18.4577446, -21.5490837, 18.4337463, -39.9849319, 40.0068283
5: -11.9812374, 22.8419342, -11.9831276, 22.7924118, -34.7736511, 34.8250618
6: -50.6985245, -3.6136642, -50.6765633, -3.6168675, -40.5424423, 40.4952278
7: -16.3694077, 18.4621735, -16.3725014, 18.4031906, -34.7725983, 34.8346748
8: -18.2802162, 21.3025246, -18.2793694, 21.2756615, -39.5558777, 39.5818939
9: -16.6990719, 23.2816544, -16.6952190, 23.2256660, -38.5908585, 38.6281128
10: -24.2787437, 38.5091248, -24.2746849, 38.4571266, -61.7480011, 61.7909126
11: -24.7737408, 17.5960865, -24.7594681, 17.5928249, -42.3665657, 42.3555527
12: -28.6639996, 20.0960808, -28.6424522, 20.0891609, -46.9009819, 46.8681335
13: -32.9341049, 28.7861385, -32.9207458, 28.7553825, -61.6894875, 61.7068863
14: -23.4715347, 39.1937408, -23.4527016, 39.1598625, -60.0248337, 60.0179291
15: -18.9262791, 25.8451004, -18.9144173, 25.8368511, -44.7631302, 44.7595177
16: -32.7189255, 19.9322128, -32.7089958, 19.8549709, -52.5738983, 52.6412086
17: -17.7701302, 38.4760170, -17.7442017, 38.4302750, -55.2329025, 55.2509117
18: -25.8094959, 19.6154060, -25.7745228, 19.6119270, -45.4214249, 45.3899307
19: -26.4443550, 12.4729052, -26.3979111, 12.4782524, -38.9226074, 38.8708153
20: -21.1304569, 20.4246616, -21.0725403, 20.4252720, -41.5557289, 41.4972000
21: -25.7287807, 18.8608341, -25.6783504, 18.8648529, -44.5936356, 44.5391846
22: -22.1614151, 24.5098362, -22.0809231, 24.5132141, -46.6746292, 46.5907593
23: -21.7140446, 17.4924812, -21.6800175, 17.4925365, -39.2065811, 39.1725006
24: -32.1421204, 11.8902435, -32.1083527, 11.8912630, -44.0333824, 43.9985962
25: -18.1442566, 25.4130936, -18.0854378, 25.4149284, -43.5591850, 43.4985313
26: -29.3043461, 26.9434700, -29.2137661, 26.9469566, -56.2513046, 56.1572342
27: -32.1334877, 16.5356216, -32.0877075, 16.5345898, -47.8067093, 47.7606659
28: -21.5675011, 21.6891518, -21.5116310, 21.6905575, -43.2580566, 43.2007828
29: -23.7369614, 22.2217255, -23.6805401, 22.2223034, -45.9592667, 45.9022675
30: -29.6268253, 16.8468704, -29.6037998, 16.8415794, -45.9412804, 45.9079208
31: -26.3754616, 19.0661430, -26.3285980, 19.0685654, -45.4440269, 45.3947411
32: -42.2368622, 8.4605923, -42.2113953, 8.4579840, -47.5566673, 47.5179520
33: -72.3753662, -5.6505003, -72.3162994, -5.6536255, -61.2975464, 61.2392654
34: -56.5203857, -5.5171814, -56.4541473, -5.5165005, -43.6205368, 43.5513840
35: -50.1723404, 0.0083714, -50.1037292, 0.0087643, -48.2436981, 48.1930237
36: -47.8233070, 4.9064884, -47.7361946, 4.9106646, -52.0374603, 51.9395294
37: -83.6995850, -17.4745159, -83.6267929, -17.4736118, -58.4938431, 58.3422241
38: -58.7171669, 3.1915855, -58.5955925, 3.1901293, -61.3239899, 61.1884155
39: -78.9729462, -11.6308155, -78.9136734, -11.6303053, -65.2868576, 65.2189026
40: -67.6573944, -18.3419762, -67.6340485, -18.3468132, -41.2076416, 41.1261101
41: -55.1896935, -6.8584433, -55.1669540, -6.8581896, -42.3006973, 42.2111359
42: -33.9682426, 6.8113546, -33.9467506, 6.8078442, -37.7103806, 37.6722260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0157590, upper bound: 44.9774643
time: 52.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0157822, upper bound: 45.0157822
time: 61.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.0009422, 17.0507889, -28.1329613, 17.0886478, -44.3050690, 44.4032402
1: -13.6051378, 17.0562897, -13.6722240, 17.0801086, -30.6852455, 30.7285137
2: -14.0182123, 21.6415520, -14.0837946, 21.6564980, -35.4307709, 35.4928513
3: -12.8311996, 23.3944950, -12.8935795, 23.4275513, -36.2587509, 36.2880745
4: -21.4981766, 18.4410706, -21.5652046, 18.4522514, -39.9504280, 40.0062752
5: -11.9334507, 22.8096695, -12.0025902, 22.8252792, -34.7587280, 34.8122597
6: -50.6780396, -3.6731873, -50.7818642, -3.5769844, -40.5612526, 40.5527267
7: -16.3302269, 18.4390259, -16.3966064, 18.4340477, -34.7642746, 34.8356323
8: -18.2269211, 21.2895775, -18.3214111, 21.3428288, -39.5697479, 39.6109886
9: -16.6185665, 23.2358360, -16.7086258, 23.2798958, -38.5699425, 38.5938644
10: -24.1921692, 38.4465256, -24.2972927, 38.5317078, -61.7495804, 61.7521439
11: -24.7330723, 17.5267067, -24.7910423, 17.5791836, -42.3122559, 42.3177490
12: -28.6208878, 20.0470428, -28.6477585, 20.1272621, -46.8956985, 46.8229179
13: -32.8602219, 28.7397842, -32.9206314, 28.7972851, -61.6575089, 61.6604156
14: -23.3524914, 39.1471901, -23.5136089, 39.2355347, -59.9913406, 60.0314331
15: -18.8590813, 25.8245144, -18.9435844, 25.8865471, -44.7456284, 44.7680969
16: -32.6507416, 19.8802528, -32.7267532, 19.9108047, -52.5615463, 52.6070061
17: -17.6864738, 38.4351196, -17.7736664, 38.5017471, -55.2251778, 55.2395439
18: -25.7770977, 19.5693188, -25.8407059, 19.6090469, -45.3861465, 45.4100266
19: -26.3851147, 12.3985395, -26.4823437, 12.4844933, -38.8696060, 38.8808823
20: -21.0801010, 20.3621235, -21.1584549, 20.4377670, -41.5178680, 41.5205765
21: -25.6679497, 18.7850590, -25.7803822, 18.8817806, -44.5497284, 44.5654411
22: -22.1071587, 24.4638100, -22.1601734, 24.5252647, -46.6324234, 46.6239853
23: -21.6619759, 17.4230118, -21.7124977, 17.4891586, -39.1511345, 39.1355095
24: -32.0727463, 11.8148975, -32.2040596, 11.8826389, -43.9553833, 44.0189590
25: -18.0918884, 25.3512135, -18.1340504, 25.4089737, -43.5008621, 43.4852638
26: -29.2374611, 26.8572063, -29.2601929, 26.9375534, -56.1750145, 56.1174011
27: -32.0725594, 16.4614487, -32.1478806, 16.5284462, -47.7475853, 47.7580566
28: -21.5153999, 21.6163940, -21.5551796, 21.6870060, -43.2024078, 43.1715736
29: -23.6816959, 22.1756153, -23.7449074, 22.2186050, -45.9002991, 45.9205246
30: -29.5824966, 16.7807465, -29.6871376, 16.8521652, -45.8989296, 45.9238777
31: -26.3173313, 18.9931183, -26.4410419, 19.0833416, -45.4006729, 45.4341583
32: -42.2153702, 8.4307823, -42.2832909, 8.4973440, -47.5675201, 47.5584564
33: -72.3190308, -5.7192192, -72.4608459, -5.6090460, -61.2833862, 61.3165588
34: -56.4949493, -5.5662441, -56.5475693, -5.4811611, -43.6308708, 43.6009026
35: -50.1327896, -0.0372715, -50.2213974, 0.0482759, -48.2423706, 48.2647972
36: -47.7756119, 4.8351660, -47.8322639, 4.9458065, -52.0242920, 51.9694290
37: -83.6224518, -17.5557289, -83.7337341, -17.4596558, -58.4275970, 58.3866234
38: -58.6513481, 3.0977125, -58.7489281, 3.2302561, -61.2972641, 61.2570953
39: -78.9044800, -11.6918535, -79.0717010, -11.5854282, -65.2611465, 65.3255386
40: -67.6167297, -18.3839417, -67.7234802, -18.3274708, -41.1805916, 41.1834259
41: -55.1522980, -6.9324942, -55.2260246, -6.8330507, -42.2890511, 42.2108459
42: -33.9468880, 6.7614565, -33.9581871, 6.8119574, -37.6862373, 37.6358948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9592727, upper bound: 44.9967231
time: 27.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9596313, upper bound: 45.0354286
time: 27.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.0791817, 17.0803356, -28.1569366, 17.0914421, -44.3887634, 44.4649200
1: -13.6515722, 17.0815926, -13.6871862, 17.0817451, -30.7333183, 30.7687798
2: -14.0611229, 21.6608505, -14.0977077, 21.6581688, -35.4780350, 35.5335503
3: -12.8818855, 23.4277916, -12.9103928, 23.4307137, -36.3125992, 36.3381844
4: -21.5511875, 18.4577446, -21.5817947, 18.4549561, -40.0061417, 40.0395393
5: -11.9812374, 22.8419342, -12.0183554, 22.8283272, -34.8095627, 34.8602905
6: -50.6985245, -3.6136642, -50.7838364, -3.5576582, -40.6022186, 40.6114120
7: -16.3694077, 18.4621735, -16.4091835, 18.4370079, -34.8064156, 34.8713570
8: -18.2802162, 21.3025246, -18.3381348, 21.3454266, -39.6256409, 39.6406593
9: -16.6990719, 23.2816544, -16.7346420, 23.2831726, -38.6474609, 38.6663246
10: -24.2787437, 38.5091248, -24.3256569, 38.5381165, -61.8349457, 61.8437386
11: -24.7737408, 17.5960865, -24.7938328, 17.6015472, -42.3752899, 42.3899193
12: -28.6639996, 20.0960808, -28.6610947, 20.1361580, -46.9439316, 46.8851624
13: -32.9341049, 28.7861385, -32.9442520, 28.8033180, -61.7374229, 61.7303925
14: -23.4715347, 39.1937408, -23.5513840, 39.2385483, -60.1052322, 60.1173782
15: -18.9262791, 25.8451004, -18.9650574, 25.8900738, -44.8163528, 44.8101578
16: -32.7189255, 19.9322128, -32.7471390, 19.9153862, -52.6343117, 52.6793518
17: -17.7701302, 38.4760170, -17.7999706, 38.5044136, -55.3071289, 55.3072929
18: -25.8094959, 19.6154060, -25.8447418, 19.6235771, -45.4330750, 45.4601479
19: -26.4443550, 12.4729052, -26.4869118, 12.5094452, -38.9538002, 38.9598160
20: -21.1304569, 20.4246616, -21.1626511, 20.4585018, -41.5889587, 41.5873108
21: -25.7287807, 18.8608341, -25.7856293, 18.9071598, -44.6359406, 44.6464615
22: -22.1614151, 24.5098362, -22.1648102, 24.5402260, -46.7016411, 46.6746445
23: -21.7140446, 17.4924812, -21.7162743, 17.5119057, -39.2259521, 39.2087555
24: -32.1421204, 11.8902435, -32.2078857, 11.9077168, -44.0498352, 44.0981293
25: -18.1442566, 25.4130936, -18.1393375, 25.4292526, -43.5735092, 43.5524292
26: -29.3043461, 26.9434700, -29.2659130, 26.9665508, -56.2708969, 56.2093811
27: -32.1334877, 16.5356216, -32.1517029, 16.5530453, -47.8328094, 47.8309174
28: -21.5675011, 21.6891518, -21.5594635, 21.7112522, -43.2787552, 43.2486153
29: -23.7369614, 22.2217255, -23.7484798, 22.2336025, -45.9705658, 45.9702072
30: -29.6268253, 16.8468704, -29.6902847, 16.8735352, -45.9655457, 45.9923248
31: -26.3754616, 19.0661430, -26.4477539, 19.1075478, -45.4830093, 45.5138969
32: -42.2368622, 8.4605923, -42.2861023, 8.5059175, -47.6040649, 47.5935287
33: -72.3753662, -5.6505003, -72.4664841, -5.5867548, -61.3646927, 61.3896790
34: -56.5203857, -5.5171814, -56.5504341, -5.4655285, -43.6734314, 43.6520996
35: -50.1723404, 0.0083714, -50.2255096, 0.0633526, -48.2991638, 48.3155518
36: -47.8233070, 4.9064884, -47.8368607, 4.9697161, -52.0963821, 52.0422516
37: -83.6995850, -17.4745159, -83.7397995, -17.4326763, -58.5342178, 58.4576454
38: -58.7171669, 3.1915855, -58.7542915, 3.2603741, -61.3947525, 61.3470306
39: -78.9729462, -11.6308155, -79.0779953, -11.5653391, -65.3519897, 65.3848419
40: -67.6573944, -18.3419762, -67.7279053, -18.3136978, -41.2393074, 41.2220955
41: -55.1896935, -6.8584433, -55.2287292, -6.8088894, -42.3515358, 42.2771606
42: -33.9682426, 6.8113546, -33.9604111, 6.8278599, -37.7323875, 37.6870728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1764

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9950180, upper bound: 44.9968609
time: 51.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9950413, upper bound: 45.0354352
time: 41.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.1328697, 17.1241760, -27.9912834, 16.9412460, -44.2873077, 44.3428917
1: -13.6688633, 17.0968666, -13.5942917, 16.9353371, -30.6042004, 30.6911583
2: -14.0769234, 21.6723976, -13.9903955, 21.4964123, -35.3348885, 35.4385033
3: -12.8869886, 23.4351730, -12.8125706, 23.2824059, -36.1693954, 36.2477417
4: -21.5588379, 18.4384155, -21.4547958, 18.3154449, -39.8742828, 39.8932114
5: -11.9962463, 22.8498116, -11.9211330, 22.6922302, -34.6884766, 34.7709427
6: -50.7922440, -3.5823946, -50.6241684, -3.6623220, -40.5688400, 40.4685173
7: -16.3887138, 18.4616814, -16.3140526, 18.2933159, -34.6820297, 34.7757339
8: -18.3142929, 21.3363304, -18.2194672, 21.1728477, -39.4871407, 39.5557976
9: -16.6767693, 23.3273239, -16.5382996, 23.1197052, -38.4596214, 38.5344772
10: -24.2513943, 38.5720062, -24.0755253, 38.3082962, -61.5737762, 61.6870346
11: -24.7818108, 17.5748329, -24.6546326, 17.4884834, -42.2702942, 42.2294655
12: -28.6133633, 20.1250916, -28.4422989, 19.9786301, -46.7349586, 46.7191391
13: -32.9133263, 28.8178730, -32.7971725, 28.6820507, -61.5953751, 61.6150436
14: -23.4750061, 39.2646027, -23.2241592, 39.0583572, -59.9245987, 59.8930359
15: -18.9433174, 25.8781815, -18.8260765, 25.7730656, -44.7163849, 44.7042580
16: -32.7164307, 19.9783916, -32.6019173, 19.7544670, -52.4708977, 52.5803070
17: -17.7617035, 38.5414391, -17.5876503, 38.3523254, -55.1461067, 55.1853409
18: -25.8609486, 19.5965271, -25.6754837, 19.5203323, -45.3812790, 45.2720108
19: -26.5131168, 12.4751282, -26.2966156, 12.4103384, -38.9234543, 38.7717438
20: -21.1946411, 20.4327354, -20.9613552, 20.3555393, -41.5501785, 41.3940887
21: -25.8048153, 18.8723831, -25.5503101, 18.7876396, -44.5924530, 44.4226913
22: -22.2217331, 24.5164528, -21.9799232, 24.4665928, -46.6883240, 46.4963760
23: -21.7342148, 17.4851151, -21.5976276, 17.4086113, -39.1428261, 39.0827408
24: -32.2295914, 11.8646898, -32.0010529, 11.7800255, -44.0096169, 43.8657417
25: -18.1767178, 25.4020214, -17.9955997, 25.3372688, -43.5139847, 43.3976212
26: -29.3222008, 26.9272709, -29.0691605, 26.8153801, -56.1375809, 55.9964294
27: -32.1829376, 16.5125427, -31.9759254, 16.4278584, -47.8169823, 47.6327400
28: -21.5952606, 21.6809540, -21.4166832, 21.6076527, -43.2029114, 43.0976372
29: -23.7761765, 22.2131538, -23.5573425, 22.1446571, -45.9208336, 45.7704964
30: -29.6898422, 16.8499527, -29.5068836, 16.7534866, -45.9060898, 45.8156509
31: -26.4715538, 19.0753002, -26.2302017, 19.0106812, -45.4822350, 45.3055038
32: -42.2833443, 8.4940214, -42.1209450, 8.4222574, -47.5442505, 47.4559898
33: -72.5150757, -5.6296749, -72.2287140, -5.7332535, -61.3522797, 61.1673431
34: -56.6078682, -5.4905005, -56.4035873, -5.5460148, -43.6802139, 43.5128517
35: -50.2837296, 0.0373707, -50.0414886, -0.0132999, -48.3254623, 48.1152840
36: -47.9021797, 4.9364090, -47.6444244, 4.8821144, -52.0802612, 51.8729858
37: -83.7946854, -17.4730797, -83.5170593, -17.5526123, -58.4353180, 58.2278709
38: -58.8565559, 3.2219887, -58.4818954, 3.1362019, -61.4105835, 61.1120987
39: -79.1214066, -11.5940027, -78.8134384, -11.6530800, -65.4124680, 65.1554947
40: -67.7388763, -18.3368626, -67.5748062, -18.4022408, -41.1806984, 41.0870171
41: -55.2408409, -6.8423519, -55.1060181, -6.9223833, -42.2225723, 42.1792946
42: -33.9627266, 6.8072386, -33.8785477, 6.7197075, -37.6114044, 37.5936737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0325312, upper bound: 44.7371227
time: 48.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0347933, upper bound: 44.8670757
time: 37.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.1569519, 17.1269855, -28.0697098, 16.9707947, -44.3490982, 44.4267921
1: -13.6838274, 17.0985146, -13.6407728, 16.9606323, -30.6444588, 30.7392883
2: -14.0908766, 21.6740532, -14.0333385, 21.5156975, -35.3756409, 35.4858170
3: -12.9038086, 23.4383736, -12.8633041, 23.3157082, -36.2195168, 36.3016777
4: -21.5754356, 18.4411316, -21.5078087, 18.3321419, -39.9075775, 39.9489403
5: -12.0120039, 22.8528728, -11.9689236, 22.7244682, -34.7364731, 34.8217964
6: -50.7942657, -3.5629525, -50.6447105, -3.6027327, -40.6276550, 40.5096474
7: -16.4012737, 18.4646397, -16.3532372, 18.3164597, -34.7177353, 34.8178787
8: -18.3310013, 21.3389568, -18.2727585, 21.1858521, -39.5168533, 39.6117172
9: -16.7028141, 23.3306198, -16.6188583, 23.1655369, -38.5321732, 38.6120758
10: -24.2798023, 38.5784378, -24.1621170, 38.3708801, -61.6654358, 61.7724876
11: -24.7845993, 17.5972672, -24.6953201, 17.5579109, -42.3425102, 42.2925873
12: -28.6267281, 20.1339645, -28.4854317, 20.0276527, -46.7972374, 46.7674904
13: -32.9370193, 28.8238697, -32.8710938, 28.7285290, -61.6655502, 61.6949615
14: -23.5128918, 39.2676239, -23.3433228, 39.1049232, -60.0106697, 60.0070343
15: -18.9648209, 25.8817406, -18.8933811, 25.7936821, -44.7585030, 44.7751236
16: -32.7368889, 19.9829655, -32.6701508, 19.8064327, -52.5433197, 52.6531143
17: -17.7880669, 38.5441017, -17.6713638, 38.3932228, -55.2139359, 55.2673683
18: -25.8649750, 19.6110821, -25.7079487, 19.5664730, -45.4314499, 45.3190308
19: -26.5176849, 12.5000849, -26.3558865, 12.4847059, -39.0023918, 38.8559723
20: -21.1988468, 20.4535332, -21.0117016, 20.4181175, -41.6169662, 41.4652328
21: -25.8100624, 18.8977985, -25.6111355, 18.8634453, -44.6735077, 44.5089340
22: -22.2263756, 24.5314007, -22.0341511, 24.5126305, -46.7390060, 46.5655518
23: -21.7380333, 17.5078659, -21.6497536, 17.4780807, -39.2161140, 39.1576195
24: -32.2334595, 11.8898067, -32.0705376, 11.8554106, -44.0888710, 43.9603424
25: -18.1820030, 25.4222679, -18.0479603, 25.3991432, -43.5811462, 43.4702301
26: -29.3279400, 26.9562912, -29.1360703, 26.9017181, -56.2296600, 56.0923615
27: -32.1867714, 16.5371857, -32.0368309, 16.5021114, -47.8899193, 47.7180519
28: -21.5995350, 21.7052383, -21.4687881, 21.6804581, -43.2799911, 43.1740265
29: -23.7797680, 22.2281685, -23.6126404, 22.1907883, -45.9705582, 45.8408089
30: -29.6930008, 16.8714085, -29.5512238, 16.8196507, -45.9745865, 45.8823547
31: -26.4782677, 19.0995026, -26.2884121, 19.0837345, -45.5620041, 45.3879166
32: -42.2862091, 8.5026379, -42.1424103, 8.4521160, -47.5791321, 47.4924660
33: -72.5207443, -5.6073904, -72.2849655, -5.6644421, -61.4254761, 61.2486191
34: -56.6107521, -5.4748125, -56.4290543, -5.4969273, -43.7314606, 43.5554390
35: -50.2878494, 0.0525045, -50.0810509, 0.0323677, -48.3762512, 48.1720352
36: -47.9067688, 4.9603043, -47.6921234, 4.9534483, -52.1531372, 51.9449844
37: -83.8007965, -17.4461098, -83.5941544, -17.4713306, -58.5064163, 58.3344421
38: -58.8619270, 3.2521353, -58.5477905, 3.2301540, -61.5005951, 61.2096481
39: -79.1276855, -11.5739603, -78.8818741, -11.5920553, -65.4717865, 65.2462540
40: -67.7433167, -18.3230515, -67.6154633, -18.3602219, -41.2194710, 41.1457405
41: -55.2435684, -6.8181524, -55.1434135, -6.8482933, -42.2889900, 42.2419243
42: -33.9649658, 6.8232126, -33.8999329, 6.7696428, -37.6626511, 37.6398849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0325354, upper bound: 44.7723659
time: 91.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0347997, upper bound: 44.9024092
time: 64.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.8918839, 17.0469589, -28.0389099, 17.0215607, -44.1159973, 44.3084221
1: -13.5294924, 17.0494576, -13.6307268, 17.0363159, -30.5658073, 30.6801834
2: -13.9465160, 21.6331558, -14.0441208, 21.6099911, -35.3051682, 35.4480667
3: -12.7720537, 23.3803520, -12.8637915, 23.3886318, -36.1606865, 36.2441444
4: -21.4050789, 18.4265785, -21.5196571, 18.4356461, -39.8407249, 39.9462357
5: -11.8670158, 22.7840576, -11.9669685, 22.7891235, -34.6561394, 34.7510262
6: -50.7422104, -3.7144775, -50.6712112, -3.6172371, -40.5857468, 40.3893585
7: -16.2506218, 18.4053307, -16.3552723, 18.3997021, -34.6503220, 34.7606049
8: -18.1445160, 21.3207359, -18.2746353, 21.2740421, -39.4185562, 39.5953712
9: -16.5411339, 23.2207680, -16.6638870, 23.2211361, -38.4309769, 38.5235901
10: -24.1330643, 38.4480095, -24.2473869, 38.4506149, -61.6000061, 61.6892242
11: -24.7343140, 17.4620857, -24.7560349, 17.5567436, -42.2910576, 42.2181206
12: -28.5965767, 19.9992447, -28.6237888, 20.0867767, -46.8244629, 46.7531738
13: -32.7727165, 28.7226639, -32.8733521, 28.7556381, -61.5283546, 61.5960159
14: -23.2823181, 39.1782227, -23.4282780, 39.1566925, -59.8373489, 59.9627380
15: -18.8248100, 25.8147011, -18.9044113, 25.8224564, -44.6472664, 44.7191124
16: -32.5541687, 19.8457680, -32.6718750, 19.8508530, -52.4050217, 52.5176430
17: -17.6253605, 38.4498901, -17.7236519, 38.4275589, -55.0883560, 55.1944199
18: -25.7869968, 19.5174217, -25.7690659, 19.5847931, -45.3717880, 45.2864876
19: -26.4069271, 12.3425770, -26.3941174, 12.4505825, -38.8575096, 38.7366943
20: -21.0887985, 20.2994461, -21.0691814, 20.3998013, -41.4886017, 41.3686295
21: -25.6977119, 18.7204952, -25.6748734, 18.8408928, -44.5386047, 44.3953705
22: -22.0938129, 24.3875580, -22.0762291, 24.4895859, -46.5833969, 46.4637871
23: -21.6458378, 17.3555489, -21.6778450, 17.4493828, -39.0952225, 39.0333939
24: -32.1134796, 11.7427559, -32.1030464, 11.8512383, -43.9647179, 43.8458023
25: -18.0623302, 25.2571411, -18.0816174, 25.3734226, -43.4357529, 43.3387604
26: -29.1721878, 26.7335205, -29.2069817, 26.8887177, -56.0609055, 55.9405022
27: -32.0668793, 16.3754292, -32.0845337, 16.4915409, -47.7098351, 47.6018791
28: -21.4871521, 21.5212288, -21.5080757, 21.6452217, -43.1323738, 43.0293045
29: -23.6780663, 22.0932236, -23.6759300, 22.1881065, -45.8661728, 45.7691536
30: -29.6241226, 16.7241993, -29.6015015, 16.8139076, -45.8937378, 45.7807198
31: -26.3561344, 18.9236794, -26.3234730, 19.0434952, -45.3996277, 45.2471542
32: -42.2397728, 8.3918266, -42.2049522, 8.4603977, -47.5535011, 47.4395256
33: -72.3872375, -5.7639008, -72.3102875, -5.6463728, -61.3058472, 61.1166077
34: -56.5126762, -5.6319094, -56.4497910, -5.5194416, -43.6088333, 43.4362602
35: -50.1666489, -0.0848904, -50.0984688, 0.0176401, -48.2408142, 48.0923080
36: -47.7711754, 4.7650509, -47.7298546, 4.9018555, -51.9736328, 51.7927475
37: -83.6333771, -17.5907440, -83.6180573, -17.4909477, -58.3782349, 58.2299881
38: -58.6558189, 3.0107622, -58.5904083, 3.1773605, -61.2309570, 61.0062408
39: -78.9786682, -11.6975298, -78.9057083, -11.6101303, -65.2993546, 65.1483459
40: -67.6581573, -18.3923950, -67.6259918, -18.3484764, -41.1917496, 41.0729332
41: -55.1692390, -6.9780922, -55.1598740, -6.8738222, -42.2535553, 42.0918350
42: -33.9271088, 6.6922779, -33.9443970, 6.7773447, -37.6292191, 37.5593948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9557441, upper bound: 44.8812007
time: 62.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9579642, upper bound: 45.0334702
time: 53.08 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.9826927, 17.0770016, -28.0670738, 17.0247250, -44.2140274, 44.3734207
1: -13.5824909, 17.0757370, -13.6479216, 17.0381699, -30.6206608, 30.7236595
2: -13.9956570, 21.6522999, -14.0597925, 21.6118965, -35.3601608, 35.4893303
3: -12.8283005, 23.4159679, -12.8825054, 23.3925056, -36.2208061, 36.2984734
4: -21.4639206, 18.4452972, -21.5379543, 18.4389038, -39.9028244, 39.9832535
5: -11.9183798, 22.8157692, -11.9839811, 22.7927265, -34.7111053, 34.7997513
6: -50.7661057, -3.6463523, -50.6737823, -3.5958228, -40.6334000, 40.4609566
7: -16.2923393, 18.4284668, -16.3686924, 18.4030266, -34.6953659, 34.7971573
8: -18.2022858, 21.3340492, -18.2928276, 21.2769623, -39.4792480, 39.6268768
9: -16.6289692, 23.2687836, -16.6924591, 23.2250004, -38.5178375, 38.6006012
10: -24.2258301, 38.5112839, -24.2780895, 38.4578705, -61.6965027, 61.7841454
11: -24.7774544, 17.5399723, -24.7595081, 17.5817108, -42.3591652, 42.2994804
12: -28.6452942, 20.0526047, -28.6390648, 20.0966358, -46.8802948, 46.8214226
13: -32.8521843, 28.7709751, -32.8989754, 28.7622032, -61.6143875, 61.6699524
14: -23.4131622, 39.2261810, -23.4699993, 39.1600571, -59.9654236, 60.0534439
15: -18.8985004, 25.8373795, -18.9278870, 25.8267479, -44.7252502, 44.7652664
16: -32.6319504, 19.8957710, -32.6953850, 19.8560677, -52.4880180, 52.5911560
17: -17.7182560, 38.4929085, -17.7532635, 38.4307747, -55.1823883, 55.2679710
18: -25.8205452, 19.5674515, -25.7734547, 19.6007423, -45.4212875, 45.3409042
19: -26.4689541, 12.4251270, -26.3993015, 12.4783983, -38.9473534, 38.8244286
20: -21.1420765, 20.3705406, -21.0739269, 20.4233704, -41.5654449, 41.4444656
21: -25.7604980, 18.8049202, -25.6806545, 18.8691711, -44.6296692, 44.4855728
22: -22.1486320, 24.4382439, -22.0812912, 24.5060425, -46.6546745, 46.5195351
23: -21.6996117, 17.4300880, -21.6823521, 17.4740562, -39.1736679, 39.1124420
24: -32.1856155, 11.8237123, -32.1076698, 11.8782797, -44.0638962, 43.9313812
25: -18.1162281, 25.3242550, -18.0875816, 25.3955784, -43.5118065, 43.4118347
26: -29.2402496, 26.8284836, -29.2133083, 26.9205017, -56.1607513, 56.0417938
27: -32.1292725, 16.4555836, -32.0887756, 16.5181618, -47.7989464, 47.6833267
28: -21.5403290, 21.5989456, -21.5129128, 21.6713219, -43.2116508, 43.1118584
29: -23.7343807, 22.1448402, -23.6800652, 22.2047520, -45.9391327, 45.8249054
30: -29.6715374, 16.7969093, -29.6051693, 16.8374100, -45.9656334, 45.8567963
31: -26.4163628, 19.0044422, -26.3310776, 19.0704288, -45.4867935, 45.3355179
32: -42.2640724, 8.4270735, -42.2087631, 8.4702816, -47.5933151, 47.4828720
33: -72.4432907, -5.6895390, -72.3168030, -5.6220951, -61.3883743, 61.1967621
34: -56.5331497, -5.5794411, -56.4532700, -5.5029507, -43.6462708, 43.4914093
35: -50.2048149, -0.0383921, -50.1032524, 0.0330563, -48.2965088, 48.1456299
36: -47.8177948, 4.8411436, -47.7350578, 4.9272518, -52.0460434, 51.8720169
37: -83.7140732, -17.5047741, -83.6253738, -17.4622421, -58.4898834, 58.3101082
38: -58.7220039, 3.1127529, -58.5965805, 3.2099009, -61.3306427, 61.1099472
39: -79.0490646, -11.6313763, -78.9134064, -11.5883570, -65.3938904, 65.2155685
40: -67.6972122, -18.3457947, -67.6311646, -18.3331413, -41.2500916, 41.1225700
41: -55.2078705, -6.8996353, -55.1632156, -6.8483753, -42.3181229, 42.1671600
42: -33.9498672, 6.7484474, -33.9470444, 6.7947245, -37.6769867, 37.6174240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9942937, upper bound: 44.8812445
time: 47.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9965441, upper bound: 45.0335100
time: 42.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.0719948, 17.1162300, -28.0900192, 17.0258980, -44.3040962, 44.4303818
1: -13.6334925, 17.0994720, -13.6616364, 17.0383644, -30.6718559, 30.7611084
2: -14.0459499, 21.6887989, -14.0741854, 21.6119423, -35.4077873, 35.5355072
3: -12.8505669, 23.4302578, -12.8875713, 23.3929996, -36.2435684, 36.3178291
4: -21.5234756, 18.4598579, -21.5546227, 18.4385471, -39.9620209, 40.0144806
5: -11.9635878, 22.8460274, -11.9962692, 22.7928410, -34.7564278, 34.8422966
6: -50.7814369, -3.6245031, -50.6766319, -3.5912523, -40.6545258, 40.4811935
7: -16.3632545, 18.4725533, -16.3899651, 18.4029808, -34.7662354, 34.8625183
8: -18.2796974, 21.3583717, -18.3150902, 21.2775803, -39.5572777, 39.6734619
9: -16.6483612, 23.2908459, -16.6945877, 23.2253513, -38.5439148, 38.6285172
10: -24.2344246, 38.5264168, -24.2762146, 38.4584732, -61.7124329, 61.8113098
11: -24.7645092, 17.5248661, -24.7606049, 17.5740223, -42.3385315, 42.2854691
12: -28.6326027, 20.0887070, -28.6280632, 20.1116753, -46.8895569, 46.8438873
13: -32.8761711, 28.7850647, -32.9018517, 28.7649632, -61.6411362, 61.6869164
14: -23.4356918, 39.2241859, -23.4682903, 39.1603241, -59.9967957, 60.0560112
15: -18.9011269, 25.8751564, -18.9235497, 25.8386002, -44.7397270, 44.7987061
16: -32.6771393, 19.9427013, -32.7063599, 19.8568878, -52.5340271, 52.6490631
17: -17.7304306, 38.5067253, -17.7468891, 38.4308090, -55.1962128, 55.2856293
18: -25.8455048, 19.5757751, -25.7761688, 19.6010284, -45.4465332, 45.3519440
19: -26.4709034, 12.4193382, -26.3994503, 12.4746056, -38.9455109, 38.8187866
20: -21.1670303, 20.3846970, -21.0751839, 20.4250565, -41.5920868, 41.4598808
21: -25.7729397, 18.8164215, -25.6822128, 18.8704300, -44.6433716, 44.4986343
22: -22.1902637, 24.4849052, -22.0830383, 24.5196419, -46.7099075, 46.5679436
23: -21.6959877, 17.4354496, -21.6827965, 17.4729900, -39.1689758, 39.1182480
24: -32.1686630, 11.8237419, -32.1091728, 11.8752928, -44.0439568, 43.9329147
25: -18.1438522, 25.3586159, -18.0888996, 25.4041195, -43.5479736, 43.4475174
26: -29.2879810, 26.8653183, -29.2159805, 26.9285507, -56.2165298, 56.0812988
27: -32.1346970, 16.4718666, -32.0910454, 16.5202465, -47.8237686, 47.7079430
28: -21.5618286, 21.6300011, -21.5136261, 21.6778488, -43.2396774, 43.1436272
29: -23.7480431, 22.1800404, -23.6806126, 22.2141266, -45.9621696, 45.8606529
30: -29.6655369, 16.8040562, -29.6066971, 16.8358612, -45.9642258, 45.8684540
31: -26.4333286, 19.0222454, -26.3324776, 19.0734138, -45.5067444, 45.3547211
32: -42.2860603, 8.4722576, -42.2112656, 8.4837189, -47.6241150, 47.5260658
33: -72.4693451, -5.6598787, -72.3169174, -5.6163263, -61.4232330, 61.2322388
34: -56.5962219, -5.5198307, -56.4546928, -5.4855537, -43.7299271, 43.5495186
35: -50.2556610, 0.0154238, -50.1043587, 0.0480347, -48.3631439, 48.1990547
36: -47.8772888, 4.8875237, -47.7366142, 4.9395914, -52.1188507, 51.9234543
37: -83.7310333, -17.5218048, -83.6255493, -17.4707146, -58.5102234, 58.3100166
38: -58.8093834, 3.1573029, -58.6000214, 3.2207422, -61.4379578, 61.1653442
39: -79.0664520, -11.6338425, -78.9139252, -11.5914383, -65.4092331, 65.2241287
40: -67.7116547, -18.3568764, -67.6351013, -18.3392944, -41.2554855, 41.1171150
41: -55.2123337, -6.8895521, -55.1659012, -6.8476000, -42.3284912, 42.1880608
42: -33.9589920, 6.7736511, -33.9475327, 6.8008156, -37.6893806, 37.6420288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9940052, upper bound: 44.8824626
time: 50.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9962252, upper bound: 45.0347612
time: 25.24 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.1630058, 17.1462364, -28.1182060, 17.0290508, -44.4021645, 44.4953651
1: -13.6865225, 17.1257706, -13.6787996, 17.0402222, -30.7267456, 30.8045692
2: -14.0951166, 21.7079449, -14.0898724, 21.6138668, -35.4627953, 35.5767593
3: -12.9068127, 23.4658489, -12.9062958, 23.3968887, -36.3037033, 36.3721466
4: -21.5824280, 18.4785748, -21.5729218, 18.4417801, -40.0242081, 40.0514984
5: -12.0149775, 22.8777428, -12.0132904, 22.7964382, -34.8114166, 34.8910332
6: -50.8053360, -3.5563264, -50.6791840, -3.5698242, -40.7021866, 40.5528870
7: -16.4050026, 18.4956665, -16.4033737, 18.4062958, -34.8112984, 34.8990402
8: -18.3375664, 21.3716965, -18.3332767, 21.2805119, -39.6180801, 39.7049713
9: -16.7362118, 23.3388062, -16.7231407, 23.2292194, -38.6308250, 38.7055054
10: -24.3271961, 38.5896187, -24.3069286, 38.4657135, -61.8089447, 61.9062424
11: -24.8076591, 17.6027813, -24.7640858, 17.5989838, -42.4066429, 42.3668671
12: -28.6813431, 20.1420517, -28.6433563, 20.1215439, -46.9454422, 46.9122658
13: -32.9556961, 28.8332329, -32.9274597, 28.7714939, -61.7271881, 61.7606926
14: -23.5665398, 39.2721291, -23.5099945, 39.1636543, -60.1249008, 60.1466560
15: -18.9748611, 25.8978729, -18.9470577, 25.8429031, -44.8177643, 44.8449326
16: -32.7549019, 19.9926510, -32.7298660, 19.8621025, -52.6170044, 52.7225189
17: -17.8233891, 38.5497208, -17.7764931, 38.4340096, -55.2902603, 55.3591805
18: -25.8790226, 19.6258698, -25.7805653, 19.6169815, -45.4960022, 45.4064331
19: -26.5328865, 12.5019093, -26.4046631, 12.5024300, -39.0353165, 38.9065704
20: -21.2202568, 20.4558201, -21.0799179, 20.4486141, -41.6688690, 41.5357361
21: -25.8356686, 18.9008522, -25.6880035, 18.8987122, -44.7343826, 44.5888557
22: -22.2450256, 24.5355530, -22.0880928, 24.5361061, -46.7811317, 46.6236458
23: -21.7497406, 17.5099792, -21.6873150, 17.4976540, -39.2473946, 39.1972961
24: -32.2407646, 11.9046869, -32.1137924, 11.9023399, -44.1431046, 44.0184784
25: -18.1976833, 25.4257317, -18.0948601, 25.4262352, -43.6239166, 43.5205917
26: -29.3559742, 26.9602852, -29.2223148, 26.9603271, -56.3162994, 56.1826019
27: -32.1970978, 16.5520287, -32.0952606, 16.5468826, -47.9128723, 47.7893829
28: -21.6149292, 21.7077579, -21.5184631, 21.7039680, -43.3188972, 43.2262192
29: -23.8043423, 22.2316628, -23.6847706, 22.2307739, -46.0351181, 45.9164352
30: -29.7129383, 16.8768082, -29.6103706, 16.8593845, -46.0361328, 45.9445610
31: -26.4935150, 19.1030159, -26.3400803, 19.1003418, -45.5938568, 45.4430962
32: -42.3103638, 8.5075941, -42.2150726, 8.4936209, -47.6639214, 47.5694962
33: -72.5253983, -5.5854797, -72.3234024, -5.5920620, -61.5057907, 61.3124161
34: -56.6167030, -5.4673738, -56.4581299, -5.4690857, -43.7673225, 43.6046944
35: -50.2938042, 0.0619326, -50.1091537, 0.0634594, -48.4188461, 48.2523842
36: -47.9238129, 4.9636211, -47.7418327, 4.9649868, -52.1912003, 52.0027847
37: -83.8117599, -17.4357529, -83.6328888, -17.4419918, -58.6218491, 58.3901749
38: -58.8754883, 3.2593136, -58.6061935, 3.2533016, -61.5375595, 61.2690506
39: -79.1368942, -11.5676603, -78.9215851, -11.5696211, -65.5037918, 65.2913361
40: -67.7507629, -18.3102493, -67.6403351, -18.3239994, -41.3138504, 41.1667480
41: -55.2509003, -6.8110704, -55.1692276, -6.8221436, -42.3930435, 42.2633896
42: -33.9817352, 6.8298674, -33.9501648, 6.8182220, -37.7371559, 37.7001190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=210, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 920

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0325354, upper bound: 44.8825027
time: 52.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0347997, upper bound: 45.0347997
time: 30.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 85.44 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9417538, upper bound: 44.9760237
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9417538, upper bound: 45.0144944
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9417538, upper bound: 44.9761802
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9775451, upper bound: 45.0144965
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9210467, upper bound: 44.9954294
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9214209, upper bound: 45.0341410
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9567934, upper bound: 44.9955723
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9568165, upper bound: 45.0341457
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9800037, upper bound: 44.9773150
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9804278, upper bound: 45.0157797
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0157590, upper bound: 44.9774643
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0157822, upper bound: 45.0157822
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9592727, upper bound: 44.9967231
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9596313, upper bound: 45.0354286
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9950180, upper bound: 44.9968609
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9950413, upper bound: 45.0354352
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0325312, upper bound: 44.7371227
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0347933, upper bound: 44.8670757
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0325354, upper bound: 44.7723659
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0347997, upper bound: 44.9024092
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9557441, upper bound: 44.8812007
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9579642, upper bound: 45.0334702
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9942937, upper bound: 44.8812445
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9965441, upper bound: 45.0335100
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9940052, upper bound: 44.8824626
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -44.9962252, upper bound: 45.0347612
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0325354, upper bound: 44.8825027
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 85.44
Output dim: 14, lower bound: -45.0347997, upper bound: 45.0347997

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -27.8183823, 16.9812489, -27.9875946, 17.0175819, -44.0456429, 44.1906052
1: -13.4999084, 17.0061150, -13.6010218, 17.0332355, -30.5331440, 30.6071358
2: -13.9175949, 21.5857487, -14.0145588, 21.6067581, -35.2763596, 35.3673973
3: -12.7514286, 23.3441200, -12.8392067, 23.3832302, -36.1346588, 36.1833267
4: -21.3784676, 18.4075050, -21.4935169, 18.4272251, -39.8056946, 39.9010239
5: -11.8356285, 22.7473545, -11.9341316, 22.7846565, -34.6202850, 34.6814880
6: -50.6384773, -3.7647657, -50.6681862, -3.6673560, -40.4253082, 40.3420601
7: -16.2166157, 18.3715000, -16.3221645, 18.3960514, -34.6126671, 34.6936646
8: -18.0905914, 21.2515297, -18.2183247, 21.2683105, -39.3589020, 39.4698563
9: -16.5094261, 23.1654167, -16.6323738, 23.2171440, -38.3974152, 38.4394264
10: -24.0886574, 38.3675308, -24.2106514, 38.4411316, -61.5461502, 61.5641479
11: -24.7025032, 17.4621277, -24.7510300, 17.5476475, -42.2501526, 42.2131577
12: -28.5837078, 19.9569607, -28.6212826, 20.0530014, -46.7841949, 46.7066154
13: -32.7549438, 28.6768265, -32.8629456, 28.7382183, -61.4931641, 61.5397720
14: -23.1959038, 39.1009407, -23.3647499, 39.1524315, -59.7472839, 59.8216248
15: -18.7812538, 25.7636490, -18.8687363, 25.8160324, -44.5972862, 44.6323853
16: -32.5260239, 19.7828979, -32.6486778, 19.8430500, -52.3690720, 52.4315758
17: -17.5792427, 38.3779297, -17.6878681, 38.4233665, -55.0399170, 55.0815010
18: -25.7180462, 19.5097885, -25.7618389, 19.5776711, -45.2957153, 45.2716293
19: -26.3206730, 12.3198690, -26.3865852, 12.4233456, -38.7440186, 38.7064552
20: -21.0014648, 20.2751656, -21.0611153, 20.3739967, -41.3754616, 41.3362808
21: -25.5922241, 18.6871777, -25.6643238, 18.8037968, -44.3960190, 44.3515015
22: -22.0102787, 24.3653355, -22.0682068, 24.4647045, -46.4749832, 46.4335403
23: -21.6113319, 17.3413811, -21.6698208, 17.4408436, -39.0521774, 39.0112000
24: -32.0168457, 11.7320976, -32.0964813, 11.8363447, -43.8531914, 43.8285789
25: -18.0097961, 25.2482662, -18.0712776, 25.3592300, -43.3690262, 43.3195419
26: -29.1211491, 26.7230759, -29.1974659, 26.8708496, -55.9919968, 55.9205399
27: -32.0042648, 16.3632717, -32.0761337, 16.4756222, -47.5972748, 47.5786819
28: -21.4402866, 21.5058136, -21.5004997, 21.6279335, -43.0682220, 43.0063133
29: -23.6112194, 22.0876141, -23.6708946, 22.1775875, -45.7888069, 45.7585068
30: -29.5406895, 16.6992493, -29.5943604, 16.7929764, -45.7970352, 45.7489738
31: -26.2392578, 18.8928070, -26.3103752, 19.0088043, -45.2480621, 45.2031822
32: -42.1684418, 8.3496084, -42.2001686, 8.4236736, -47.4459343, 47.3953247
33: -72.2363281, -5.8249969, -72.3022308, -5.7113686, -61.0897675, 61.0468292
34: -56.4109459, -5.6794491, -56.4451523, -5.5696821, -43.4528503, 43.3847656
35: -50.0432930, -0.1386499, -50.0923080, -0.0400171, -48.0583191, 48.0329742
36: -47.6690521, 4.7109585, -47.7235260, 4.8436022, -51.8118744, 51.7328491
37: -83.5240631, -17.6267433, -83.6111526, -17.5272427, -58.2323914, 58.1868591
38: -58.4972153, 2.9488716, -58.5789833, 3.1096010, -61.0052490, 60.9352875
39: -78.8159103, -11.7571440, -78.8969879, -11.6738958, -65.0723877, 65.0800476
40: -67.5627441, -18.4205856, -67.6190033, -18.3733959, -41.0752182, 41.0403595
41: -55.1088181, -7.0228357, -55.1571159, -6.9142427, -42.1485825, 42.0455551
42: -33.9148140, 6.6787853, -33.9407845, 6.7644196, -37.5984268, 37.5407066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7893924, upper bound: 45.0116209
time: 32.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9415516, upper bound: 45.0138581
time: 23.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -27.8966370, 17.0108490, -28.0116959, 17.0203972, -44.1294250, 44.2524414
1: -13.5463953, 17.0314140, -13.6160107, 17.0348835, -30.5812798, 30.6474247
2: -13.9605122, 21.6050415, -14.0285015, 21.6084404, -35.3236771, 35.4081268
3: -12.8021545, 23.3774681, -12.8560028, 23.3864346, -36.1885910, 36.2334709
4: -21.4314041, 18.4241924, -21.5101223, 18.4299431, -39.8613472, 39.9343147
5: -11.8834267, 22.7796345, -11.9499197, 22.7877159, -34.6711426, 34.7295532
6: -50.6590042, -3.7052212, -50.6702003, -3.6479220, -40.4664307, 40.4007835
7: -16.2558193, 18.3946648, -16.3347397, 18.3990078, -34.6548271, 34.7294044
8: -18.1437798, 21.2644997, -18.2350311, 21.2709522, -39.4147339, 39.4995308
9: -16.5899239, 23.2112770, -16.6584053, 23.2204437, -38.4749413, 38.5119438
10: -24.1751919, 38.4301987, -24.2390671, 38.4475937, -61.6315155, 61.6558418
11: -24.7432060, 17.5315685, -24.7538242, 17.5701008, -42.3133087, 42.2853928
12: -28.6268215, 20.0060005, -28.6346226, 20.0618725, -46.8324432, 46.7687798
13: -32.8288040, 28.7233601, -32.8866730, 28.7442207, -61.5730247, 61.6100311
14: -23.3150120, 39.1475449, -23.4026470, 39.1554565, -59.8612175, 59.9077492
15: -18.8484726, 25.7842331, -18.8901958, 25.8195953, -44.6680679, 44.6744308
16: -32.5942268, 19.8349228, -32.6690979, 19.8476696, -52.4418945, 52.5040207
17: -17.6629219, 38.4188499, -17.7142296, 38.4260254, -55.1218872, 55.1493416
18: -25.7504807, 19.5558662, -25.7658882, 19.5922489, -45.3427277, 45.3217545
19: -26.3799610, 12.3942375, -26.3911934, 12.4482880, -38.8282471, 38.7854309
20: -21.0518665, 20.3377380, -21.0653267, 20.3947754, -41.4466400, 41.4030647
21: -25.6531200, 18.7629814, -25.6695957, 18.8292084, -44.4823303, 44.4325790
22: -22.0645752, 24.4113846, -22.0728550, 24.4796505, -46.5442276, 46.4842377
23: -21.6634521, 17.4108524, -21.6736355, 17.4635963, -39.1270485, 39.0844879
24: -32.0863457, 11.8074789, -32.1003761, 11.8614674, -43.9478149, 43.9078560
25: -18.0622692, 25.3101387, -18.0766068, 25.3795280, -43.4417953, 43.3867455
26: -29.1881123, 26.8093987, -29.2031975, 26.8998299, -56.0879440, 56.0125961
27: -32.0651894, 16.4374695, -32.0799866, 16.5002556, -47.6826096, 47.6515884
28: -21.4924393, 21.5785751, -21.5048122, 21.6522179, -43.1446571, 43.0833893
29: -23.6665630, 22.1337357, -23.6744957, 22.1925774, -45.8591385, 45.8082314
30: -29.5850410, 16.7653732, -29.5975380, 16.8144341, -45.8637085, 45.8174400
31: -26.2975235, 18.9658222, -26.3171291, 19.0329952, -45.3305206, 45.2829514
32: -42.1898956, 8.3793821, -42.2030182, 8.4322901, -47.4824142, 47.4301262
33: -72.2926941, -5.7562599, -72.3079071, -5.6890326, -61.1710510, 61.1199799
34: -56.4364624, -5.6303854, -56.4480362, -5.5540056, -43.4954643, 43.4359741
35: -50.0828705, -0.0930166, -50.0964355, -0.0249329, -48.1151352, 48.0837631
36: -47.7168274, 4.7822809, -47.7281227, 4.8675098, -51.8840179, 51.8057098
37: -83.6011887, -17.5454903, -83.6172791, -17.5002365, -58.3390350, 58.2579613
38: -58.5631142, 3.0428057, -58.5843506, 3.1397285, -61.1028595, 61.0253525
39: -78.8843842, -11.6960850, -78.9033203, -11.6538544, -65.1632233, 65.1394348
40: -67.6033783, -18.3786125, -67.6234589, -18.3595638, -41.1339417, 41.0790901
41: -55.1462631, -6.9487658, -55.1598587, -6.8900166, -42.2111969, 42.1119537
42: -33.9362030, 6.7286940, -33.9430084, 6.7803783, -37.6446457, 37.5919075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8246502, upper bound: 45.0116218
time: 46.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9769097, upper bound: 45.0138604
time: 29.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.8183823, 16.9812489, -28.0744514, 17.0835323, -44.1163788, 44.2725716
1: -13.4999084, 17.0061150, -13.6374722, 17.0775375, -30.5774460, 30.6435871
2: -13.9175949, 21.5857487, -14.0500078, 21.6540184, -35.3283958, 35.4005623
3: -12.7514286, 23.3441200, -12.8656664, 23.4218655, -36.1732941, 36.2097855
4: -21.3784676, 18.4075050, -21.5262260, 18.4484272, -39.8268967, 39.9337311
5: -11.8356285, 22.7473545, -11.9693565, 22.8205662, -34.6561966, 34.7167130
6: -50.6384773, -3.7647657, -50.7754517, -3.6081719, -40.4850159, 40.4582367
7: -16.2166157, 18.3715000, -16.3588104, 18.4298687, -34.6464844, 34.7303085
8: -18.0905914, 21.2515297, -18.2770462, 21.3380585, -39.4286499, 39.5285759
9: -16.5094261, 23.1654167, -16.6717453, 23.2746811, -38.4540329, 38.4775925
10: -24.0886574, 38.3675308, -24.2615891, 38.5221710, -61.6331024, 61.6169586
11: -24.7025032, 17.4621277, -24.7853642, 17.5563812, -42.2588844, 42.2474899
12: -28.5837078, 19.9569607, -28.6399384, 20.0998726, -46.8270493, 46.7236443
13: -32.7549438, 28.6768265, -32.8865166, 28.7860565, -61.5410004, 61.5633430
14: -23.1959038, 39.1009407, -23.4634399, 39.2311172, -59.8276749, 59.9210739
15: -18.7812538, 25.7636490, -18.9192429, 25.8692513, -44.6505051, 44.6828918
16: -32.5260239, 19.7828979, -32.6868134, 19.9035339, -52.4295578, 52.4697113
17: -17.5792427, 38.3779297, -17.7436104, 38.4974823, -55.1141663, 55.1378899
18: -25.7180462, 19.5097885, -25.8320923, 19.5893517, -45.3073959, 45.3418808
19: -26.3206730, 12.3198690, -26.4756241, 12.4545021, -38.7751770, 38.7954941
20: -21.0014648, 20.2751656, -21.1512451, 20.4072247, -41.4086914, 41.4264107
21: -25.5922241, 18.6871777, -25.7716408, 18.8460884, -44.4383125, 44.4588165
22: -22.0102787, 24.3653355, -22.1521015, 24.4916840, -46.5019608, 46.5174370
23: -21.6113319, 17.3413811, -21.7060719, 17.4602566, -39.0715866, 39.0474548
24: -32.0168457, 11.7320976, -32.1960449, 11.8527803, -43.8696251, 43.9281425
25: -18.0097961, 25.2482662, -18.1252403, 25.3735542, -43.3833504, 43.3735046
26: -29.1211491, 26.7230759, -29.2496071, 26.8904152, -56.0115662, 55.9726830
27: -32.0042648, 16.3632717, -32.1401749, 16.4940205, -47.6233864, 47.6489639
28: -21.4402866, 21.5058136, -21.5483551, 21.6485958, -43.0888824, 43.0541687
29: -23.6112194, 22.0876141, -23.7388439, 22.1888466, -45.8000641, 45.8264580
30: -29.5406895, 16.6992493, -29.6808872, 16.8249321, -45.8212585, 45.8333664
31: -26.2392578, 18.8928070, -26.4295864, 19.0477104, -45.2869682, 45.3223953
32: -42.1684418, 8.3496084, -42.2748680, 8.4715576, -47.4933090, 47.4709167
33: -72.2363281, -5.8249969, -72.4524078, -5.6444607, -61.1568680, 61.1972427
34: -56.4109459, -5.6794491, -56.5414429, -5.5186558, -43.5057297, 43.4854927
35: -50.0432930, -0.1386499, -50.2140503, 0.0145254, -48.1137619, 48.1554871
36: -47.6690521, 4.7109585, -47.8242378, 4.9026175, -51.8707886, 51.8356018
37: -83.5240631, -17.6267433, -83.7241669, -17.4862900, -58.2727356, 58.3023186
38: -58.4972153, 2.9488716, -58.7377129, 3.1798162, -61.0760574, 61.0939407
39: -78.8159103, -11.7571440, -79.0613556, -11.6089554, -65.1375122, 65.2460175
40: -67.5627441, -18.4205856, -67.7128906, -18.3402977, -41.1068230, 41.1363754
41: -55.1088181, -7.0228357, -55.2188721, -6.8649120, -42.1993828, 42.1115837
42: -33.9148140, 6.6787853, -33.9544182, 6.7844124, -37.6204529, 37.5555420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7686775, upper bound: 45.0312706
time: 50.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9207864, upper bound: 45.0335053
time: 53.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -27.8966370, 17.0108490, -28.0985374, 17.0863495, -44.2001610, 44.3344002
1: -13.5463953, 17.0314140, -13.6524487, 17.0791759, -30.6255722, 30.6838627
2: -13.9605122, 21.6050415, -14.0639629, 21.6557198, -35.3757019, 35.4413185
3: -12.8021545, 23.3774681, -12.8824854, 23.4250641, -36.2272186, 36.2599525
4: -21.4314041, 18.4241924, -21.5428314, 18.4511375, -39.8825417, 39.9670258
5: -11.8834267, 22.7796345, -11.9851322, 22.8236332, -34.7070618, 34.7647667
6: -50.6590042, -3.7052212, -50.7774734, -3.5887365, -40.5261345, 40.5169716
7: -16.2558193, 18.3946648, -16.3714027, 18.4328308, -34.6886520, 34.7660675
8: -18.1437798, 21.2644997, -18.2937584, 21.3406982, -39.4844780, 39.5582581
9: -16.5899239, 23.2112770, -16.6977921, 23.2779770, -38.5315437, 38.5501175
10: -24.1751919, 38.4301987, -24.2900200, 38.5285721, -61.7184830, 61.7086563
11: -24.7432060, 17.5315685, -24.7881641, 17.5788269, -42.3220329, 42.3197327
12: -28.6268215, 20.0060005, -28.6532860, 20.1088066, -46.8753357, 46.7857933
13: -32.8288040, 28.7233601, -32.9102020, 28.7920895, -61.6208954, 61.6335602
14: -23.3150120, 39.1475449, -23.5013428, 39.2341461, -59.9416275, 60.0071716
15: -18.8484726, 25.7842331, -18.9407463, 25.8728027, -44.7212753, 44.7249794
16: -32.5942268, 19.8349228, -32.7072525, 19.9081154, -52.5023422, 52.5421753
17: -17.6629219, 38.4188499, -17.7700081, 38.5001564, -55.1961441, 55.2057304
18: -25.7504807, 19.5558662, -25.8361359, 19.6038952, -45.3543777, 45.3920021
19: -26.3799610, 12.3942375, -26.4802094, 12.4794703, -38.8594322, 38.8744469
20: -21.0518665, 20.3377380, -21.1554508, 20.4279976, -41.4798660, 41.4931870
21: -25.6531200, 18.7629814, -25.7769012, 18.8714981, -44.5246201, 44.5398827
22: -22.0645752, 24.4113846, -22.1567383, 24.5066338, -46.5712090, 46.5681229
23: -21.6634521, 17.4108524, -21.7099018, 17.4829979, -39.1464500, 39.1207542
24: -32.0863457, 11.8074789, -32.1999283, 11.8779011, -43.9642487, 44.0074081
25: -18.0622692, 25.3101387, -18.1305466, 25.3938293, -43.4561005, 43.4406853
26: -29.1881123, 26.8093987, -29.2553444, 26.9193993, -56.1075134, 56.0647430
27: -32.0651894, 16.4374695, -32.1440163, 16.5186729, -47.7087173, 47.7218628
28: -21.4924393, 21.5785751, -21.5526638, 21.6729126, -43.1653519, 43.1312408
29: -23.6665630, 22.1337357, -23.7424297, 22.2038536, -45.8704147, 45.8761673
30: -29.5850410, 16.7653732, -29.6840382, 16.8463745, -45.8879433, 45.9018326
31: -26.2975235, 18.9658222, -26.4363079, 19.0719357, -45.3694611, 45.4021301
32: -42.1898956, 8.3793821, -42.2777100, 8.4801741, -47.5298195, 47.5057449
33: -72.2926941, -5.7562599, -72.4580917, -5.6221790, -61.2381592, 61.2704010
34: -56.4364624, -5.6303854, -56.5443344, -5.5029917, -43.5483322, 43.5366974
35: -50.0828705, -0.0930166, -50.2182045, 0.0296154, -48.1705704, 48.2062836
36: -47.7168274, 4.7822809, -47.8288345, 4.9265308, -51.9429321, 51.9084473
37: -83.6011887, -17.5454903, -83.7302856, -17.4593086, -58.3793640, 58.3733902
38: -58.5631142, 3.0428057, -58.7430573, 3.2099600, -61.1736298, 61.1839676
39: -78.8843842, -11.6960850, -79.0676498, -11.5888844, -65.2283554, 65.3053665
40: -67.6033783, -18.3786125, -67.7173233, -18.3264484, -41.1655579, 41.1750679
41: -55.1462631, -6.9487658, -55.2216072, -6.8406925, -42.2619934, 42.1779823
42: -33.9362030, 6.7286940, -33.9566650, 6.8003569, -37.6666489, 37.6067581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7686775, upper bound: 45.0312751
time: 54.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9561813, upper bound: 45.0335100
time: 116.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -27.9984894, 17.0505409, -28.0386772, 17.0219116, -44.2335663, 44.3125572
1: -13.6038742, 17.0561295, -13.6318960, 17.0352840, -30.6391582, 30.6880264
2: -14.0170031, 21.6414032, -14.0446138, 21.6087227, -35.3789597, 35.4548607
3: -12.8298683, 23.3940620, -12.8629770, 23.3876190, -36.2174873, 36.2570381
4: -21.4968987, 18.4407730, -21.5284653, 18.4301300, -39.9270287, 39.9692383
5: -11.9321976, 22.8093605, -11.9634514, 22.7883701, -34.7205658, 34.7728119
6: -50.6777039, -3.6748438, -50.6735802, -3.6414099, -40.4940643, 40.4338226
7: -16.3292332, 18.4387321, -16.3568554, 18.3993320, -34.7285652, 34.7955856
8: -18.2256603, 21.2891674, -18.2587814, 21.2718582, -39.4975204, 39.5479507
9: -16.6165848, 23.2355099, -16.6630421, 23.2213688, -38.5102997, 38.5443687
10: -24.1899719, 38.4459763, -24.2394733, 38.4490280, -61.6585770, 61.6862564
11: -24.7327099, 17.5249252, -24.7556114, 17.5649014, -42.2976112, 42.2805367
12: -28.6197548, 20.0462532, -28.6255589, 20.0778484, -46.8492203, 46.7972336
13: -32.8584213, 28.7391167, -32.8914528, 28.7474842, -61.6059036, 61.6305695
14: -23.3492241, 39.1469231, -23.4047279, 39.1560516, -59.9067192, 59.9148674
15: -18.8574257, 25.8241158, -18.8877907, 25.8321686, -44.6895943, 44.7119064
16: -32.6489792, 19.8798504, -32.6831436, 19.8490753, -52.4980545, 52.5629959
17: -17.6842957, 38.4347839, -17.7110672, 38.4266281, -55.1477585, 55.1726952
18: -25.7765884, 19.5681915, -25.7689323, 19.5939217, -45.3705101, 45.3371239
19: -26.3846722, 12.3966389, -26.3919697, 12.4473572, -38.8320312, 38.7886086
20: -21.0797195, 20.3603859, -21.0671120, 20.3992558, -41.4789734, 41.4274979
21: -25.6674824, 18.7830772, -25.6716805, 18.8333187, -44.5008011, 44.4547577
22: -22.1067371, 24.4626694, -22.0750237, 24.4947357, -46.6014709, 46.5376930
23: -21.6614838, 17.4213104, -21.6747456, 17.4644585, -39.1259422, 39.0960541
24: -32.0720673, 11.8130388, -32.1026001, 11.8603992, -43.9324646, 43.9156380
25: -18.0913811, 25.3497066, -18.0785770, 25.3899002, -43.4812813, 43.4282837
26: -29.2369614, 26.8548374, -29.2064362, 26.9106789, -56.1476402, 56.0612717
27: -32.0721588, 16.4596176, -32.0826378, 16.5042915, -47.7112846, 47.6847267
28: -21.5149632, 21.6145515, -21.5060558, 21.6605415, -43.1755066, 43.1206055
29: -23.6812229, 22.1744213, -23.6756001, 22.2035961, -45.8848190, 45.8500214
30: -29.5821533, 16.7790604, -29.5995598, 16.8149357, -45.8675079, 45.8366966
31: -26.3164997, 18.9912949, -26.3193550, 19.0386696, -45.3551712, 45.3106499
32: -42.2147102, 8.4299955, -42.2064819, 8.4470043, -47.5165405, 47.4818153
33: -72.3184509, -5.7209606, -72.3088684, -5.6813202, -61.2071381, 61.1624451
34: -56.4945374, -5.5674000, -56.4500580, -5.5357981, -43.5739441, 43.4980469
35: -50.1323318, -0.0383501, -50.0982094, -0.0096426, -48.1807098, 48.1397057
36: -47.7751808, 4.8334408, -47.7303123, 4.8813448, -51.9571381, 51.8636093
37: -83.6217651, -17.5577698, -83.6186676, -17.5069981, -58.3643951, 58.2668877
38: -58.6508331, 3.0954132, -58.5886497, 3.1529436, -61.2123184, 61.0943832
39: -78.9037781, -11.6934195, -78.9052277, -11.6551476, -65.1823349, 65.1558304
40: -67.6162720, -18.3851147, -67.6281281, -18.3642349, -41.1389236, 41.0843582
41: -55.1518974, -6.9343042, -55.1631203, -6.8880186, -42.2235641, 42.1417694
42: -33.9466782, 6.7601328, -33.9439087, 6.7878761, -37.6586304, 37.6233444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8275285, upper bound: 45.0128873
time: 31.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9797921, upper bound: 45.0151439
time: 28.50 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -28.0487137, 17.0769386, -27.9717674, 16.9946976, -44.2524223, 44.2763252
1: -13.6331816, 17.0795670, -13.5938578, 17.0106354, -30.6438179, 30.6734238
2: -14.0442896, 21.6587601, -14.0094290, 21.5912495, -35.3850327, 35.4405632
3: -12.8618631, 23.4235077, -12.8235226, 23.3552055, -36.2170677, 36.2470322
4: -21.5316238, 18.4542179, -21.4861259, 18.4141235, -39.9457474, 39.9403458
5: -11.9629574, 22.8380203, -11.9278450, 22.7597294, -34.7226868, 34.7658653
6: -50.6956863, -3.6366825, -50.6517029, -3.6901131, -40.4635391, 40.4449310
7: -16.3550396, 18.4585781, -16.3276901, 18.3791466, -34.7341843, 34.7862701
8: -18.2607803, 21.2992382, -18.2176704, 21.2611618, -39.5219421, 39.5169067
9: -16.6685600, 23.2774830, -16.6012497, 23.1766777, -38.5108376, 38.5299988
10: -24.2458572, 38.5013580, -24.1750717, 38.3922119, -61.6491089, 61.6814346
11: -24.7699203, 17.5693817, -24.7152958, 17.5094490, -42.2793694, 42.2846756
12: -28.6476059, 20.0854187, -28.5901661, 20.0334530, -46.8291473, 46.8036575
13: -32.9067421, 28.7790394, -32.8356247, 28.7052917, -61.6120338, 61.6146622
14: -23.4266453, 39.1901398, -23.3117485, 39.1111183, -59.9300156, 59.8728333
15: -18.9012184, 25.8404503, -18.8355942, 25.8130035, -44.7142220, 44.6760445
16: -32.6936913, 19.9266396, -32.6258163, 19.8036919, -52.4973831, 52.5524559
17: -17.7384109, 38.4724960, -17.6444893, 38.3862839, -55.1561890, 55.1464920
18: -25.8046112, 19.5983582, -25.7394218, 19.5583534, -45.3629646, 45.3377800
19: -26.4387417, 12.4431992, -26.3345318, 12.3897581, -38.8284988, 38.7777328
20: -21.1253548, 20.3994179, -21.0180740, 20.3489265, -41.4742813, 41.4174919
21: -25.7225475, 18.8306122, -25.6141720, 18.7742882, -44.4968338, 44.4447861
22: -22.1559639, 24.4922791, -22.0248795, 24.4590378, -46.6150017, 46.5171585
23: -21.7090988, 17.4661179, -21.6248360, 17.4126625, -39.1217613, 39.0909538
24: -32.1369247, 11.8613682, -32.0343704, 11.8045406, -43.9414673, 43.8957367
25: -18.1378250, 25.3894501, -18.0300369, 25.3430920, -43.4809189, 43.4194870
26: -29.2975464, 26.9093647, -29.1441708, 26.8447514, -56.1422958, 56.0535355
27: -32.1288452, 16.5072231, -32.0240860, 16.4487877, -47.7150803, 47.6686058
28: -21.5622711, 21.6612015, -21.4572258, 21.6070976, -43.1693687, 43.1184273
29: -23.7323990, 22.2038994, -23.6228962, 22.1669941, -45.8993912, 45.8267975
30: -29.6228371, 16.8217030, -29.5553360, 16.7636719, -45.8581314, 45.8333092
31: -26.3671017, 19.0374069, -26.2658844, 18.9821587, -45.3492584, 45.3032913
32: -42.2323837, 8.4499493, -42.1850166, 8.4203091, -47.5096512, 47.4769058
33: -72.3682556, -5.6764898, -72.2585144, -5.7333746, -61.2082748, 61.1530609
34: -56.5165482, -5.5348110, -56.4324493, -5.5725946, -43.5613594, 43.5118828
35: -50.1670990, -0.0081377, -50.0642052, -0.0410519, -48.1841507, 48.1347809
36: -47.8176651, 4.8793287, -47.6883316, 4.8291492, -51.9498749, 51.8640900
37: -83.6915817, -17.5052643, -83.5440674, -17.5660362, -58.3908691, 58.2263565
38: -58.7104874, 3.1567860, -58.5278206, 3.0811033, -61.2061920, 61.0847321
39: -78.9645615, -11.6541977, -78.8411026, -11.7013302, -65.2060242, 65.1206436
40: -67.6517334, -18.3584061, -67.5934830, -18.3970203, -41.1481400, 41.0647659
41: -55.1860123, -6.8856430, -55.1272621, -6.9422531, -42.2108231, 42.1436005
42: -33.9654236, 6.7927055, -33.9234123, 6.7476578, -37.6467590, 37.6267929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.8627566, upper bound: 44.9746047
time: 29.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0151230, upper bound: 44.9768286
time: 48.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -28.0768738, 17.0800934, -28.0627708, 17.0247269, -44.3173981, 44.3743515
1: -13.6503725, 17.0814247, -13.6468801, 17.0369186, -30.6872902, 30.7283058
2: -14.0599499, 21.6606827, -14.0585794, 21.6104107, -35.4262962, 35.4955788
3: -12.8806000, 23.4273758, -12.8797913, 23.3908234, -36.2714233, 36.3071671
4: -21.5499191, 18.4574490, -21.5450745, 18.4328308, -39.9827499, 40.0025253
5: -11.9799910, 22.8416195, -11.9792347, 22.7914410, -34.7714310, 34.8208542
6: -50.6982269, -3.6152515, -50.6755943, -3.6219730, -40.5351791, 40.4926109
7: -16.3684654, 18.4618874, -16.3694344, 18.4022865, -34.7707520, 34.8313217
8: -18.2789612, 21.3021584, -18.2754650, 21.2745037, -39.5534668, 39.5776215
9: -16.6971130, 23.2813549, -16.6890984, 23.2246628, -38.5878716, 38.6168823
10: -24.2765694, 38.5085831, -24.2678585, 38.4554253, -61.7440033, 61.7779388
11: -24.7734013, 17.5943413, -24.7584133, 17.5873547, -42.3607559, 42.3527527
12: -28.6628819, 20.0953159, -28.6389351, 20.0867138, -46.8974800, 46.8595734
13: -32.9323502, 28.7855396, -32.9151611, 28.7534962, -61.6858444, 61.7006989
14: -23.4683552, 39.1934738, -23.4426346, 39.1590843, -60.0206604, 60.0009270
15: -18.9246407, 25.8447437, -18.9092655, 25.8357239, -44.7603645, 44.7540092
16: -32.7171822, 19.9318237, -32.7035942, 19.8536835, -52.5708656, 52.6354179
17: -17.7679901, 38.4756927, -17.7374344, 38.4292831, -55.2297516, 55.2405205
18: -25.8090096, 19.6143188, -25.7729931, 19.6084766, -45.4174881, 45.3873138
19: -26.4439163, 12.4710007, -26.3965549, 12.4723063, -38.9162216, 38.8675537
20: -21.1300850, 20.4229889, -21.0713482, 20.4200401, -41.5501251, 41.4943390
21: -25.7283344, 18.8588753, -25.6769390, 18.8587570, -44.5870895, 44.5358124
22: -22.1610146, 24.5087261, -22.0796604, 24.5097008, -46.6707153, 46.5883865
23: -21.7135983, 17.4907990, -21.6785736, 17.4872360, -39.2008362, 39.1693726
24: -32.1415482, 11.8884087, -32.1065063, 11.8854980, -44.0270462, 43.9949150
25: -18.1437893, 25.4115887, -18.0839195, 25.4102039, -43.5539932, 43.4955063
26: -29.3038502, 26.9411736, -29.2121944, 26.9396858, -56.2435379, 56.1533661
27: -32.1331100, 16.5338440, -32.0865250, 16.5289345, -47.7965813, 47.7576675
28: -21.5670929, 21.6873112, -21.5103626, 21.6848373, -43.2519302, 43.1976738
29: -23.7365341, 22.2205353, -23.6791821, 22.2185936, -45.9551277, 45.8997192
30: -29.6264973, 16.8452148, -29.6027470, 16.8363857, -45.9342079, 45.9052010
31: -26.3746738, 19.0643253, -26.3261433, 19.0628853, -45.4375610, 45.3904686
32: -42.2362022, 8.4598141, -42.2093315, 8.4556007, -47.5530319, 47.5167351
33: -72.3748093, -5.6521740, -72.3145142, -5.6589794, -61.2884674, 61.2356033
34: -56.5200043, -5.5183115, -56.4529495, -5.5201149, -43.6165276, 43.5492592
35: -50.1719093, 0.0073214, -50.1023331, 0.0054369, -48.2374649, 48.1904831
36: -47.8228912, 4.9047594, -47.7348938, 4.9052362, -52.0292358, 51.9364624
37: -83.6989212, -17.4765320, -83.6247864, -17.4800110, -58.4710388, 58.3379822
38: -58.7166748, 3.1893473, -58.5940170, 3.1831169, -61.3098602, 61.1843872
39: -78.9722824, -11.6323833, -78.9115143, -11.6351233, -65.2731857, 65.2151718
40: -67.6569138, -18.3431091, -67.6325836, -18.3504181, -41.1977005, 41.1231194
41: -55.1893234, -6.8602161, -55.1658707, -6.8637857, -42.2861786, 42.2081757
42: -33.9680405, 6.8100777, -33.9461288, 6.8038254, -37.7048340, 37.6745796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8627795, upper bound: 45.0128883
time: 23.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0151462, upper bound: 45.0151465
time: 44.22 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.9984894, 17.0505409, -28.1255569, 17.0878716, -44.3043060, 44.3945770
1: -13.6038742, 17.0561295, -13.6683817, 17.0795784, -30.6834526, 30.7245102
2: -14.0170031, 21.6414032, -14.0800924, 21.6559868, -35.4309921, 35.4880447
3: -12.8298683, 23.3940620, -12.8894634, 23.4262524, -36.2561188, 36.2835236
4: -21.4968987, 18.4407730, -21.5611725, 18.4513245, -39.9482231, 40.0019455
5: -11.9321976, 22.8093605, -11.9986591, 22.8242989, -34.7564964, 34.8080215
6: -50.6777039, -3.6748438, -50.7808685, -3.5822201, -40.5538254, 40.5499954
7: -16.3292332, 18.4387321, -16.3935356, 18.4331360, -34.7623672, 34.8322678
8: -18.2256603, 21.2891674, -18.3175278, 21.3416061, -39.5672684, 39.6066971
9: -16.6165848, 23.2355099, -16.7024460, 23.2788868, -38.5669174, 38.5825462
10: -24.1899719, 38.4459763, -24.2904224, 38.5300293, -61.7455597, 61.7390976
11: -24.7327099, 17.5249252, -24.7899647, 17.5736332, -42.3063431, 42.3148880
12: -28.6197548, 20.0462532, -28.6441994, 20.1248150, -46.8921280, 46.8142700
13: -32.8584213, 28.7391167, -32.9149628, 28.7953911, -61.6538124, 61.6540794
14: -23.3492241, 39.1469231, -23.5034294, 39.2347412, -59.9871063, 60.0143013
15: -18.8574257, 25.8241158, -18.9384003, 25.8853760, -44.7428017, 44.7625160
16: -32.6489792, 19.8798504, -32.7212906, 19.9095325, -52.5585098, 52.6011429
17: -17.6842957, 38.4347839, -17.7668648, 38.5007401, -55.2220001, 55.2290993
18: -25.7765884, 19.5681915, -25.8391609, 19.6055756, -45.3821640, 45.4073524
19: -26.3846722, 12.3966389, -26.4809799, 12.4785233, -38.8631973, 38.8776169
20: -21.0797195, 20.3603859, -21.1572571, 20.4324818, -41.5121994, 41.5176430
21: -25.6674824, 18.7830772, -25.7789612, 18.8756390, -44.5431213, 44.5620384
22: -22.1067371, 24.4626694, -22.1588688, 24.5217247, -46.6284637, 46.6215363
23: -21.6614838, 17.4213104, -21.7110004, 17.4838467, -39.1453323, 39.1323090
24: -32.0720673, 11.8130388, -32.2021484, 11.8768559, -43.9489212, 44.0151863
25: -18.0913811, 25.3497066, -18.1325302, 25.4042473, -43.4956284, 43.4822388
26: -29.2369614, 26.8548374, -29.2585945, 26.9302483, -56.1672096, 56.1134338
27: -32.0721588, 16.4596176, -32.1466599, 16.5227280, -47.7373962, 47.7550125
28: -21.5149632, 21.6145515, -21.5538864, 21.6812477, -43.1962128, 43.1684380
29: -23.6812229, 22.1744213, -23.7435226, 22.2148838, -45.8961067, 45.9179459
30: -29.5821533, 16.7790604, -29.6860657, 16.8468971, -45.8917732, 45.9210968
31: -26.3164997, 18.9912949, -26.4385395, 19.0776443, -45.3941422, 45.4298325
32: -42.2147102, 8.4299955, -42.2811966, 8.4949045, -47.5639343, 47.5574265
33: -72.3184509, -5.7209606, -72.4590454, -5.6144056, -61.2742462, 61.3128815
34: -56.4945374, -5.5674000, -56.5463257, -5.4847975, -43.6268272, 43.5987625
35: -50.1323318, -0.0383501, -50.2199860, 0.0449514, -48.2361526, 48.2622070
36: -47.7751808, 4.8334408, -47.8309975, 4.9403687, -52.0160370, 51.9663467
37: -83.6217651, -17.5577698, -83.7317047, -17.4660759, -58.4047394, 58.3823204
38: -58.6508331, 3.0954132, -58.7473679, 3.2231970, -61.2831116, 61.2530365
39: -78.9037781, -11.6934195, -79.0695343, -11.5902071, -65.2474442, 65.3218002
40: -67.6162720, -18.3851147, -67.7220001, -18.3311329, -41.1705666, 41.1803627
41: -55.1518974, -6.9343042, -55.2248878, -6.8386993, -42.2743988, 42.2077904
42: -33.9466782, 6.7601328, -33.9575653, 6.8079157, -37.6806335, 37.6381912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8067824, upper bound: 45.0325311
time: 28.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9589963, upper bound: 45.0347933
time: 29.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.0768738, 17.0800934, -28.1496429, 17.0906754, -44.3881378, 44.4563713
1: -13.6503725, 17.0814247, -13.6833506, 17.0812225, -30.7315941, 30.7647743
2: -14.0599499, 21.6606827, -14.0940361, 21.6576672, -35.4783173, 35.5287704
3: -12.8806000, 23.4273758, -12.9062872, 23.4294567, -36.3100586, 36.3336639
4: -21.5499191, 18.4574490, -21.5777664, 18.4540329, -40.0039520, 40.0352173
5: -11.9799910, 22.8416195, -12.0144482, 22.8273697, -34.8073616, 34.8560677
6: -50.6982269, -3.6152515, -50.7828751, -3.5627551, -40.5949478, 40.6087837
7: -16.3684654, 18.4618874, -16.4061146, 18.4360962, -34.8045616, 34.8680038
8: -18.2789612, 21.3021584, -18.3342476, 21.3442345, -39.6231956, 39.6364059
9: -16.6971130, 23.2813549, -16.7285004, 23.2821884, -38.6444626, 38.6550903
10: -24.2765694, 38.5085831, -24.3188305, 38.5363998, -61.8309784, 61.8307686
11: -24.7734013, 17.5943413, -24.7927494, 17.5961037, -42.3695068, 42.3870926
12: -28.6628819, 20.0953159, -28.6575584, 20.1337128, -46.9404449, 46.8765869
13: -32.9323502, 28.7855396, -32.9386787, 28.8014240, -61.7337723, 61.7242203
14: -23.4683552, 39.1934738, -23.5413208, 39.2377625, -60.1010818, 60.1003799
15: -18.9246407, 25.8447437, -18.9599094, 25.8889389, -44.8135796, 44.8046532
16: -32.7171822, 19.9318237, -32.7417450, 19.9141541, -52.6313362, 52.6735687
17: -17.7679901, 38.4756927, -17.7932243, 38.5034103, -55.3039665, 55.2969093
18: -25.8090096, 19.6143188, -25.8432007, 19.6201210, -45.4291306, 45.4575195
19: -26.4439163, 12.4710007, -26.4855556, 12.5034885, -38.9474030, 38.9565582
20: -21.1300850, 20.4229889, -21.1614609, 20.4532623, -41.5833473, 41.5844498
21: -25.7283344, 18.8588753, -25.7842140, 18.9010544, -44.6293869, 44.6430893
22: -22.1610146, 24.5087261, -22.1635208, 24.5366917, -46.6977081, 46.6722488
23: -21.7135983, 17.4907990, -21.7148361, 17.5065994, -39.2201996, 39.2056351
24: -32.1415482, 11.8884087, -32.2060242, 11.9019470, -44.0434952, 44.0944328
25: -18.1437893, 25.4115887, -18.1378517, 25.4244995, -43.5682907, 43.5494385
26: -29.3038502, 26.9411736, -29.2643490, 26.9592800, -56.2631302, 56.2055206
27: -32.1331100, 16.5338440, -32.1504974, 16.5473709, -47.8226929, 47.8279190
28: -21.5670929, 21.6873112, -21.5581951, 21.7055435, -43.2726364, 43.2455063
29: -23.7365341, 22.2205353, -23.7471352, 22.2298737, -45.9664078, 45.9676704
30: -29.6264973, 16.8452148, -29.6892357, 16.8683414, -45.9584732, 45.9896011
31: -26.3746738, 19.0643253, -26.4452896, 19.1018581, -45.4765320, 45.5096130
32: -42.2362022, 8.4598141, -42.2840424, 8.5035028, -47.6004562, 47.5923119
33: -72.3748093, -5.6521740, -72.4647369, -5.5921440, -61.3555603, 61.3859863
34: -56.5200043, -5.5183115, -56.5492249, -5.4691200, -43.6694107, 43.6499863
35: -50.1719093, 0.0073214, -50.2241020, 0.0600309, -48.2929230, 48.3129883
36: -47.8228912, 4.9047594, -47.8355789, 4.9642582, -52.0881500, 52.0391693
37: -83.6989212, -17.4765320, -83.7378082, -17.4390774, -58.5113983, 58.4533920
38: -58.7166748, 3.1893473, -58.7526703, 3.2533302, -61.3806152, 61.3430099
39: -78.9722824, -11.6323833, -79.0758209, -11.5701752, -65.3382874, 65.3810883
40: -67.6569138, -18.3431091, -67.7264481, -18.3173103, -41.2293625, 41.2190895
41: -55.1893234, -6.8602161, -55.2276192, -6.8144798, -42.3369980, 42.2742081
42: -33.9680405, 6.8100777, -33.9597855, 6.8238506, -37.7268524, 37.6894379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 920

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8421129, upper bound: 45.0325354
time: 24.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9944058, upper bound: 45.0347997
time: 25.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -28.1272926, 17.1116028, -27.9644585, 16.9020920, -44.2421799, 44.3013420
1: -13.6664925, 17.0812302, -13.5725508, 16.8886528, -30.5551453, 30.6537819
2: -14.0737600, 21.6529121, -13.9556589, 21.4386692, -35.2703247, 35.3834305
3: -12.8842916, 23.4170628, -12.7967682, 23.2292309, -36.1135216, 36.2138290
4: -21.5543633, 18.4211063, -21.4167786, 18.2632751, -39.8176384, 39.8378830
5: -11.9942055, 22.8357182, -11.9064913, 22.6497478, -34.6439514, 34.7422104
6: -50.7858200, -3.5878725, -50.6083221, -3.6897187, -40.5242729, 40.4468346
7: -16.3852158, 18.4372864, -16.2841911, 18.2261391, -34.6113548, 34.7214775
8: -18.3093929, 21.3154755, -18.1799393, 21.1109161, -39.4203110, 39.4954147
9: -16.6525650, 23.3224125, -16.4646931, 23.0744171, -38.3919678, 38.4535675
10: -24.2318077, 38.5645027, -24.0141563, 38.2502365, -61.4966202, 61.6090317
11: -24.7728004, 17.5706902, -24.6330261, 17.4657154, -42.2385178, 42.2037163
12: -28.5732613, 20.1190987, -28.3244495, 19.9096222, -46.6253891, 46.5918617
13: -32.8883057, 28.8101082, -32.7202530, 28.6265659, -61.5148697, 61.5303612
14: -23.4250717, 39.2618484, -23.0705299, 39.0114174, -59.8266525, 59.7343216
15: -18.9357262, 25.8680515, -18.7891006, 25.7572098, -44.6929359, 44.6571503
16: -32.7067795, 19.9712791, -32.5688553, 19.7193108, -52.4260902, 52.5401344
17: -17.7311497, 38.5378532, -17.4948959, 38.3263512, -55.0894356, 55.0893326
18: -25.8510971, 19.5921421, -25.6237888, 19.5053444, -45.3564415, 45.2159309
19: -26.5053482, 12.4738798, -26.2699394, 12.4058561, -38.9112053, 38.7438202
20: -21.1785011, 20.4308205, -20.9096317, 20.3346615, -41.5131607, 41.3404541
21: -25.7887402, 18.8700924, -25.4997425, 18.7686615, -44.5574036, 44.3698349
22: -22.2060356, 24.5135536, -21.9247379, 24.4476414, -46.6536789, 46.4382935
23: -21.7270966, 17.4825554, -21.5695744, 17.3989735, -39.1260681, 39.0521317
24: -32.2243271, 11.8526058, -31.9584560, 11.7430592, -43.9673843, 43.8110619
25: -18.1695404, 25.3991432, -17.9702339, 25.3257809, -43.4953232, 43.3693771
26: -29.2960892, 26.9247112, -28.9799480, 26.7841053, -56.0801926, 55.9046593
27: -32.1750832, 16.5038185, -31.9272232, 16.4019775, -47.7835464, 47.5742722
28: -21.5866013, 21.6793842, -21.3855972, 21.6005993, -43.1872025, 43.0649796
29: -23.7583656, 22.2109871, -23.5032482, 22.1197834, -45.8781509, 45.7142334
30: -29.6848869, 16.8460083, -29.4910507, 16.7318649, -45.8745346, 45.7866669
31: -26.4627666, 19.0674171, -26.1885090, 18.9839363, -45.4467010, 45.2559280
32: -42.2697830, 8.4895735, -42.0799255, 8.3870621, -47.4836159, 47.4101562
33: -72.5110779, -5.6538105, -72.1876297, -5.8085623, -61.2720032, 61.1007538
34: -56.6021347, -5.4991169, -56.3691483, -5.5757179, -43.6396484, 43.4637718
35: -50.2778778, 0.0290222, -50.0105591, -0.0401096, -48.2922134, 48.0651512
36: -47.8815536, 4.9336882, -47.5814590, 4.8614244, -52.0335312, 51.8030319
37: -83.7861862, -17.4860344, -83.4796371, -17.5921288, -58.3873215, 58.1770401
38: -58.8463020, 3.2190924, -58.4452972, 3.1228695, -61.3704453, 61.0602341
39: -79.1159363, -11.6021194, -78.7825241, -11.6786709, -65.3839340, 65.1175232
40: -67.7334747, -18.3584747, -67.5419388, -18.4661770, -41.1448860, 41.0674477
41: -55.2353477, -6.8503971, -55.0799866, -6.9508018, -42.1810913, 42.1659431
42: -33.9597015, 6.8025351, -33.8783875, 6.6882620, -37.5659828, 37.5849113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236081, upper bound: 44.6926997
time: 36.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236081, upper bound: 44.7281654
time: 42.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -28.1326675, 17.1240845, -27.9892883, 16.9402637, -44.2746849, 44.3407440
1: -13.6687689, 17.0967369, -13.5933790, 16.9338341, -30.6026039, 30.6901169
2: -14.0768585, 21.6722717, -13.9896059, 21.4949608, -35.3202362, 35.4375687
3: -12.8869314, 23.4348679, -12.8119411, 23.2787285, -36.1656609, 36.2468109
4: -21.5587502, 18.4383297, -21.4538956, 18.3143940, -39.8731461, 39.8922272
5: -11.9961891, 22.8496456, -11.9204550, 22.6901531, -34.6863403, 34.7700996
6: -50.7921410, -3.5824914, -50.6229553, -3.6634202, -40.5817490, 40.4664268
7: -16.3886147, 18.4612904, -16.3130856, 18.2889137, -34.6775284, 34.7743759
8: -18.3141632, 21.3361530, -18.2182617, 21.1709919, -39.4851532, 39.5544128
9: -16.6766167, 23.3272781, -16.5367413, 23.1191101, -38.4587479, 38.5145302
10: -24.2511635, 38.5719452, -24.0730820, 38.3072281, -61.5722198, 61.6681366
11: -24.7816772, 17.5747490, -24.6532497, 17.4874420, -42.2691193, 42.2279968
12: -28.6131611, 20.1249847, -28.4401817, 19.9778309, -46.7339554, 46.7000771
13: -32.9131622, 28.8177433, -32.7953758, 28.6805363, -61.5936966, 61.6131210
14: -23.4745979, 39.2645531, -23.2198620, 39.0578766, -59.9236526, 59.8577194
15: -18.9427872, 25.8780155, -18.8206120, 25.7713127, -44.7140999, 44.6986275
16: -32.7162704, 19.9782772, -32.6000214, 19.7531586, -52.4694290, 52.5783005
17: -17.7614803, 38.5413475, -17.5853539, 38.3514557, -55.1450157, 55.1658478
18: -25.8607712, 19.5964603, -25.6736565, 19.5196686, -45.3804398, 45.2701187
19: -26.5130138, 12.4750881, -26.2955055, 12.4099541, -38.9229660, 38.7705917
20: -21.1945057, 20.4327030, -20.9599495, 20.3551064, -41.5496140, 41.3926544
21: -25.8045464, 18.8723392, -25.5474739, 18.7872295, -44.5917740, 44.4198151
22: -22.2208099, 24.5164032, -21.9700775, 24.4660568, -46.6868668, 46.4864807
23: -21.7341480, 17.4850044, -21.5968304, 17.4074326, -39.1415787, 39.0818329
24: -32.2294998, 11.8646374, -32.0000916, 11.7794094, -44.0089111, 43.8647308
25: -18.1764793, 25.4019585, -17.9931488, 25.3367176, -43.5131989, 43.3951073
26: -29.3212147, 26.9271927, -29.0592709, 26.8144379, -56.1356506, 55.9864655
27: -32.1828003, 16.5124664, -31.9743481, 16.4271469, -47.7923813, 47.6310425
28: -21.5951805, 21.6808910, -21.4157352, 21.6070232, -43.2022018, 43.0966263
29: -23.7756195, 22.2131138, -23.5516529, 22.1442566, -45.9198761, 45.7647667
30: -29.6897488, 16.8498669, -29.5058727, 16.7525997, -45.9049377, 45.8056412
31: -26.4713879, 19.0750294, -26.2285004, 19.0077591, -45.4791489, 45.3035278
32: -42.2832222, 8.4939566, -42.1193237, 8.4215136, -47.5598145, 47.4535828
33: -72.5150223, -5.6298466, -72.2281952, -5.7349968, -61.3381805, 61.1666489
34: -56.6077957, -5.4909229, -56.4029236, -5.5505934, -43.6717834, 43.5118790
35: -50.2836723, 0.0372305, -50.0408325, -0.0147276, -48.3234100, 48.1317520
36: -47.9017715, 4.9363432, -47.6396866, 4.8813381, -52.0845337, 51.8683090
37: -83.7946014, -17.4731998, -83.5158844, -17.5536575, -58.4316101, 58.2208824
38: -58.8564110, 3.2218943, -58.4804955, 3.1353483, -61.4195175, 61.1071548
39: -79.1213074, -11.5940666, -78.8125916, -11.6539974, -65.4043884, 65.1543884
40: -67.7387238, -18.3371506, -67.5733948, -18.4048424, -41.2033272, 41.0825348
41: -55.2407913, -6.8426714, -55.1051025, -6.9254169, -42.2559967, 42.1749420
42: -33.9626846, 6.8071451, -33.8780899, 6.7188110, -37.6295242, 37.5921593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258381, upper bound: 44.8225909
time: 24.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258381, upper bound: 44.8581746
time: 41.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.1513672, 17.1144161, -28.0428848, 16.9316425, -44.3039742, 44.3852310
1: -13.6814795, 17.0828705, -13.6190681, 16.9139595, -30.5954399, 30.7019386
2: -14.0877199, 21.6546021, -13.9986134, 21.4579620, -35.3110809, 35.4307480
3: -12.9010906, 23.4202690, -12.8474846, 23.2625504, -36.1636429, 36.2677536
4: -21.5709763, 18.4238148, -21.4698143, 18.2799816, -39.8509598, 39.8936310
5: -12.0099754, 22.8387833, -11.9542637, 22.6819878, -34.6919632, 34.7930450
6: -50.7878304, -3.5684309, -50.6288338, -3.6301184, -40.5830879, 40.4879684
7: -16.3978081, 18.4402447, -16.3233929, 18.2492867, -34.6470947, 34.7636375
8: -18.3260918, 21.3181171, -18.2332191, 21.1239243, -39.4500160, 39.5513382
9: -16.6786327, 23.3257065, -16.5452881, 23.1202698, -38.4645233, 38.5311775
10: -24.2602272, 38.5708923, -24.1007767, 38.3128510, -61.5882721, 61.6944809
11: -24.7755966, 17.5931244, -24.6737251, 17.5351410, -42.3107376, 42.2668495
12: -28.5865917, 20.1280384, -28.3675709, 19.9586716, -46.6876678, 46.6401939
13: -32.9120178, 28.8161469, -32.7941742, 28.6730766, -61.5850945, 61.6103210
14: -23.4629726, 39.2648544, -23.1896935, 39.0579910, -59.9127045, 59.8483467
15: -18.9572487, 25.8715897, -18.8563652, 25.7778263, -44.7350769, 44.7279549
16: -32.7272034, 19.9758644, -32.6370544, 19.7712975, -52.4985008, 52.6129189
17: -17.7574921, 38.5405197, -17.5786247, 38.3672791, -55.1572762, 55.1713562
18: -25.8551311, 19.6066990, -25.6562653, 19.5514927, -45.4066238, 45.2629623
19: -26.5099258, 12.4988413, -26.3291855, 12.4802361, -38.9901619, 38.8280258
20: -21.1826973, 20.4516258, -20.9599876, 20.3972511, -41.5799484, 41.4116135
21: -25.7939930, 18.8955154, -25.5605736, 18.8444500, -44.6384430, 44.4560890
22: -22.2106724, 24.5285072, -21.9789581, 24.4936714, -46.7043457, 46.5074654
23: -21.7309494, 17.5053253, -21.6216946, 17.4684391, -39.1993866, 39.1270218
24: -32.2282181, 11.8777342, -32.0279541, 11.8184509, -44.0466690, 43.9056892
25: -18.1748447, 25.4194260, -18.0226135, 25.3876534, -43.5625000, 43.4420395
26: -29.3018303, 26.9537163, -29.0468540, 26.8704071, -56.1722374, 56.0005722
27: -32.1789093, 16.5284653, -31.9881382, 16.4762077, -47.8565178, 47.6595612
28: -21.5909100, 21.7036724, -21.4377003, 21.6733818, -43.2642899, 43.1413727
29: -23.7619648, 22.2260017, -23.5585976, 22.1659164, -45.9278793, 45.7845993
30: -29.6880322, 16.8674545, -29.5354042, 16.7980175, -45.9430122, 45.8533897
31: -26.4694862, 19.0916328, -26.2467117, 19.0570049, -45.5264893, 45.3383446
32: -42.2726288, 8.4981794, -42.1013985, 8.4168997, -47.5185280, 47.4466438
33: -72.5167160, -5.6315041, -72.2439499, -5.7397566, -61.3452225, 61.1820145
34: -56.6050262, -5.4834318, -56.3946877, -5.5266247, -43.6908951, 43.5063820
35: -50.2819672, 0.0441017, -50.0501060, 0.0055265, -48.3430481, 48.1219749
36: -47.8861465, 4.9575558, -47.6291351, 4.9327545, -52.1064148, 51.8750229
37: -83.7923203, -17.4590416, -83.5567627, -17.5108795, -58.4584122, 58.2836914
38: -58.8516388, 3.2492599, -58.5111542, 3.2168512, -61.4604492, 61.1577835
39: -79.1222534, -11.5821199, -78.8508911, -11.6175985, -65.4432831, 65.2083130
40: -67.7379074, -18.3446617, -67.5826111, -18.4241447, -41.1836319, 41.1262016
41: -55.2380829, -6.8261833, -55.1173706, -6.8767166, -42.2475166, 42.2285538
42: -33.9619293, 6.8185139, -33.8997841, 6.7381916, -37.6172066, 37.6311264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236103, upper bound: 44.7279761
time: 72.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236103, upper bound: 44.7633671
time: 30.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.1567688, 17.1268921, -28.0677032, 16.9698105, -44.3365021, 44.4246712
1: -13.6837559, 17.0983734, -13.6398735, 16.9591370, -30.6428928, 30.7382469
2: -14.0908089, 21.6739349, -14.0325756, 21.5142670, -35.3609810, 35.4848862
3: -12.9037485, 23.4380741, -12.8626804, 23.3120575, -36.2158051, 36.3007545
4: -21.5753536, 18.4410419, -21.5069218, 18.3311005, -39.9064560, 39.9479637
5: -12.0119467, 22.8527069, -11.9682579, 22.7224007, -34.7343483, 34.8209648
6: -50.7941513, -3.5630608, -50.6434746, -3.6038165, -40.6405716, 40.5075684
7: -16.4011803, 18.4642487, -16.3522835, 18.3120594, -34.7132416, 34.8165321
8: -18.3308868, 21.3387871, -18.2715626, 21.1839943, -39.5148811, 39.6103516
9: -16.7026768, 23.3305664, -16.6173058, 23.1649551, -38.5312843, 38.5921326
10: -24.2795715, 38.5783615, -24.1596832, 38.3698578, -61.6639023, 61.7535629
11: -24.7844734, 17.5971947, -24.6939373, 17.5568867, -42.3413620, 42.2911301
12: -28.6265221, 20.1338997, -28.4833145, 20.0268917, -46.7962341, 46.7484245
13: -32.9368591, 28.8237419, -32.8692818, 28.7270470, -61.6639061, 61.6930237
14: -23.5124969, 39.2675819, -23.3390160, 39.1044350, -60.0097122, 59.9717026
15: -18.9643021, 25.8815689, -18.8879547, 25.7919235, -44.7562256, 44.7695236
16: -32.7367020, 19.9828529, -32.6682434, 19.8051319, -52.5418320, 52.6510963
17: -17.7878380, 38.5440216, -17.6690903, 38.3923531, -55.2128448, 55.2478905
18: -25.8647976, 19.6110191, -25.7061176, 19.5658092, -45.4306068, 45.3171387
19: -26.5175686, 12.5000458, -26.3547573, 12.4843245, -39.0018921, 38.8548050
20: -21.1987171, 20.4535141, -21.0102978, 20.4176846, -41.6164017, 41.4638138
21: -25.8097992, 18.8977547, -25.6083107, 18.8630180, -44.6728172, 44.5060654
22: -22.2254601, 24.5313454, -22.0243149, 24.5120850, -46.7375450, 46.5556602
23: -21.7379627, 17.5077438, -21.6489601, 17.4768982, -39.2148590, 39.1567039
24: -32.2333946, 11.8897381, -32.0695877, 11.8547792, -44.0881729, 43.9593277
25: -18.1817780, 25.4222298, -18.0455151, 25.3985939, -43.5803719, 43.4677429
26: -29.3269711, 26.9562016, -29.1261616, 26.9007759, -56.2277451, 56.0823631
27: -32.1866417, 16.5371284, -32.0352783, 16.5014095, -47.8653488, 47.7163544
28: -21.5994644, 21.7052002, -21.4678497, 21.6798077, -43.2792740, 43.1730499
29: -23.7792377, 22.2281265, -23.6069756, 22.1903839, -45.9696198, 45.8351021
30: -29.6928997, 16.8713226, -29.5502300, 16.8187580, -45.9734344, 45.8723755
31: -26.4780979, 19.0992317, -26.2867184, 19.0808296, -45.5589294, 45.3859482
32: -42.2860641, 8.5025673, -42.1408081, 8.4513693, -47.5947189, 47.4900589
33: -72.5206909, -5.6075478, -72.2844925, -5.6662130, -61.4113617, 61.2479401
34: -56.6106834, -5.4752541, -56.4283867, -5.5014963, -43.7230453, 43.5544662
35: -50.2877960, 0.0523338, -50.0803452, 0.0309381, -48.3742065, 48.1885338
36: -47.9063263, 4.9602480, -47.6874161, 4.9526587, -52.1573792, 51.9403458
37: -83.8006897, -17.4462185, -83.5930176, -17.4723854, -58.5027466, 58.3274994
38: -58.8617897, 3.2520638, -58.5463715, 3.2292767, -61.5095291, 61.2047348
39: -79.1276093, -11.5740395, -78.8810120, -11.5929317, -65.4637299, 65.2451324
40: -67.7431793, -18.3233242, -67.6140137, -18.3628311, -41.2420731, 41.1412773
41: -55.2434845, -6.8184414, -55.1424980, -6.8513117, -42.3224030, 42.2375679
42: -33.9649200, 6.8231144, -33.8994751, 6.7687731, -37.6807632, 37.6383667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8579542
time: 52.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8934650
time: 55.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -27.8916740, 17.0468636, -28.0369911, 17.0206108, -44.1036339, 44.3063431
1: -13.5293903, 17.0493164, -13.6298647, 17.0348434, -30.5642338, 30.6791801
2: -13.9464560, 21.6330204, -14.0433798, 21.6086864, -35.2906876, 35.4471588
3: -12.7719879, 23.3800449, -12.8632097, 23.3856583, -36.1576462, 36.2432556
4: -21.4049988, 18.4264832, -21.5188293, 18.4346733, -39.8396721, 39.9453125
5: -11.8669415, 22.7839031, -11.9663086, 22.7876530, -34.6545944, 34.7502136
6: -50.7421036, -3.7145624, -50.6702499, -3.6182628, -40.5989571, 40.3875275
7: -16.2505150, 18.4049187, -16.3543930, 18.3955154, -34.6460304, 34.7593117
8: -18.1444016, 21.3205528, -18.2734890, 21.2722263, -39.4166260, 39.5940399
9: -16.5409966, 23.2207127, -16.6624374, 23.2205582, -38.4301300, 38.5037384
10: -24.1328430, 38.4479141, -24.2450371, 38.4497070, -61.5986176, 61.6704750
11: -24.7341690, 17.4619923, -24.7549019, 17.5557880, -42.2899551, 42.2168961
12: -28.5963840, 19.9991608, -28.6217270, 20.0860252, -46.8234711, 46.7342033
13: -32.7725563, 28.7225399, -32.8716049, 28.7543659, -61.5269241, 61.5941467
14: -23.2819290, 39.1781845, -23.4240761, 39.1562653, -59.8364792, 59.9275322
15: -18.8242950, 25.8145237, -18.8989105, 25.8210983, -44.6453934, 44.7134323
16: -32.5539703, 19.8456306, -32.6703262, 19.8496284, -52.4035988, 52.5159569
17: -17.6251488, 38.4498215, -17.7214432, 38.4267426, -55.0872993, 55.1754341
18: -25.7868309, 19.5173569, -25.7673378, 19.5842056, -45.3710365, 45.2846947
19: -26.4068203, 12.3425426, -26.3931770, 12.4502163, -38.8570366, 38.7357178
20: -21.0886593, 20.2994041, -21.0679054, 20.3994102, -41.4880676, 41.3673096
21: -25.6974430, 18.7204514, -25.6723022, 18.8404999, -44.5379410, 44.3927536
22: -22.0929070, 24.3874989, -22.0667801, 24.4891434, -46.5820503, 46.4542770
23: -21.6457615, 17.3554211, -21.6771374, 17.4481468, -39.0939102, 39.0325584
24: -32.1133881, 11.7426949, -32.1020966, 11.8506374, -43.9640274, 43.8447914
25: -18.0620899, 25.2570953, -18.0791035, 25.3729553, -43.4350433, 43.3361969
26: -29.1711941, 26.7334404, -29.1971359, 26.8878231, -56.0590172, 55.9305763
27: -32.0667343, 16.3753662, -32.0831909, 16.4908714, -47.6852379, 47.6004066
28: -21.4870758, 21.5211678, -21.5072937, 21.6445999, -43.1316757, 43.0284615
29: -23.6775284, 22.0931892, -23.6711082, 22.1877651, -45.8652954, 45.7642975
30: -29.6240120, 16.7241020, -29.6005459, 16.8130398, -45.8926353, 45.7708511
31: -26.3559647, 18.9234047, -26.3219032, 19.0405922, -45.3965569, 45.2453079
32: -42.2396278, 8.3917542, -42.2035484, 8.4596939, -47.5690994, 47.4373322
33: -72.3871765, -5.7640724, -72.3097839, -5.6479139, -61.2920380, 61.1159363
34: -56.5126114, -5.6323280, -56.4492378, -5.5238762, -43.6005478, 43.4354095
35: -50.1665916, -0.0850277, -50.0978508, 0.0163946, -48.2389450, 48.1088486
36: -47.7707825, 4.7649727, -47.7252998, 4.9011259, -51.9783936, 51.7887192
37: -83.6332474, -17.5908508, -83.6169586, -17.4919872, -58.3759613, 58.2231293
38: -58.6556778, 3.0106716, -58.5890617, 3.1765823, -61.2404785, 61.0014420
39: -78.9785843, -11.6976242, -78.9049225, -11.6109867, -65.2912369, 65.1472626
40: -67.6580200, -18.3926506, -67.6247025, -18.3506927, -41.2149544, 41.0686455
41: -55.1691742, -6.9783859, -55.1591797, -6.8768091, -42.2895012, 42.0877914
42: -33.9270630, 6.6921835, -33.9440269, 6.7765074, -37.6473732, 37.5580406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9487730, upper bound: 44.9889166
time: 32.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9487730, upper bound: 45.0244471
time: 27.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -27.9824944, 17.0769157, -28.0651283, 17.0237637, -44.2016258, 44.3713341
1: -13.5823994, 17.0756092, -13.6470375, 17.0367012, -30.6191006, 30.7226467
2: -13.9955788, 21.6521721, -14.0590630, 21.6105957, -35.3456841, 35.4884109
3: -12.8282471, 23.4156570, -12.8819199, 23.3895416, -36.2177887, 36.2975769
4: -21.4638367, 18.4451981, -21.5370808, 18.4379158, -39.9017525, 39.9822769
5: -11.9183178, 22.8156147, -11.9833250, 22.7912788, -34.7095947, 34.7989388
6: -50.7659988, -3.6464562, -50.6728172, -3.5968676, -40.6466370, 40.4591293
7: -16.2922497, 18.4280701, -16.3678017, 18.3988285, -34.6910782, 34.7958717
8: -18.2021675, 21.3338776, -18.2916870, 21.2751713, -39.4773407, 39.6255646
9: -16.6288223, 23.2687225, -16.6910019, 23.2244225, -38.5169754, 38.5807343
10: -24.2255993, 38.5111923, -24.2757759, 38.4569473, -61.6951141, 61.7654114
11: -24.7773113, 17.5398750, -24.7583733, 17.5807629, -42.3580742, 42.2982483
12: -28.6450844, 20.0525513, -28.6369972, 20.0959053, -46.8793411, 46.8024368
13: -32.8520279, 28.7708473, -32.8972054, 28.7609234, -61.6129532, 61.6680527
14: -23.4127579, 39.2261429, -23.4657745, 39.1596222, -59.9645195, 60.0181961
15: -18.8979931, 25.8372002, -18.9223900, 25.8254013, -44.7233963, 44.7595901
16: -32.6317520, 19.8956490, -32.6938248, 19.8548431, -52.4865952, 52.5894737
17: -17.7180519, 38.4928207, -17.7510452, 38.4299431, -55.1813202, 55.2489853
18: -25.8203564, 19.5673866, -25.7717476, 19.6001511, -45.4205093, 45.3391342
19: -26.4688473, 12.4250879, -26.3983765, 12.4780235, -38.9468689, 38.8234634
20: -21.1419487, 20.3705082, -21.0726357, 20.4229717, -41.5649185, 41.4431458
21: -25.7602291, 18.8048744, -25.6780796, 18.8687668, -44.6289978, 44.4829559
22: -22.1477203, 24.4381771, -22.0718346, 24.5055695, -46.6532898, 46.5100098
23: -21.6995277, 17.4299774, -21.6816425, 17.4728012, -39.1723289, 39.1116180
24: -32.1855240, 11.8236523, -32.1067200, 11.8776760, -44.0632019, 43.9303741
25: -18.1159916, 25.3242207, -18.0850716, 25.3950958, -43.5110855, 43.4092941
26: -29.2392502, 26.8284035, -29.2034550, 26.9195900, -56.1588402, 56.0318604
27: -32.1291351, 16.4555092, -32.0874557, 16.5174866, -47.7743378, 47.6818428
28: -21.5402489, 21.5988846, -21.5121403, 21.6707077, -43.2109566, 43.1110229
29: -23.7338314, 22.1448040, -23.6752548, 22.2043915, -45.9382248, 45.8200607
30: -29.6714363, 16.7968216, -29.6042213, 16.8365612, -45.9645157, 45.8469467
31: -26.4161987, 19.0041695, -26.3295059, 19.0675392, -45.4837379, 45.3336754
32: -42.2639275, 8.4270296, -42.2073555, 8.4695873, -47.6089020, 47.4806633
33: -72.4432373, -5.6897030, -72.3163147, -5.6236486, -61.3746109, 61.1960983
34: -56.5330811, -5.5798779, -56.4527092, -5.5074034, -43.6380005, 43.4905701
35: -50.2047729, -0.0385418, -50.1026382, 0.0318222, -48.2946548, 48.1622086
36: -47.8173752, 4.8410883, -47.7305222, 4.9265194, -52.0507660, 51.8679886
37: -83.7139587, -17.5048847, -83.6243057, -17.4632645, -58.4876251, 58.3032265
38: -58.7218895, 3.1126785, -58.5952225, 3.2091389, -61.3401642, 61.1051636
39: -79.0489883, -11.6314592, -78.9125824, -11.5892200, -65.3858185, 65.2144623
40: -67.6970673, -18.3460732, -67.6298676, -18.3353844, -41.2732735, 41.1182823
41: -55.2077866, -6.8999310, -55.1625366, -6.8513927, -42.3540840, 42.1631317
42: -33.9498329, 6.7483740, -33.9466667, 6.7938967, -37.6951370, 37.6160812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9873345, upper bound: 44.9889564
time: 27.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9873345, upper bound: 45.0244905
time: 54.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -28.0717964, 17.1161385, -28.0881310, 17.0249443, -44.2917099, 44.4283142
1: -13.6334066, 17.0993290, -13.6607590, 17.0368862, -30.6702919, 30.7600880
2: -14.0458784, 21.6886787, -14.0734444, 21.6106453, -35.3932953, 35.5346146
3: -12.8505058, 23.4299488, -12.8869848, 23.3900414, -36.2405472, 36.3169327
4: -21.5234051, 18.4597511, -21.5537720, 18.4375725, -39.9609756, 40.0135231
5: -11.9635181, 22.8458767, -11.9956169, 22.7914028, -34.7549210, 34.8414917
6: -50.7813339, -3.6245985, -50.6756668, -3.5922842, -40.6677475, 40.4793587
7: -16.3631687, 18.4721451, -16.3891029, 18.3987923, -34.7619629, 34.8612480
8: -18.2795982, 21.3581924, -18.3139324, 21.2757797, -39.5553780, 39.6721268
9: -16.6482162, 23.2907944, -16.6931229, 23.2247696, -38.5430603, 38.6086540
10: -24.2341843, 38.5263214, -24.2738533, 38.4575424, -61.7110748, 61.7925911
11: -24.7643795, 17.5247707, -24.7594814, 17.5730534, -42.3374329, 42.2842522
12: -28.6323891, 20.0886421, -28.6260090, 20.1109543, -46.8885803, 46.8249130
13: -32.8760071, 28.7849197, -32.9001007, 28.7636681, -61.6396751, 61.6850204
14: -23.4352760, 39.2241516, -23.4640732, 39.1598930, -59.9959221, 60.0207787
15: -18.9006081, 25.8749695, -18.9180641, 25.8372478, -44.7378540, 44.7930336
16: -32.6769333, 19.9425716, -32.7047958, 19.8556538, -52.5325851, 52.6473694
17: -17.7302246, 38.5066452, -17.7446785, 38.4299812, -55.1951675, 55.2666206
18: -25.8453445, 19.5757103, -25.7744408, 19.6004353, -45.4457779, 45.3501511
19: -26.4707947, 12.4193001, -26.3985329, 12.4742355, -38.9450302, 38.8178329
20: -21.1668987, 20.3846569, -21.0739174, 20.4246635, -41.5915604, 41.4585724
21: -25.7726669, 18.8163872, -25.6796417, 18.8700256, -44.6426926, 44.4960289
22: -22.1893482, 24.4848557, -22.0735798, 24.5191841, -46.7085342, 46.5584335
23: -21.6959095, 17.4353218, -21.6820717, 17.4717484, -39.1676559, 39.1173935
24: -32.1685791, 11.8236704, -32.1082458, 11.8746853, -44.0432663, 43.9319153
25: -18.1436081, 25.3585625, -18.0864143, 25.4036446, -43.5472527, 43.4449768
26: -29.2869873, 26.8652458, -29.2061272, 26.9276657, -56.2146530, 56.0713730
27: -32.1345520, 16.4717922, -32.0897217, 16.5195770, -47.7991867, 47.7064781
28: -21.5617409, 21.6299515, -21.5128517, 21.6772270, -43.2389679, 43.1428032
29: -23.7474899, 22.1800060, -23.6757927, 22.2137642, -45.9612541, 45.8557968
30: -29.6654491, 16.8039703, -29.6057663, 16.8350201, -45.9631348, 45.8585968
31: -26.4331665, 19.0219688, -26.3309135, 19.0705166, -45.5036850, 45.3528824
32: -42.2859192, 8.4721975, -42.2098579, 8.4830046, -47.6397324, 47.5238571
33: -72.4693069, -5.6600370, -72.3164139, -5.6179142, -61.4094086, 61.2315750
34: -56.5961685, -5.5202808, -56.4541435, -5.4900188, -43.7216644, 43.5486870
35: -50.2556076, 0.0152874, -50.1037598, 0.0468092, -48.3612823, 48.2155991
36: -47.8768578, 4.8874598, -47.7320824, 4.9389057, -52.1235962, 51.9194412
37: -83.7309418, -17.5218906, -83.6244812, -17.4717560, -58.5079803, 58.3031616
38: -58.8092308, 3.1572218, -58.5986977, 3.2199440, -61.4475250, 61.1605301
39: -79.0663681, -11.6339235, -78.9131317, -11.5922842, -65.4011612, 65.2230377
40: -67.7115250, -18.3571625, -67.6338272, -18.3415451, -41.2787094, 41.1128349
41: -55.2122383, -6.8898668, -55.1651993, -6.8505974, -42.3645096, 42.1840210
42: -33.9589539, 6.7735758, -33.9471588, 6.7999754, -37.7075386, 37.6406746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9872670, upper bound: 44.9901945
time: 41.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9872670, upper bound: 45.0257990
time: 29.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -28.1574364, 17.1336594, -28.0888138, 16.9889297, -44.3568344, 44.4528122
1: -13.6841650, 17.1100960, -13.6569843, 16.9931602, -30.6773262, 30.7670803
2: -14.0919580, 21.6884460, -14.0550804, 21.5555458, -35.3976097, 35.5215683
3: -12.9041214, 23.4477158, -12.8880377, 23.3421001, -36.2462234, 36.3357544
4: -21.5779686, 18.4612236, -21.5348339, 18.3893089, -39.9672775, 39.9960556
5: -12.0129232, 22.8632469, -11.9963856, 22.7525215, -34.7654457, 34.8596344
6: -50.7989120, -3.5617700, -50.6628532, -3.5971422, -40.6576767, 40.5306969
7: -16.4015350, 18.4712429, -16.3720512, 18.3358650, -34.7374001, 34.8432922
8: -18.3326588, 21.3508110, -18.2937431, 21.2181549, -39.5508118, 39.6445541
9: -16.7120171, 23.3338966, -16.6490974, 23.1837177, -38.5628052, 38.6239853
10: -24.3076057, 38.5820923, -24.2453899, 38.4068604, -61.7312088, 61.8278999
11: -24.7986565, 17.5986595, -24.7422295, 17.5763760, -42.3750305, 42.3408890
12: -28.6411896, 20.1361027, -28.5251045, 20.0524597, -46.8357391, 46.7845917
13: -32.9306259, 28.8255310, -32.8502121, 28.7158279, -61.6464539, 61.6757431
14: -23.5165443, 39.2693558, -23.3556061, 39.1164284, -60.0266075, 59.9871674
15: -18.9673138, 25.8877563, -18.9102001, 25.8267860, -44.7940979, 44.7979584
16: -32.7452545, 19.9855537, -32.6963806, 19.8271561, -52.5724106, 52.6819344
17: -17.7928352, 38.5461617, -17.6831074, 38.4080238, -55.2334595, 55.2625351
18: -25.8691635, 19.6214790, -25.7287197, 19.6017685, -45.4709320, 45.3501968
19: -26.5251369, 12.5006676, -26.3776474, 12.4979439, -39.0230789, 38.8783150
20: -21.2040997, 20.4538994, -21.0279675, 20.4275513, -41.6316528, 41.4818649
21: -25.8195801, 18.8985729, -25.6370068, 18.8796215, -44.6992035, 44.5355797
22: -22.2293301, 24.5326805, -22.0321732, 24.5171127, -46.7464447, 46.5648537
23: -21.7426529, 17.5074654, -21.6590500, 17.4879456, -39.2305984, 39.1665154
24: -32.2354889, 11.8925972, -32.0711060, 11.8650684, -44.1005554, 43.9637032
25: -18.1905155, 25.4228649, -18.0692482, 25.4141693, -43.6046829, 43.4921112
26: -29.3298683, 26.9577312, -29.1322708, 26.9233990, -56.2532654, 56.0900040
27: -32.1892204, 16.5432777, -32.0460014, 16.5208893, -47.8792191, 47.7303734
28: -21.6062546, 21.7061729, -21.4868736, 21.6968555, -43.3031082, 43.1930466
29: -23.7864628, 22.2295074, -23.6290035, 22.2042351, -45.9906998, 45.8585129
30: -29.7079811, 16.8728638, -29.5944004, 16.8376713, -46.0045013, 45.9153442
31: -26.4847527, 19.0951252, -26.2982540, 19.0735168, -45.5582695, 45.3933792
32: -42.2967529, 8.5031576, -42.1733971, 8.4583445, -47.6032677, 47.5230103
33: -72.5213928, -5.6096592, -72.2821960, -5.6679029, -61.4249268, 61.2455902
34: -56.6109543, -5.4759693, -56.4233398, -5.4986486, -43.7265167, 43.5551033
35: -50.2879295, 0.0535564, -50.0779419, 0.0365314, -48.3854294, 48.2019196
36: -47.9031830, 4.9608746, -47.6778717, 4.9437132, -52.1433563, 51.9312897
37: -83.8033218, -17.4486923, -83.5952759, -17.4817944, -58.5737000, 58.3389587
38: -58.8652420, 3.2564030, -58.5692863, 3.2404613, -61.4971466, 61.2167435
39: -79.1314850, -11.5758553, -78.8906097, -11.5955200, -65.4748688, 65.2531815
40: -67.7453995, -18.3318710, -67.6072845, -18.3882446, -41.2786560, 41.1467361
41: -55.2454300, -6.8190908, -55.1417923, -6.8509045, -42.3512268, 42.2484055
42: -33.9786911, 6.8251619, -33.9497223, 6.7867718, -37.6916809, 37.6910324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8380079
time: 56.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8735212
time: 31.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -28.1628227, 17.1461449, -28.1162796, 17.0280952, -44.3897858, 44.4933014
1: -13.6864424, 17.1256294, -13.6779375, 17.0387421, -30.7251854, 30.8035660
2: -14.0950451, 21.7078228, -14.0891161, 21.6125679, -35.4483109, 35.5758553
3: -12.9067516, 23.4655380, -12.9057102, 23.3939133, -36.3006668, 36.3712463
4: -21.5823479, 18.4784775, -21.5720444, 18.4408169, -40.0231628, 40.0505219
5: -12.0149059, 22.8775921, -12.0126448, 22.7950096, -34.8099136, 34.8902359
6: -50.8052216, -3.5564308, -50.6782341, -3.5708709, -40.7154312, 40.5510521
7: -16.4049149, 18.4952621, -16.4024963, 18.4021034, -34.8070183, 34.8977585
8: -18.3374443, 21.3715210, -18.3321438, 21.2787132, -39.6161575, 39.7036667
9: -16.7360725, 23.3387527, -16.7216873, 23.2286339, -38.6299744, 38.6856651
10: -24.3269920, 38.5895081, -24.3045959, 38.4647865, -61.8075867, 61.8875122
11: -24.8075237, 17.6026936, -24.7629509, 17.5980377, -42.4055634, 42.3656464
12: -28.6811562, 20.1419678, -28.6412773, 20.1208134, -46.9444733, 46.8933067
13: -32.9555206, 28.8330936, -32.9257088, 28.7702408, -61.7257614, 61.7588043
14: -23.5661373, 39.2720871, -23.5057564, 39.1632233, -60.1240311, 60.1114044
15: -18.9743385, 25.8977070, -18.9415474, 25.8415413, -44.8158798, 44.8392563
16: -32.7547150, 19.9925346, -32.7283096, 19.8608646, -52.6155777, 52.7208443
17: -17.8231735, 38.5496407, -17.7742729, 38.4331932, -55.2891922, 55.3401680
18: -25.8788433, 19.6258144, -25.7788506, 19.6163769, -45.4952202, 45.4046631
19: -26.5327854, 12.5018711, -26.4037266, 12.5020561, -39.0348434, 38.9055977
20: -21.2201271, 20.4557705, -21.0786381, 20.4482193, -41.6683464, 41.5344086
21: -25.8354111, 18.9008083, -25.6854172, 18.8982925, -44.7337036, 44.5862274
22: -22.2441177, 24.5354958, -22.0786533, 24.5356407, -46.7797585, 46.6141510
23: -21.7496758, 17.5098667, -21.6865749, 17.4964371, -39.2461128, 39.1964417
24: -32.2406731, 11.9046345, -32.1128616, 11.9017334, -44.1424065, 44.0174942
25: -18.1974354, 25.4256935, -18.0923748, 25.4257889, -43.6232224, 43.5180664
26: -29.3549995, 26.9601784, -29.2124596, 26.9594345, -56.3144341, 56.1726379
27: -32.1969604, 16.5519485, -32.0939560, 16.5461864, -47.8882599, 47.7879448
28: -21.6148624, 21.7076797, -21.5176868, 21.7033501, -43.3182144, 43.2253647
29: -23.8037910, 22.2316246, -23.6799412, 22.2304134, -46.0342026, 45.9115677
30: -29.7128410, 16.8767242, -29.6094151, 16.8585281, -46.0350266, 45.9347343
31: -26.4933510, 19.1027508, -26.3385181, 19.0974445, -45.5907974, 45.4412689
32: -42.3102264, 8.5075159, -42.2136650, 8.4929047, -47.6795731, 47.5673141
33: -72.5253448, -5.5856457, -72.3229370, -5.5936089, -61.4919891, 61.3117371
34: -56.6166534, -5.4678020, -56.4575920, -5.4735298, -43.7590675, 43.6038742
35: -50.2937279, 0.0617790, -50.1085434, 0.0622196, -48.4169922, 48.2689323
36: -47.9233856, 4.9635582, -47.7372742, 4.9642782, -52.1959686, 51.9987793
37: -83.8116608, -17.4358635, -83.6318130, -17.4430180, -58.6196442, 58.3833122
38: -58.8753510, 3.2592010, -58.6048660, 3.2524929, -61.5471954, 61.2642212
39: -79.1368332, -11.5677395, -78.9208069, -11.5704794, -65.4957199, 65.2902374
40: -67.7506409, -18.3105354, -67.6390457, -18.3262310, -41.3370590, 41.1624489
41: -55.2508202, -6.8113594, -55.1685410, -6.8251753, -42.4290810, 42.2593727
42: -33.9816895, 6.8297787, -33.9497871, 6.8173828, -37.7553177, 37.6987457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=209, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 936
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 186
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1250
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1265
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1266
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 951
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 2016
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 953
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 480
type: A, layer: 1, pos: 1289
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 765

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258411, upper bound: 44.9902334
time: 50.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0258411, upper bound: 45.0258410
time: 25.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 79.10 seconds
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.7893924, upper bound: 45.0116209
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9415516, upper bound: 45.0138581
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8246502, upper bound: 45.0116218
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9769097, upper bound: 45.0138604
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.7686775, upper bound: 45.0312706
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9207864, upper bound: 45.0335053
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.7686775, upper bound: 45.0312751
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9561813, upper bound: 45.0335100
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8275285, upper bound: 45.0128873
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9797921, upper bound: 45.0151439
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8627566, upper bound: 44.9746047
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0151230, upper bound: 44.9768286
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8627795, upper bound: 45.0128883
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0151462, upper bound: 45.0151465
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8067824, upper bound: 45.0325311
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9589963, upper bound: 45.0347933
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.8421129, upper bound: 45.0325354
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9944058, upper bound: 45.0347997
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236081, upper bound: 44.6926997
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236081, upper bound: 44.7281654
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258381, upper bound: 44.8225909
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258381, upper bound: 44.8581746
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236103, upper bound: 44.7279761
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236103, upper bound: 44.7633671
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8579542
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8934650
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9487730, upper bound: 44.9889166
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9487730, upper bound: 45.0244471
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9873345, upper bound: 44.9889564
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9873345, upper bound: 45.0244905
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9872670, upper bound: 44.9901945
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -44.9872670, upper bound: 45.0257990
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8380079
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8735212
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258411, upper bound: 44.9902334
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 79.10
Output dim: 14, lower bound: -45.0258411, upper bound: 45.0258410

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.7890358, 16.9411297, -27.9820328, 17.0049858, -44.0030594, 44.1452866
1: -13.4780693, 16.9590244, -13.5986786, 17.0175591, -30.4956284, 30.5577030
2: -13.8828201, 21.5274696, -14.0114002, 21.5872536, -35.2211914, 35.3022461
3: -12.7331867, 23.2893448, -12.8364897, 23.3651009, -36.0982895, 36.1258354
4: -21.3403587, 18.3550606, -21.4890518, 18.4099045, -39.7502632, 39.8441124
5: -11.8187008, 22.7038193, -11.9320984, 22.7701473, -34.5888481, 34.6359177
6: -50.6221695, -3.7920485, -50.6617584, -3.6728039, -40.4031944, 40.2975998
7: -16.1852875, 18.3010483, -16.3186989, 18.3716373, -34.5569229, 34.6197472
8: -18.0510178, 21.1891556, -18.2134342, 21.2474365, -39.2984543, 39.4025879
9: -16.4353752, 23.1199265, -16.6081409, 23.2122383, -38.3158875, 38.3714066
10: -24.0270996, 38.3087540, -24.1910362, 38.4335938, -61.4678345, 61.4864845
11: -24.6806736, 17.4395905, -24.7420311, 17.5435219, -42.2241974, 42.1816216
12: -28.4654846, 19.8878555, -28.5811234, 20.0470486, -46.6565208, 46.5969505
13: -32.6776466, 28.6211605, -32.8379059, 28.7304420, -61.4080887, 61.4590683
14: -23.0414085, 39.0537262, -23.3147621, 39.1496620, -59.5877151, 59.7233238
15: -18.7445335, 25.7475300, -18.8611908, 25.8059196, -44.5504532, 44.6087189
16: -32.4925270, 19.7481956, -32.6390419, 19.8359375, -52.3284645, 52.3872375
17: -17.4858589, 38.3519707, -17.6572552, 38.4197769, -54.9432640, 55.0247040
18: -25.6662235, 19.4945831, -25.7519875, 19.5732975, -45.2395210, 45.2465706
19: -26.2936840, 12.3153887, -26.3788395, 12.4220972, -38.7157822, 38.6942291
20: -20.9495316, 20.2540989, -21.0449333, 20.3720551, -41.3215866, 41.2990341
21: -25.5412846, 18.6680832, -25.6482353, 18.8015041, -44.3427887, 44.3163185
22: -21.9544029, 24.3463573, -22.0525265, 24.4618053, -46.4162064, 46.3988838
23: -21.5830956, 17.3316498, -21.6627140, 17.4383259, -39.0214233, 38.9943619
24: -31.9742126, 11.6948376, -32.0912018, 11.8242769, -43.7984886, 43.7860413
25: -17.9841995, 25.2361965, -18.0640831, 25.3563786, -43.3405762, 43.3002777
26: -29.0313492, 26.6861324, -29.1712799, 26.8682861, -55.8996353, 55.8574142
27: -31.9551201, 16.3372803, -32.0682602, 16.4668808, -47.5382881, 47.5450592
28: -21.4087944, 21.4986820, -21.4918404, 21.6263695, -43.0351639, 42.9905243
29: -23.5560169, 22.0610809, -23.6524544, 22.1754150, -45.7314301, 45.7135353
30: -29.5247383, 16.6775856, -29.5893917, 16.7890453, -45.7678223, 45.7173538
31: -26.1974716, 18.8659477, -26.3016205, 19.0009155, -45.1983871, 45.1675682
32: -42.1268082, 8.3143473, -42.1865730, 8.4192514, -47.3995056, 47.3346939
33: -72.1951294, -5.9008589, -72.2982254, -5.7354918, -61.0229492, 60.9659500
34: -56.3760834, -5.7090549, -56.4394188, -5.5782566, -43.4034195, 43.3437309
35: -50.0121536, -0.1655951, -50.0864067, -0.0484190, -48.0079727, 47.9995575
36: -47.6052475, 4.6896610, -47.7029037, 4.8408413, -51.7407990, 51.6850357
37: -83.4864502, -17.6665134, -83.6026611, -17.5401955, -58.1816101, 58.1386681
38: -58.4604187, 2.9359751, -58.5686913, 3.1067104, -60.9532623, 60.8950577
39: -78.7848969, -11.7830849, -78.8915558, -11.6820765, -65.0343170, 65.0512238
40: -67.5297241, -18.4848633, -67.6136322, -18.3950272, -41.0552635, 41.0042915
41: -55.0815163, -7.0515966, -55.1516190, -6.9222498, -42.1352577, 42.0040474
42: -33.9144058, 6.6473646, -33.9377365, 6.7596960, -37.5894203, 37.4952621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7447073, upper bound: 45.0026205
time: 94.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7802455, upper bound: 45.0026205
time: 89.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.8164291, 16.9802742, -27.9874153, 17.0174828, -44.0435410, 44.1780167
1: -13.4990196, 17.0046310, -13.6009378, 17.0330887, -30.5321083, 30.6055679
2: -13.9168310, 21.5844097, -14.0144920, 21.6066341, -35.2754517, 35.3528824
3: -12.7508268, 23.3410664, -12.8391523, 23.3829346, -36.1337624, 36.1802177
4: -21.3776016, 18.4065132, -21.4934425, 18.4271393, -39.8047409, 39.8999557
5: -11.8349447, 22.7457905, -11.9340611, 22.7845001, -34.6194458, 34.6798515
6: -50.6374359, -3.7658243, -50.6680756, -3.6674690, -40.4234085, 40.3550072
7: -16.2157097, 18.3672409, -16.3220673, 18.3956451, -34.6113548, 34.6893082
8: -18.0893898, 21.2497082, -18.2182121, 21.2681313, -39.3575211, 39.4679184
9: -16.5079327, 23.1648502, -16.6322117, 23.2170925, -38.3775024, 38.4385681
10: -24.0862694, 38.3666000, -24.2104244, 38.4410477, -61.5273285, 61.5627480
11: -24.7012615, 17.4611664, -24.7508945, 17.5475616, -42.2488251, 42.2120590
12: -28.5816193, 19.9561920, -28.6210747, 20.0529308, -46.7652016, 46.7056236
13: -32.7531624, 28.6754913, -32.8627815, 28.7380810, -61.4912415, 61.5382729
14: -23.1916122, 39.1005135, -23.3643589, 39.1523895, -59.7119637, 59.8207397
15: -18.7757206, 25.7621040, -18.8681984, 25.8158417, -44.5915604, 44.6303024
16: -32.5243073, 19.7816277, -32.6484985, 19.8429356, -52.3672409, 52.4301262
17: -17.5769920, 38.3770752, -17.6876469, 38.4232788, -55.0204697, 55.0804443
18: -25.7162628, 19.5091820, -25.7616653, 19.5776176, -45.2938805, 45.2708473
19: -26.3196678, 12.3195038, -26.3864899, 12.4233150, -38.7429810, 38.7059937
20: -21.0001202, 20.2747688, -21.0609741, 20.3739491, -41.3740692, 41.3357430
21: -25.5895367, 18.6867714, -25.6640511, 18.8037682, -44.3933029, 44.3508224
22: -22.0008144, 24.3648415, -22.0672855, 24.4646568, -46.4654694, 46.4321289
23: -21.6105804, 17.3401222, -21.6697540, 17.4407234, -39.0513039, 39.0098763
24: -32.0159149, 11.7314949, -32.0963898, 11.8362980, -43.8522110, 43.8278847
25: -18.0073013, 25.2477551, -18.0710297, 25.3591766, -43.3664780, 43.3187866
26: -29.1112347, 26.7221737, -29.1964817, 26.8707657, -55.9820023, 55.9186554
27: -32.0028839, 16.3625584, -32.0760117, 16.4755421, -47.5957947, 47.5540504
28: -21.4394798, 21.5051708, -21.5004196, 21.6278648, -43.0673447, 43.0055923
29: -23.6062813, 22.0872421, -23.6703968, 22.1775475, -45.7838287, 45.7576370
30: -29.5397129, 16.6983814, -29.5942726, 16.7929077, -45.7871132, 45.7478485
31: -26.2376461, 18.8898926, -26.3102112, 19.0085278, -45.2461739, 45.2001038
32: -42.1669846, 8.3488503, -42.2000275, 8.4236135, -47.4436760, 47.4109154
33: -72.2358627, -5.8266373, -72.3021851, -5.7115135, -61.0890961, 61.0329285
34: -56.4103813, -5.6839714, -56.4450989, -5.5700941, -43.4519958, 43.3764839
35: -50.0426788, -0.1399813, -50.0922394, -0.0401649, -48.0748444, 48.0309982
36: -47.6645012, 4.7102041, -47.7231064, 4.8435183, -51.8078308, 51.7372894
37: -83.5229492, -17.6277885, -83.6110382, -17.5273170, -58.2254333, 58.1836395
38: -58.4958839, 2.9480400, -58.5788651, 3.1095438, -61.0003662, 60.9436035
39: -78.8150940, -11.7579956, -78.8968964, -11.6739693, -65.0713196, 65.0719299
40: -67.5613861, -18.4229355, -67.6188507, -18.3736000, -41.0708160, 41.0627594
41: -55.1080933, -7.0258570, -55.1570435, -6.9145269, -42.1447220, 42.0798988
42: -33.9144287, 6.6779261, -33.9407425, 6.7643242, -37.5970306, 37.5588531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8968111, upper bound: 45.0048403
time: 31.29 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9324096, upper bound: 45.0048403
time: 20.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -27.8672791, 16.9707241, -28.0061150, 17.0078068, -44.0868340, 44.2071266
1: -13.5245342, 16.9843330, -13.6136370, 17.0192184, -30.5437527, 30.5979691
2: -13.9257250, 21.5467720, -14.0253506, 21.5889301, -35.2684975, 35.3430099
3: -12.7838726, 23.3226814, -12.8533077, 23.3683090, -36.1521835, 36.1759872
4: -21.3932838, 18.3717537, -21.5056572, 18.4126034, -39.8058853, 39.8774109
5: -11.8664665, 22.7360992, -11.9478655, 22.7732315, -34.6396980, 34.6839638
6: -50.6427078, -3.7324986, -50.6637955, -3.6533766, -40.4442902, 40.3563194
7: -16.2244759, 18.3242111, -16.3312817, 18.3746033, -34.5990791, 34.6554947
8: -18.1042137, 21.2021523, -18.2301502, 21.2500744, -39.3542862, 39.4323044
9: -16.5158730, 23.1657982, -16.6342010, 23.2155399, -38.3934364, 38.4439430
10: -24.1136360, 38.3714561, -24.2194366, 38.4400253, -61.5531998, 61.5782051
11: -24.7213631, 17.5089951, -24.7448292, 17.5659485, -42.2873116, 42.2538223
12: -28.5085735, 19.9369240, -28.5944958, 20.0559196, -46.7047615, 46.6590881
13: -32.7515335, 28.6677399, -32.8616028, 28.7364693, -61.4880028, 61.5293427
14: -23.1605415, 39.1003342, -23.3526440, 39.1526794, -59.7016716, 59.8094406
15: -18.8116798, 25.7681389, -18.8826561, 25.8094597, -44.6211395, 44.6507950
16: -32.5607452, 19.8001766, -32.6594696, 19.8405647, -52.4013100, 52.4596481
17: -17.5695076, 38.3928833, -17.6836319, 38.4224548, -55.0252075, 55.0925598
18: -25.6986923, 19.5406532, -25.7560349, 19.5878563, -45.2865486, 45.2966881
19: -26.3529835, 12.3897543, -26.3834476, 12.4470348, -38.8000183, 38.7732010
20: -20.9999428, 20.3166847, -21.0491562, 20.3928528, -41.3927956, 41.3658409
21: -25.6021976, 18.7438908, -25.6534843, 18.8269138, -44.4291115, 44.3973770
22: -22.0087032, 24.3924103, -22.0572014, 24.4767685, -46.4854736, 46.4496117
23: -21.6352005, 17.4011230, -21.6665478, 17.4610538, -39.0962524, 39.0676727
24: -32.0437012, 11.7702255, -32.0951004, 11.8493586, -43.8930588, 43.8653259
25: -18.0366669, 25.2980766, -18.0694027, 25.3766556, -43.4133224, 43.3674774
26: -29.0982933, 26.7724342, -29.1770115, 26.8972740, -55.9955673, 55.9494476
27: -32.0160751, 16.4114876, -32.0721207, 16.4915123, -47.6236343, 47.6179810
28: -21.4609528, 21.5714588, -21.4961395, 21.6506519, -43.1116028, 43.0675964
29: -23.6113605, 22.1071949, -23.6560802, 22.1904030, -45.8017654, 45.7632751
30: -29.5691032, 16.7436905, -29.5926056, 16.8105087, -45.8345222, 45.7857933
31: -26.2557411, 18.9389915, -26.3083763, 19.0251141, -45.2808533, 45.2473679
32: -42.1482773, 8.3441362, -42.1894150, 8.4278669, -47.4359932, 47.3695107
33: -72.2514954, -5.8320866, -72.3038940, -5.7132244, -61.1042480, 61.0391312
34: -56.4016495, -5.6599712, -56.4422874, -5.5625801, -43.4460297, 43.3949852
35: -50.0517159, -0.1199722, -50.0905533, -0.0333128, -48.0647430, 48.0503235
36: -47.6529884, 4.7609949, -47.7075119, 4.8647327, -51.8128662, 51.7578964
37: -83.5635529, -17.5852795, -83.6087952, -17.5131950, -58.2882919, 58.2097549
38: -58.5263062, 3.0299072, -58.5740967, 3.1368303, -61.0508728, 60.9850769
39: -78.8534088, -11.7220116, -78.8978958, -11.6620665, -65.1251450, 65.1105347
40: -67.5703735, -18.4428387, -67.6180573, -18.3812122, -41.1139832, 41.0430107
41: -55.1189499, -6.9775515, -55.1543617, -6.8980370, -42.1978722, 42.0704422
42: -33.9357910, 6.6972847, -33.9399681, 6.7756548, -37.6356049, 37.5464478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7800101, upper bound: 45.0026349
time: 41.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8154808, upper bound: 45.0026349
time: 24.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -27.8946800, 17.0098705, -28.0115089, 17.0202999, -44.1273232, 44.2398491
1: -13.5454941, 17.0299339, -13.6159248, 17.0347366, -30.5802307, 30.6458588
2: -13.9597578, 21.6037140, -14.0284271, 21.6083164, -35.3227615, 35.3936234
3: -12.8015347, 23.3744087, -12.8559513, 23.3861408, -36.1876755, 36.2303619
4: -21.4305305, 18.4232063, -21.5100403, 18.4298420, -39.8603745, 39.9332466
5: -11.8827362, 22.7780800, -11.9498711, 22.7875671, -34.6703033, 34.7279510
6: -50.6579704, -3.7062864, -50.6700974, -3.6480188, -40.4645157, 40.4137268
7: -16.2549038, 18.3904037, -16.3346462, 18.3986130, -34.6535187, 34.7250519
8: -18.1426201, 21.2626801, -18.2349319, 21.2707710, -39.4133911, 39.4976120
9: -16.5884285, 23.2107086, -16.6582413, 23.2203903, -38.4550209, 38.5110893
10: -24.1728458, 38.4292831, -24.2388325, 38.4474983, -61.6127319, 61.6544647
11: -24.7419453, 17.5305862, -24.7536907, 17.5700111, -42.3119583, 42.2842789
12: -28.6247463, 20.0052433, -28.6344261, 20.0618019, -46.8134537, 46.7678070
13: -32.8270302, 28.7220421, -32.8864975, 28.7440681, -61.5710983, 61.6085396
14: -23.3107338, 39.1471252, -23.4022446, 39.1554298, -59.8259239, 59.9068527
15: -18.8428879, 25.7826805, -18.8896904, 25.8194103, -44.6623001, 44.6723709
16: -32.5925217, 19.8336792, -32.6689377, 19.8475628, -52.4400864, 52.5026169
17: -17.6606674, 38.4180069, -17.7139969, 38.4259415, -55.1024475, 55.1482887
18: -25.7487183, 19.5552673, -25.7657108, 19.5921936, -45.3409119, 45.3209763
19: -26.3789768, 12.3938589, -26.3910904, 12.4482555, -38.8272324, 38.7849503
20: -21.0505409, 20.3373299, -21.0652027, 20.3947468, -41.4452896, 41.4025345
21: -25.6504421, 18.7625656, -25.6693382, 18.8291569, -44.4795990, 44.4319038
22: -22.0551262, 24.4108925, -22.0719452, 24.4795990, -46.5347252, 46.4828377
23: -21.6627102, 17.4095917, -21.6735592, 17.4634857, -39.1261978, 39.0831528
24: -32.0854111, 11.8068790, -32.1002884, 11.8614073, -43.9468193, 43.9071655
25: -18.0597382, 25.3096542, -18.0763588, 25.3794708, -43.4392090, 43.3860130
26: -29.1782055, 26.8084774, -29.2022171, 26.8997593, -56.0779648, 56.0106964
27: -32.0638351, 16.4367733, -32.0798492, 16.5001984, -47.6811295, 47.6269722
28: -21.4916420, 21.5779400, -21.5047302, 21.6521549, -43.1437988, 43.0826721
29: -23.6616001, 22.1333618, -23.6740017, 22.1925430, -45.8541412, 45.8073654
30: -29.5840759, 16.7645130, -29.5974579, 16.8143501, -45.8538094, 45.8163147
31: -26.2958832, 18.9629326, -26.3169670, 19.0327187, -45.3286018, 45.2798996
32: -42.1884460, 8.3786440, -42.2028732, 8.4322119, -47.4801407, 47.4457664
33: -72.2921906, -5.7578716, -72.3078461, -5.6892147, -61.1703873, 61.1060638
34: -56.4358749, -5.6348782, -56.4479942, -5.5544090, -43.4946289, 43.4276962
35: -50.0822525, -0.0943422, -50.0963860, -0.0250502, -48.1316757, 48.0817795
36: -47.7122612, 4.7815218, -47.7277107, 4.8674307, -51.8799591, 51.8101730
37: -83.6000900, -17.5465012, -83.6171646, -17.5003452, -58.3321075, 58.2547531
38: -58.5617599, 3.0419779, -58.5842133, 3.1396294, -61.0979614, 61.0337219
39: -78.8835754, -11.6969690, -78.9032593, -11.6539307, -65.1621323, 65.1313171
40: -67.6020355, -18.3809242, -67.6233139, -18.3597755, -41.1295471, 41.1014633
41: -55.1455116, -6.9517956, -55.1597900, -6.8903112, -42.2073593, 42.1462898
42: -33.9358063, 6.7278347, -33.9429741, 6.7802868, -37.6432495, 37.6100464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9321783, upper bound: 45.0048451
time: 45.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9677080, upper bound: 45.0048451
time: 28.48 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.7890358, 16.9411297, -28.0688553, 17.0709610, -44.0737991, 44.2272530
1: -13.4780693, 16.9590244, -13.6351023, 17.0618591, -30.5399284, 30.5941277
2: -13.8828201, 21.5274696, -14.0468493, 21.6345291, -35.2732086, 35.3354340
3: -12.7331867, 23.2893448, -12.8629494, 23.4037437, -36.1369324, 36.1522942
4: -21.3403587, 18.3550606, -21.5217514, 18.4310989, -39.7714577, 39.8768120
5: -11.8187008, 22.7038193, -11.9673042, 22.8060818, -34.6247826, 34.6711235
6: -50.6221695, -3.7920485, -50.7690659, -3.6136293, -40.4628754, 40.4137764
7: -16.1852875, 18.3010483, -16.3553734, 18.4054470, -34.5907364, 34.6564217
8: -18.0510178, 21.1891556, -18.2721558, 21.3172035, -39.3682213, 39.4613113
9: -16.4353752, 23.1199265, -16.6475277, 23.2697754, -38.3725014, 38.4095612
10: -24.0270996, 38.3087540, -24.2419949, 38.5145950, -61.5547943, 61.5393143
11: -24.6806736, 17.4395905, -24.7763729, 17.5522652, -42.2329407, 42.2159653
12: -28.4654846, 19.8878555, -28.5997772, 20.0939445, -46.6993599, 46.6139870
13: -32.6776466, 28.6211605, -32.8614578, 28.7783127, -61.4559593, 61.4826202
14: -23.0414085, 39.0537262, -23.4134598, 39.2283630, -59.6681099, 59.8227959
15: -18.7445335, 25.7475300, -18.9116955, 25.8591366, -44.6036682, 44.6592255
16: -32.4925270, 19.7481956, -32.6771851, 19.8964291, -52.3889542, 52.4253807
17: -17.4858589, 38.3519707, -17.7130299, 38.4939041, -55.0175133, 55.0810890
18: -25.6662235, 19.4945831, -25.8222752, 19.5849648, -45.2511902, 45.3168564
19: -26.2936840, 12.3153887, -26.4678688, 12.4532604, -38.7469444, 38.7832565
20: -20.9495316, 20.2540989, -21.1350784, 20.4053020, -41.3548355, 41.3891754
21: -25.5412846, 18.6680832, -25.7555237, 18.8437958, -44.3850784, 44.4236069
22: -21.9544029, 24.3463573, -22.1364288, 24.4888153, -46.4432182, 46.4827881
23: -21.5830956, 17.3316498, -21.6989784, 17.4577217, -39.0408173, 39.0306282
24: -31.9742126, 11.6948376, -32.1907578, 11.8407106, -43.8149223, 43.8855972
25: -17.9841995, 25.2361965, -18.1180534, 25.3707047, -43.3549042, 43.3542480
26: -29.0313492, 26.6861324, -29.2234116, 26.8878479, -55.9191971, 55.9095459
27: -31.9551201, 16.3372803, -32.1323280, 16.4852848, -47.5644226, 47.6153526
28: -21.4087944, 21.4986820, -21.5396843, 21.6470337, -43.0558281, 43.0383682
29: -23.5560169, 22.0610809, -23.7204018, 22.1866970, -45.7427139, 45.7814827
30: -29.5247383, 16.6775856, -29.6759148, 16.8209915, -45.7920532, 45.8017540
31: -26.1974716, 18.8659477, -26.4208145, 19.0398445, -45.2373161, 45.2867622
32: -42.1268082, 8.3143473, -42.2612534, 8.4671164, -47.4469070, 47.4103088
33: -72.1951294, -5.9008589, -72.4484558, -5.6686573, -61.0900421, 61.1163712
34: -56.3760834, -5.7090549, -56.5357018, -5.5272532, -43.4562836, 43.4444656
35: -50.0121536, -0.1655951, -50.2082138, 0.0061522, -48.0634232, 48.1220779
36: -47.6052475, 4.6896610, -47.8036041, 4.8998270, -51.7997284, 51.7877808
37: -83.4864502, -17.6665134, -83.7157288, -17.4992599, -58.2219543, 58.2541122
38: -58.4604187, 2.9359751, -58.7274704, 3.1769485, -61.0240326, 61.0537338
39: -78.7848969, -11.7830849, -79.0558929, -11.6171370, -65.0994186, 65.2171631
40: -67.5297241, -18.4848633, -67.7075272, -18.3619633, -41.0868530, 41.1002922
41: -55.0815163, -7.0515966, -55.2133942, -6.8729496, -42.1860504, 42.0700760
42: -33.9144058, 6.6473646, -33.9513779, 6.7797232, -37.6114197, 37.5101013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7239320, upper bound: 45.0222767
time: 25.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7596017, upper bound: 45.0222769
time: 28.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.8164291, 16.9802742, -28.0742531, 17.0834484, -44.1142807, 44.2599716
1: -13.4990196, 17.0046310, -13.6373796, 17.0773964, -30.5764160, 30.6420097
2: -13.9168310, 21.5844097, -14.0499277, 21.6539040, -35.3274765, 35.3860550
3: -12.7508268, 23.3410664, -12.8656082, 23.4215698, -36.1723976, 36.2066727
4: -21.3776016, 18.4065132, -21.5261383, 18.4483299, -39.8259315, 39.9326515
5: -11.8349447, 22.7457905, -11.9692783, 22.8204098, -34.6553535, 34.7150688
6: -50.6374359, -3.7658243, -50.7753372, -3.6082654, -40.4831085, 40.4711914
7: -16.2157097, 18.3672409, -16.3587360, 18.4294701, -34.6451797, 34.7259750
8: -18.0893898, 21.2497082, -18.2769279, 21.3378906, -39.4272804, 39.5266342
9: -16.5079327, 23.1648502, -16.6715889, 23.2746315, -38.4341240, 38.4767303
10: -24.0862694, 38.3666000, -24.2613697, 38.5220566, -61.6142883, 61.6155663
11: -24.7012615, 17.4611664, -24.7852364, 17.5562878, -42.2575493, 42.2464027
12: -28.5816193, 19.9561920, -28.6397438, 20.0998077, -46.8080597, 46.7226562
13: -32.7531624, 28.6754913, -32.8863449, 28.7859077, -61.5390701, 61.5618362
14: -23.1916122, 39.1005135, -23.4630260, 39.2310753, -59.7923660, 59.9202042
15: -18.7757206, 25.7621040, -18.9187298, 25.8690758, -44.6447983, 44.6808319
16: -32.5243073, 19.7816277, -32.6866264, 19.9034119, -52.4277191, 52.4682541
17: -17.5769920, 38.3770752, -17.7433987, 38.4974174, -55.0947227, 55.1368179
18: -25.7162628, 19.5091820, -25.8319321, 19.5892887, -45.3055496, 45.3411140
19: -26.3196678, 12.3195038, -26.4755268, 12.4544601, -38.7741280, 38.7950287
20: -21.0001202, 20.2747688, -21.1511116, 20.4071884, -41.4073105, 41.4258804
21: -25.5895367, 18.6867714, -25.7713852, 18.8460388, -44.4355774, 44.4581566
22: -22.0008144, 24.3648415, -22.1511898, 24.4916267, -46.4924393, 46.5160294
23: -21.6105804, 17.3401222, -21.7059975, 17.4601364, -39.0707169, 39.0461197
24: -32.0159149, 11.7314949, -32.1959496, 11.8527317, -43.8686447, 43.9274445
25: -18.0073013, 25.2477551, -18.1250038, 25.3734970, -43.3807983, 43.3727570
26: -29.1112347, 26.7221737, -29.2486267, 26.8903313, -56.0015640, 55.9708023
27: -32.0028839, 16.3625584, -32.1400452, 16.4939690, -47.6219025, 47.6243134
28: -21.4394798, 21.5051708, -21.5482807, 21.6485424, -43.0880203, 43.0534515
29: -23.6062813, 22.0872421, -23.7383404, 22.1888161, -45.7950974, 45.8255844
30: -29.5397129, 16.6983814, -29.6807938, 16.8248444, -45.8113289, 45.8322449
31: -26.2376461, 18.8898926, -26.4294052, 19.0474396, -45.2850876, 45.3192978
32: -42.1669846, 8.3488503, -42.2747269, 8.4714937, -47.4910660, 47.4865189
33: -72.2358627, -5.8266373, -72.4523773, -5.6446171, -61.1562042, 61.1833496
34: -56.4103813, -5.6839714, -56.5414047, -5.5190907, -43.5048828, 43.4772186
35: -50.0426788, -0.1399813, -50.2139931, 0.0143890, -48.1303101, 48.1535110
36: -47.6645012, 4.7102041, -47.8238182, 4.9025316, -51.8667450, 51.8400116
37: -83.5229492, -17.6277885, -83.7240601, -17.4863911, -58.2657928, 58.2991104
38: -58.4958839, 2.9480400, -58.7375793, 3.1797495, -61.0711823, 61.1023331
39: -78.8150940, -11.7579956, -79.0612640, -11.6090231, -65.1364059, 65.2379150
40: -67.5613861, -18.4229355, -67.7127609, -18.3405190, -41.1024208, 41.1587753
41: -55.1080933, -7.0258570, -55.2187881, -6.8652220, -42.1955490, 42.1459503
42: -33.9144287, 6.6779261, -33.9543877, 6.7843456, -37.6190491, 37.5736923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8760054, upper bound: 45.0244886
time: 47.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9117086, upper bound: 45.0244884
time: 31.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -27.8672791, 16.9707241, -28.0929642, 17.0737610, -44.1575928, 44.2890968
1: -13.5245342, 16.9843330, -13.6500854, 17.0635071, -30.5880413, 30.6344185
2: -13.9257250, 21.5467720, -14.0608196, 21.6362076, -35.3205147, 35.3761826
3: -12.7838726, 23.3226814, -12.8797722, 23.4069405, -36.1908112, 36.2024536
4: -21.3932838, 18.3717537, -21.5383816, 18.4338074, -39.8270912, 39.9101334
5: -11.8664665, 22.7360992, -11.9830742, 22.8091507, -34.6756172, 34.7191734
6: -50.6427078, -3.7324986, -50.7710648, -3.5941997, -40.5040054, 40.4724922
7: -16.2244759, 18.3242111, -16.3679523, 18.4084206, -34.6328964, 34.6921616
8: -18.1042137, 21.2021523, -18.2888603, 21.3198185, -39.4240341, 39.4910126
9: -16.5158730, 23.1657982, -16.6735840, 23.2730598, -38.4500389, 38.4821243
10: -24.1136360, 38.3714561, -24.2703991, 38.5210266, -61.6401672, 61.6310349
11: -24.7213631, 17.5089951, -24.7791805, 17.5746994, -42.2960625, 42.2881775
12: -28.5085735, 19.9369240, -28.6131306, 20.1028538, -46.7476540, 46.6761284
13: -32.7515335, 28.6677399, -32.8851662, 28.7843246, -61.5358582, 61.5529060
14: -23.1605415, 39.1003342, -23.4513397, 39.2313766, -59.7820740, 59.9088783
15: -18.8116798, 25.7681389, -18.9331970, 25.8627071, -44.6743851, 44.7013359
16: -32.5607452, 19.8001766, -32.6976166, 19.9010181, -52.4617615, 52.4977951
17: -17.5695076, 38.3928833, -17.7393913, 38.4965668, -55.0994568, 55.1489143
18: -25.6986923, 19.5406532, -25.8262882, 19.5995216, -45.2982140, 45.3669434
19: -26.3529835, 12.3897543, -26.4724560, 12.4782352, -38.8312187, 38.8622093
20: -20.9999428, 20.3166847, -21.1392899, 20.4260712, -41.4260139, 41.4559746
21: -25.6021976, 18.7438908, -25.7607994, 18.8691998, -44.4713974, 44.5046921
22: -22.0087032, 24.3924103, -22.1410789, 24.5037689, -46.5124741, 46.5334892
23: -21.6352005, 17.4011230, -21.7028027, 17.4804764, -39.1156769, 39.1039276
24: -32.0437012, 11.7702255, -32.1946640, 11.8658152, -43.9095154, 43.9648895
25: -18.0366669, 25.2980766, -18.1233826, 25.3909836, -43.4276505, 43.4214592
26: -29.0982933, 26.7724342, -29.2291641, 26.9168396, -56.0151329, 56.0015984
27: -32.0160751, 16.4114876, -32.1361427, 16.5099220, -47.6497192, 47.6882439
28: -21.4609528, 21.5714588, -21.5439758, 21.6713371, -43.1322899, 43.1154327
29: -23.6113605, 22.1071949, -23.7240009, 22.2016869, -45.8130493, 45.8311958
30: -29.5691032, 16.7436905, -29.6790905, 16.8424511, -45.8587570, 45.8702011
31: -26.2557411, 18.9389915, -26.4275475, 19.0640488, -45.3197899, 45.3665390
32: -42.1482773, 8.3441362, -42.2641068, 8.4757223, -47.4833488, 47.4451141
33: -72.2514954, -5.8320866, -72.4541168, -5.6462975, -61.1713562, 61.1895523
34: -56.4016495, -5.6599712, -56.5386124, -5.5115652, -43.4989052, 43.4957047
35: -50.0517159, -0.1199722, -50.2123184, 0.0212593, -48.1202240, 48.1728249
36: -47.6529884, 4.7609949, -47.8082008, 4.9237881, -51.8717880, 51.8606262
37: -83.5635529, -17.5852795, -83.7218094, -17.4722557, -58.3286667, 58.3251953
38: -58.5263062, 3.0299072, -58.7328072, 3.2070723, -61.1216736, 61.1437302
39: -78.8534088, -11.7220116, -79.0622406, -11.5971231, -65.1902390, 65.2765045
40: -67.5703735, -18.4428387, -67.7119598, -18.3481102, -41.1456070, 41.1389771
41: -55.1189499, -6.9775515, -55.2161255, -6.8487358, -42.2486954, 42.1364594
42: -33.9357910, 6.6972847, -33.9536285, 6.7956963, -37.6576233, 37.5613022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7593058, upper bound: 45.0222785
time: 25.24 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7948736, upper bound: 45.0222785
time: 27.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -27.8946800, 17.0098705, -28.0983467, 17.0862541, -44.1980629, 44.3218193
1: -13.5454941, 17.0299339, -13.6523628, 17.0790348, -30.6245289, 30.6822968
2: -13.9597578, 21.6037140, -14.0638924, 21.6555824, -35.3747787, 35.4268036
3: -12.8015347, 23.3744087, -12.8824234, 23.4247665, -36.2263031, 36.2568321
4: -21.4305305, 18.4232063, -21.5427437, 18.4510365, -39.8815689, 39.9659500
5: -11.8827362, 22.7780800, -11.9850712, 22.8234882, -34.7062225, 34.7631531
6: -50.6579704, -3.7062864, -50.7773514, -3.5888343, -40.5242424, 40.5299072
7: -16.2549038, 18.3904037, -16.3713169, 18.4324322, -34.6873360, 34.7617188
8: -18.1426201, 21.2626801, -18.2936401, 21.3405304, -39.4831505, 39.5563202
9: -16.5884285, 23.2107086, -16.6976509, 23.2779217, -38.5116272, 38.5492630
10: -24.1728458, 38.4292831, -24.2897911, 38.5284805, -61.6996765, 61.7072906
11: -24.7419453, 17.5305862, -24.7880268, 17.5787487, -42.3206940, 42.3186111
12: -28.6247463, 20.0052433, -28.6530876, 20.1087341, -46.8563385, 46.7848206
13: -32.8270302, 28.7220421, -32.9100380, 28.7919292, -61.6189575, 61.6320801
14: -23.3107338, 39.1471252, -23.5009136, 39.2340965, -59.9063377, 60.0062828
15: -18.8428879, 25.7826805, -18.9402351, 25.8726139, -44.7154999, 44.7229156
16: -32.5925217, 19.8336792, -32.7070694, 19.9080086, -52.5005302, 52.5407486
17: -17.6606674, 38.4180069, -17.7697697, 38.5000801, -55.1766930, 55.2046738
18: -25.7487183, 19.5552673, -25.8359642, 19.6038361, -45.3525543, 45.3912315
19: -26.3789768, 12.3938589, -26.4801121, 12.4794292, -38.8584061, 38.8739700
20: -21.0505409, 20.3373299, -21.1553211, 20.4279633, -41.4785042, 41.4926529
21: -25.6504421, 18.7625656, -25.7766266, 18.8714733, -44.5219154, 44.5391922
22: -22.0551262, 24.4108925, -22.1558285, 24.5065842, -46.5617104, 46.5667191
23: -21.6627102, 17.4095917, -21.7098160, 17.4828758, -39.1455841, 39.1194077
24: -32.0854111, 11.8068790, -32.1998329, 11.8778429, -43.9632530, 44.0067139
25: -18.0597382, 25.3096542, -18.1303062, 25.3937893, -43.4535294, 43.4399605
26: -29.1782055, 26.8084774, -29.2543354, 26.9193058, -56.0975113, 56.0628128
27: -32.0638351, 16.4367733, -32.1438866, 16.5186062, -47.7072372, 47.6972427
28: -21.4916420, 21.5779400, -21.5525856, 21.6728516, -43.1644936, 43.1305237
29: -23.6616001, 22.1333618, -23.7419357, 22.2038116, -45.8654099, 45.8752975
30: -29.5840759, 16.7645130, -29.6839504, 16.8462944, -45.8780441, 45.9006996
31: -26.2958832, 18.9629326, -26.4361496, 19.0716648, -45.3675461, 45.3990822
32: -42.1884460, 8.3786440, -42.2775726, 8.4801092, -47.5275650, 47.5213699
33: -72.2921906, -5.7578716, -72.4580231, -5.6223125, -61.2375031, 61.2564774
34: -56.4358749, -5.6348782, -56.5442810, -5.5034189, -43.5474815, 43.5284042
35: -50.0822525, -0.0943422, -50.2181473, 0.0294704, -48.1871338, 48.2043076
36: -47.7122612, 4.7815218, -47.8283997, 4.9264441, -51.9388885, 51.9128952
37: -83.6000900, -17.5465012, -83.7301788, -17.4594154, -58.3724365, 58.3702011
38: -58.5617599, 3.0419779, -58.7429314, 3.2098885, -61.1687622, 61.1923523
39: -78.8835754, -11.6969690, -79.0675583, -11.5889807, -65.2272568, 65.2972565
40: -67.6020355, -18.3809242, -67.7171936, -18.3266716, -41.1611633, 41.1974525
41: -55.1455116, -6.9517956, -55.2215309, -6.8409967, -42.2581596, 42.2123489
42: -33.9358063, 6.7278347, -33.9566269, 6.8002834, -37.6652679, 37.6248932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=245, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9114412, upper bound: 45.0244908
time: 55.54 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9470702, upper bound: 45.0244906
time: 36.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.9691200, 17.0104218, -28.0331001, 17.0093212, -44.1910019, 44.2672234
1: -13.5820265, 17.0090408, -13.6295366, 17.0196190, -30.6016464, 30.6385765
2: -13.9822178, 21.5831375, -14.0414581, 21.5892200, -35.3237762, 35.3896904
3: -12.8116322, 23.3392811, -12.8602819, 23.3694935, -36.1811256, 36.1995621
4: -21.4587784, 18.3883286, -21.5240135, 18.4127998, -39.8715782, 39.9123421
5: -11.9152832, 22.7658119, -11.9613981, 22.7738838, -34.6891670, 34.7272110
6: -50.6613846, -3.7021084, -50.6671753, -3.6468511, -40.4719391, 40.3893661
7: -16.2979431, 18.3682613, -16.3534012, 18.3749123, -34.6728554, 34.7216644
8: -18.1861095, 21.2268085, -18.2538910, 21.2509842, -39.4370956, 39.4806976
9: -16.5425339, 23.1900215, -16.6388283, 23.2164288, -38.4287643, 38.4763527
10: -24.1284313, 38.3871956, -24.2198639, 38.4414368, -61.5802307, 61.6086159
11: -24.7108784, 17.5023556, -24.7466183, 17.5607834, -42.2716599, 42.2489738
12: -28.5015488, 19.9771385, -28.5854168, 20.0719032, -46.7215424, 46.6875229
13: -32.7811241, 28.6835098, -32.8664093, 28.7397308, -61.5208549, 61.5499191
14: -23.1948166, 39.0997276, -23.3547306, 39.1532822, -59.7472115, 59.8166046
15: -18.8207054, 25.8080235, -18.8802452, 25.8220406, -44.6427460, 44.6882706
16: -32.6154938, 19.8450890, -32.6735306, 19.8419933, -52.4574890, 52.5186195
17: -17.5909367, 38.4088058, -17.6804962, 38.4230194, -55.0511475, 55.1158867
18: -25.7247810, 19.5529842, -25.7590942, 19.5895271, -45.3143082, 45.3120804
19: -26.3576965, 12.3921537, -26.3842068, 12.4461031, -38.8037987, 38.7763596
20: -21.0277958, 20.3393307, -21.0509415, 20.3973217, -41.4251175, 41.3902740
21: -25.6165619, 18.7639923, -25.6555634, 18.8310204, -44.4475822, 44.4195557
22: -22.0508575, 24.4437199, -22.0593224, 24.4918747, -46.5427322, 46.5030441
23: -21.6332474, 17.4115887, -21.6676636, 17.4619331, -39.0951805, 39.0792542
24: -32.0293808, 11.7757864, -32.0973434, 11.8483162, -43.8776970, 43.8731308
25: -18.0658092, 25.3376541, -18.0713806, 25.3870754, -43.4528847, 43.4090347
26: -29.1471329, 26.8178959, -29.1802387, 26.9081268, -56.0552597, 55.9981346
27: -32.0230331, 16.4336433, -32.0747833, 16.4955673, -47.6522636, 47.6511230
28: -21.4834747, 21.6074257, -21.4973755, 21.6589661, -43.1424408, 43.1048012
29: -23.6260357, 22.1478882, -23.6571751, 22.2014332, -45.8274689, 45.8050613
30: -29.5661793, 16.7574062, -29.5946064, 16.8110180, -45.8383141, 45.8050766
31: -26.2747192, 18.9644356, -26.3106003, 19.0308075, -45.3055267, 45.2750359
32: -42.1731300, 8.3947277, -42.1928787, 8.4425640, -47.4701195, 47.4211502
33: -72.2772369, -5.7968225, -72.3048248, -5.7055149, -61.1403351, 61.0816040
34: -56.4597702, -5.5970144, -56.4442940, -5.5444021, -43.5245285, 43.4570541
35: -50.1012001, -0.0652895, -50.0923157, -0.0180130, -48.1303253, 48.1062546
36: -47.7112961, 4.8121290, -47.7096634, 4.8785772, -51.8859177, 51.8157578
37: -83.5841141, -17.5975761, -83.6102066, -17.5199242, -58.3136215, 58.2187119
38: -58.6140251, 3.0825291, -58.5783463, 3.1500816, -61.1602631, 61.0541611
39: -78.8727493, -11.7193317, -78.8998032, -11.6633291, -65.1442108, 65.1269302
40: -67.5832672, -18.4493637, -67.6227493, -18.3858719, -41.1189423, 41.0483208
41: -55.1245918, -6.9630709, -55.1576462, -6.8960400, -42.2102547, 42.1002579
42: -33.9462624, 6.7287188, -33.9408569, 6.7831850, -37.6495743, 37.5779114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7829978, upper bound: 45.0039566
time: 54.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8185839, upper bound: 45.0039566
time: 35.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.9965019, 17.0495720, -28.0384827, 17.0218163, -44.2314491, 44.2999687
1: -13.6029787, 17.0546455, -13.6318045, 17.0351372, -30.6381149, 30.6864510
2: -14.0162554, 21.6400795, -14.0445385, 21.6085968, -35.3780518, 35.4403229
3: -12.8292789, 23.3910027, -12.8629227, 23.3873062, -36.2165833, 36.2539253
4: -21.4960022, 18.4397812, -21.5283813, 18.4300346, -39.9260368, 39.9681625
5: -11.9315147, 22.8078251, -11.9633894, 22.7882118, -34.7197266, 34.7712135
6: -50.6766739, -3.6758847, -50.6734734, -3.6415248, -40.4921646, 40.4467583
7: -16.3283386, 18.4344692, -16.3567772, 18.3989296, -34.7272682, 34.7912445
8: -18.2245140, 21.2873497, -18.2586575, 21.2716866, -39.4962006, 39.5460052
9: -16.6151085, 23.2349300, -16.6628857, 23.2213230, -38.4903793, 38.5435028
10: -24.1876106, 38.4450569, -24.2392445, 38.4489365, -61.6397324, 61.6848564
11: -24.7314529, 17.5239487, -24.7554817, 17.5648136, -42.2962646, 42.2794304
12: -28.6176605, 20.0455017, -28.6253510, 20.0777721, -46.8302155, 46.7962494
13: -32.8566437, 28.7377701, -32.8913078, 28.7473640, -61.6040077, 61.6290779
14: -23.3449554, 39.1464767, -23.4043217, 39.1560211, -59.8713989, 59.9139824
15: -18.8518867, 25.8225746, -18.8872719, 25.8319874, -44.6838760, 44.7098465
16: -32.6472588, 19.8785934, -32.6829605, 19.8489590, -52.4962158, 52.5615540
17: -17.6820641, 38.4339294, -17.7108574, 38.4265289, -55.1283264, 55.1716194
18: -25.7748337, 19.5675850, -25.7687569, 19.5938587, -45.3686905, 45.3363419
19: -26.3836594, 12.3962593, -26.3918533, 12.4473181, -38.8309784, 38.7881126
20: -21.0783806, 20.3600044, -21.0669823, 20.3992119, -41.4775925, 41.4269867
21: -25.6648026, 18.7826614, -25.6714115, 18.8332977, -44.4981003, 44.4540710
22: -22.0973053, 24.4621735, -22.0741081, 24.4946861, -46.5919914, 46.5362816
23: -21.6607285, 17.4200649, -21.6746712, 17.4643517, -39.1250801, 39.0947342
24: -32.0711098, 11.8124332, -32.1025162, 11.8603439, -43.9314537, 43.9149475
25: -18.0888710, 25.3492222, -18.0783463, 25.3898659, -43.4787369, 43.4275665
26: -29.2270870, 26.8539410, -29.2054558, 26.9105759, -56.1376648, 56.0593948
27: -32.0708008, 16.4589272, -32.0825195, 16.5042038, -47.7097664, 47.6600914
28: -21.5141582, 21.6138954, -21.5059681, 21.6604767, -43.1746368, 43.1198654
29: -23.6762695, 22.1740417, -23.6751003, 22.2035561, -45.8798256, 45.8491440
30: -29.5811844, 16.7782192, -29.5994701, 16.8148537, -45.8576164, 45.8355789
31: -26.3148613, 18.9883690, -26.3191929, 19.0384102, -45.3532715, 45.3075638
32: -42.2132721, 8.4292488, -42.2063522, 8.4469376, -47.5143013, 47.4974022
33: -72.3179779, -5.7225914, -72.3087921, -5.6814775, -61.2065125, 61.1485214
34: -56.4939842, -5.5719347, -56.4499855, -5.5362129, -43.5731010, 43.4897575
35: -50.1317291, -0.0396814, -50.0981522, -0.0097733, -48.1972427, 48.1377525
36: -47.7706528, 4.8326721, -47.7298775, 4.8812656, -51.9530869, 51.8680420
37: -83.6206360, -17.5588074, -83.6185532, -17.5070915, -58.3574524, 58.2637138
38: -58.6494789, 3.0946016, -58.5885162, 3.1528816, -61.2074432, 61.1027374
39: -78.9029617, -11.6942902, -78.9051208, -11.6552486, -65.1812592, 65.1476822
40: -67.6149139, -18.3874626, -67.6280060, -18.3644485, -41.1345444, 41.1067657
41: -55.1511459, -6.9373074, -55.1630554, -6.8883171, -42.2197037, 42.1761322
42: -33.9462814, 6.7592897, -33.9438667, 6.7877913, -37.6572342, 37.6415100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9352121, upper bound: 45.0061752
time: 26.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9708602, upper bound: 45.0061752
time: 27.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.0467567, 17.0759640, -27.9715805, 16.9946098, -44.2503128, 44.2637253
1: -13.6322861, 17.0780945, -13.5937748, 17.0104980, -30.6427841, 30.6718693
2: -14.0435371, 21.6574574, -14.0093546, 21.5911236, -35.3841324, 35.4260559
3: -12.8612747, 23.4204407, -12.8234692, 23.3549061, -36.2161789, 36.2439117
4: -21.5307293, 18.4532242, -21.4860439, 18.4140320, -39.9447632, 39.9392700
5: -11.9623013, 22.8365097, -11.9277878, 22.7595863, -34.7218857, 34.7642975
6: -50.6946335, -3.6377501, -50.6515884, -3.6902347, -40.4616394, 40.4579048
7: -16.3541298, 18.4543152, -16.3275928, 18.3787479, -34.7328796, 34.7819061
8: -18.2595901, 21.2974091, -18.2175560, 21.2609882, -39.5205765, 39.5149651
9: -16.6670647, 23.2769070, -16.6010952, 23.1766243, -38.4909325, 38.5291367
10: -24.2434540, 38.5004196, -24.1748505, 38.3921089, -61.6302948, 61.6800194
11: -24.7686768, 17.5684052, -24.7151546, 17.5093689, -42.2780457, 42.2835617
12: -28.6455135, 20.0846748, -28.5899773, 20.0333805, -46.8101273, 46.8026657
13: -32.9049683, 28.7776852, -32.8354797, 28.7051640, -61.6101303, 61.6131668
14: -23.4223747, 39.1897049, -23.3113594, 39.1110725, -59.8947372, 59.8719215
15: -18.8956680, 25.8388958, -18.8350697, 25.8128338, -44.7085037, 44.6739655
16: -32.6919670, 19.9253960, -32.6256485, 19.8035774, -52.4955444, 52.5510445
17: -17.7361526, 38.4716377, -17.6442680, 38.3862114, -55.1367607, 55.1454239
18: -25.8028584, 19.5977364, -25.7392483, 19.5583038, -45.3611603, 45.3369827
19: -26.4377422, 12.4428043, -26.3344269, 12.3897171, -38.8274612, 38.7772293
20: -21.1240044, 20.3990154, -21.0179367, 20.3488846, -41.4728889, 41.4169540
21: -25.7198772, 18.8302002, -25.6138973, 18.7742462, -44.4941254, 44.4440994
22: -22.1465073, 24.4917717, -22.0239677, 24.4590034, -46.6055107, 46.5157394
23: -21.7083549, 17.4648609, -21.6247540, 17.4125519, -39.1209068, 39.0896149
24: -32.1359787, 11.8607578, -32.0342941, 11.8044853, -43.9404640, 43.8950500
25: -18.1352959, 25.3889542, -18.0297985, 25.3430309, -43.4783249, 43.4187546
26: -29.2876606, 26.9084740, -29.1431808, 26.8446770, -56.1323395, 56.0516548
27: -32.1275024, 16.5065079, -32.0239563, 16.4487152, -47.7135773, 47.6439781
28: -21.5614452, 21.6605644, -21.4571419, 21.6070518, -43.1684952, 43.1177063
29: -23.7274265, 22.2035046, -23.6223927, 22.1669540, -45.8943787, 45.8258972
30: -29.6218529, 16.8208485, -29.5552464, 16.7635956, -45.8482285, 45.8321724
31: -26.3654785, 19.0345039, -26.2657261, 18.9818935, -45.3473740, 45.3002319
32: -42.2309418, 8.4492073, -42.1848946, 8.4202156, -47.5074158, 47.4924965
33: -72.3677521, -5.6781178, -72.2584686, -5.7335339, -61.2076111, 61.1391830
34: -56.5159760, -5.5393066, -56.4324074, -5.5730314, -43.5604935, 43.5035973
35: -50.1664543, -0.0094767, -50.0641289, -0.0411892, -48.2006912, 48.1327972
36: -47.8131485, 4.8785791, -47.6879005, 4.8290634, -51.9458313, 51.8685074
37: -83.6904755, -17.5062790, -83.5439529, -17.5661068, -58.3839035, 58.2231522
38: -58.7091522, 3.1559830, -58.5276985, 3.0810070, -61.2013245, 61.0930099
39: -78.9637909, -11.6550694, -78.8410339, -11.7014246, -65.2048874, 65.1125183
40: -67.6504059, -18.3607464, -67.5933456, -18.3972206, -41.1437569, 41.0871658
41: -55.1852608, -6.8886690, -55.1271973, -6.9425650, -42.2069855, 42.1779480
42: -33.9650307, 6.7918463, -33.9233704, 6.7475424, -37.6453629, 37.6449432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 14, lower bound: -44.9705638, upper bound: 44.9678631
time: 28.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0061518, upper bound: 44.9678631
time: 31.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.0474854, 17.0399628, -28.0571823, 17.0121346, -44.2748413, 44.3289986
1: -13.6285248, 17.0343437, -13.6445160, 17.0212631, -30.6497879, 30.6788597
2: -14.0251675, 21.6024246, -14.0554314, 21.5908966, -35.3710976, 35.4304581
3: -12.8623362, 23.3725929, -12.8771019, 23.3726921, -36.2350273, 36.2496948
4: -21.5118065, 18.4050083, -21.5406189, 18.4155064, -39.9273148, 39.9456253
5: -11.9630699, 22.7980728, -11.9771748, 22.7769470, -34.7400169, 34.7752457
6: -50.6819038, -3.6425381, -50.6691971, -3.6274190, -40.5130463, 40.4481277
7: -16.3371315, 18.3913956, -16.3659687, 18.3778725, -34.7150040, 34.7573624
8: -18.2394276, 21.2397995, -18.2705841, 21.2536201, -39.4930496, 39.5103836
9: -16.6230774, 23.2358761, -16.6648788, 23.2197495, -38.5063629, 38.5488968
10: -24.2150116, 38.4498444, -24.2482719, 38.4478760, -61.6656799, 61.7002907
11: -24.7515850, 17.5717888, -24.7494068, 17.5832367, -42.3348236, 42.3211975
12: -28.5446472, 20.0262222, -28.5987892, 20.0807915, -46.7698288, 46.7498589
13: -32.8550644, 28.7299557, -32.8901138, 28.7457428, -61.6008072, 61.6200714
14: -23.3139286, 39.1462708, -23.3926544, 39.1563187, -59.8611526, 59.9026566
15: -18.8879032, 25.8286400, -18.9017410, 25.8256073, -44.7135086, 44.7303810
16: -32.6836929, 19.8970490, -32.6939507, 19.8466148, -52.5303078, 52.5909996
17: -17.6746521, 38.4497223, -17.7068539, 38.4256859, -55.1331329, 55.1837273
18: -25.7572174, 19.5990868, -25.7631378, 19.6040916, -45.3613091, 45.3622246
19: -26.4169502, 12.4665222, -26.3887939, 12.4710426, -38.8879929, 38.8553162
20: -21.0781479, 20.4019127, -21.0551796, 20.4181099, -41.4962578, 41.4570923
21: -25.6773949, 18.8398209, -25.6608391, 18.8564415, -44.5338364, 44.5006599
22: -22.1051292, 24.4897690, -22.0639820, 24.5068245, -46.6119537, 46.5537491
23: -21.6853428, 17.4810638, -21.6714764, 17.4846878, -39.1700287, 39.1525421
24: -32.0988846, 11.8511581, -32.1012383, 11.8734093, -43.9722939, 43.9523964
25: -18.1182022, 25.3995132, -18.0767288, 25.4073524, -43.5255547, 43.4762421
26: -29.2140217, 26.9042206, -29.1859798, 26.9371223, -56.1511459, 56.0902023
27: -32.0839767, 16.5078621, -32.0785980, 16.5201912, -47.7375793, 47.7240753
28: -21.5355930, 21.6802025, -21.5016899, 21.6832771, -43.2188721, 43.1818924
29: -23.6813583, 22.1940022, -23.6607819, 22.2164116, -45.8977699, 45.8547821
30: -29.6105194, 16.8235397, -29.5977898, 16.8324585, -45.9050064, 45.8735847
31: -26.3329334, 19.0374908, -26.3173599, 19.0550137, -45.3879471, 45.3548508
32: -42.1945915, 8.4245625, -42.1957169, 8.4511690, -47.5065994, 47.4561043
33: -72.3335800, -5.7280273, -72.3105469, -5.6831570, -61.2216415, 61.1547699
34: -56.4852524, -5.5478992, -56.4472122, -5.5286999, -43.5671158, 43.5083199
35: -50.1407433, -0.0196600, -50.0964432, -0.0029554, -48.1870880, 48.1571007
36: -47.7590027, 4.8834543, -47.7142906, 4.9024897, -51.9579315, 51.8886795
37: -83.6612701, -17.5163116, -83.6163025, -17.4929504, -58.4203110, 58.2898102
38: -58.6798630, 3.1764498, -58.5837402, 3.1802263, -61.2578125, 61.1441345
39: -78.9412613, -11.6582880, -78.9060974, -11.6433201, -65.2350616, 65.1862793
40: -67.6239166, -18.4073582, -67.6272125, -18.3720627, -41.1776733, 41.0870705
41: -55.1620064, -6.8889885, -55.1603851, -6.8718033, -42.2728806, 42.1666718
42: -33.9676437, 6.7786646, -33.9430885, 6.7991514, -37.6957970, 37.6291351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8183121, upper bound: 45.0039640
time: 29.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8538154, upper bound: 45.0039640
time: 56.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.0749054, 17.0791245, -28.0625763, 17.0246201, -44.3153229, 44.3617744
1: -13.6494713, 17.0799427, -13.6467972, 17.0367851, -30.6862564, 30.7267399
2: -14.0591946, 21.6593704, -14.0585127, 21.6102810, -35.4253578, 35.4810791
3: -12.8799877, 23.4243202, -12.8797350, 23.3905296, -36.2705154, 36.3040543
4: -21.5490265, 18.4564552, -21.5449944, 18.4327316, -39.9817581, 40.0014496
5: -11.9793091, 22.8401432, -11.9791775, 22.7912922, -34.7705994, 34.8193207
6: -50.6971855, -3.6163244, -50.6754913, -3.6220741, -40.5332756, 40.5055504
7: -16.3675404, 18.4576187, -16.3693466, 18.4018936, -34.7694321, 34.8269653
8: -18.2777863, 21.3003387, -18.2753601, 21.2743225, -39.5521088, 39.5756989
9: -16.6956253, 23.2807693, -16.6889496, 23.2246094, -38.5679550, 38.6160164
10: -24.2741718, 38.5076790, -24.2676220, 38.4553375, -61.7252045, 61.7765007
11: -24.7721653, 17.5933895, -24.7582703, 17.5872574, -42.3594208, 42.3516617
12: -28.6607933, 20.0945473, -28.6387253, 20.0866489, -46.8784790, 46.8585663
13: -32.9305801, 28.7842236, -32.9150009, 28.7533646, -61.6839447, 61.6992264
14: -23.4640770, 39.1930542, -23.4422207, 39.1590424, -59.9853859, 60.0000343
15: -18.9190884, 25.8431931, -18.9087563, 25.8355465, -44.7546349, 44.7519493
16: -32.7154617, 19.9305801, -32.7033997, 19.8535709, -52.5690308, 52.6339798
17: -17.7657623, 38.4748459, -17.7372398, 38.4291954, -55.2103043, 55.2394562
18: -25.8072262, 19.6136875, -25.7728252, 19.6084194, -45.4156456, 45.3865128
19: -26.4429226, 12.4706211, -26.3964481, 12.4722748, -38.9151993, 38.8670692
20: -21.1287460, 20.4225540, -21.0712147, 20.4199982, -41.5487442, 41.4937668
21: -25.7256508, 18.8584862, -25.6766758, 18.8587074, -44.5843582, 44.5351639
22: -22.1515656, 24.5082397, -22.0787582, 24.5096512, -46.6612167, 46.5869980
23: -21.7128353, 17.4895477, -21.6784973, 17.4871159, -39.1999512, 39.1680450
24: -32.1405792, 11.8878155, -32.1063995, 11.8854351, -44.0260162, 43.9942169
25: -18.1412735, 25.4110909, -18.0836792, 25.4101543, -43.5514297, 43.4947701
26: -29.2939758, 26.9402504, -29.2112007, 26.9396038, -56.2335815, 56.1514511
27: -32.1317406, 16.5331249, -32.0863724, 16.5288677, -47.7950592, 47.7330475
28: -21.5662956, 21.6866875, -21.5102711, 21.6847725, -43.2510681, 43.1969604
29: -23.7315674, 22.2201710, -23.6786995, 22.2185555, -45.9501228, 45.8988724
30: -29.6255112, 16.8443680, -29.6026478, 16.8363037, -45.9242706, 45.9040604
31: -26.3730526, 19.0614204, -26.3259716, 19.0626163, -45.4356689, 45.3873901
32: -42.2347488, 8.4590836, -42.2091866, 8.4555273, -47.5507812, 47.5323410
33: -72.3743057, -5.6538534, -72.3144531, -5.6591711, -61.2878113, 61.2216721
34: -56.5194397, -5.5228271, -56.4528885, -5.5205441, -43.6156769, 43.5409851
35: -50.1712837, 0.0059729, -50.1022797, 0.0052986, -48.2540283, 48.1885147
36: -47.8183517, 4.9039946, -47.7344742, 4.9051600, -52.0251541, 51.9408646
37: -83.6977844, -17.4775848, -83.6246643, -17.4801197, -58.4640884, 58.3348122
38: -58.7152901, 3.1885242, -58.5938797, 3.1830435, -61.3049774, 61.1927795
39: -78.9714508, -11.6332674, -78.9114532, -11.6351900, -65.2720947, 65.2070618
40: -67.6555710, -18.3454437, -67.6324463, -18.3506317, -41.1933098, 41.1455193
41: -55.1885986, -6.8632393, -55.1658020, -6.8640823, -42.2823029, 42.2425423
42: -33.9676590, 6.8092194, -33.9460983, 6.8037500, -37.7034225, 37.6927261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9705874, upper bound: 45.0061784
time: 27.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0061781, upper bound: 45.0061784
time: 26.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -27.9691200, 17.0104218, -28.1199722, 17.0752831, -44.2617493, 44.3492393
1: -13.5820265, 17.0090408, -13.6660194, 17.0639153, -30.6459427, 30.6750603
2: -13.9822178, 21.5831375, -14.0769081, 21.6364899, -35.3758011, 35.4228897
3: -12.8116322, 23.3392811, -12.8867579, 23.4081173, -36.2197495, 36.2260399
4: -21.4587784, 18.3883286, -21.5567284, 18.4339828, -39.8927612, 39.9450569
5: -11.9152832, 22.7658119, -11.9966049, 22.8097992, -34.7250824, 34.7624168
6: -50.6613846, -3.7021084, -50.7744522, -3.5876503, -40.5316734, 40.5055389
7: -16.2979431, 18.3682613, -16.3900871, 18.4087257, -34.7066689, 34.7583466
8: -18.1861095, 21.2268085, -18.3126106, 21.3207455, -39.5068550, 39.5394211
9: -16.5425339, 23.1900215, -16.6782455, 23.2739677, -38.4853668, 38.5145378
10: -24.1284313, 38.3871956, -24.2708454, 38.5224876, -61.6671906, 61.6614494
11: -24.7108784, 17.5023556, -24.7809696, 17.5695267, -42.2804031, 42.2833252
12: -28.5015488, 19.9771385, -28.6040611, 20.1188507, -46.7644577, 46.7045708
13: -32.7811241, 28.6835098, -32.8899231, 28.7876091, -61.5687332, 61.5734329
14: -23.1948166, 39.0997276, -23.4534683, 39.2319794, -59.8276062, 59.9160843
15: -18.8207054, 25.8080235, -18.9308128, 25.8752880, -44.6959915, 44.7388382
16: -32.6154938, 19.8450890, -32.7116661, 19.9024181, -52.5179138, 52.5567551
17: -17.5909367, 38.4088058, -17.7362728, 38.4971619, -55.1253853, 55.1722946
18: -25.7247810, 19.5529842, -25.8293381, 19.6011906, -45.3259735, 45.3823242
19: -26.3576965, 12.3921537, -26.4732323, 12.4772892, -38.8349838, 38.8653870
20: -21.0277958, 20.3393307, -21.1410904, 20.4305534, -41.4583511, 41.4804230
21: -25.6165619, 18.7639923, -25.7628651, 18.8733253, -44.4898872, 44.5268555
22: -22.0508575, 24.4437199, -22.1432133, 24.5188599, -46.5697174, 46.5869331
23: -21.6332474, 17.4115887, -21.7039127, 17.4813194, -39.1145668, 39.1155014
24: -32.0293808, 11.7757864, -32.1968842, 11.8647795, -43.8941612, 43.9726715
25: -18.0658092, 25.3376541, -18.1253338, 25.4013786, -43.4671860, 43.4629898
26: -29.1471329, 26.8178959, -29.2324028, 26.9277077, -56.0748405, 56.0503006
27: -32.0230331, 16.4336433, -32.1387863, 16.5139942, -47.6783638, 47.7214050
28: -21.4834747, 21.6074257, -21.5452232, 21.6796684, -43.1631432, 43.1526489
29: -23.6260357, 22.1478882, -23.7251110, 22.2127151, -45.8387527, 45.8730011
30: -29.5661793, 16.7574062, -29.6811104, 16.8429623, -45.8625679, 45.8894920
31: -26.2747192, 18.9644356, -26.4297848, 19.0697765, -45.3444977, 45.3942184
32: -42.1731300, 8.3947277, -42.2675934, 8.4904499, -47.5174904, 47.4967613
33: -72.2772369, -5.7968225, -72.4550476, -5.6385803, -61.2074814, 61.2320023
34: -56.4597702, -5.5970144, -56.5405922, -5.4933920, -43.5773849, 43.5577774
35: -50.1012001, -0.0652895, -50.2140999, 0.0365715, -48.1857758, 48.2287827
36: -47.7112961, 4.8121290, -47.8103828, 4.9376097, -51.9448090, 51.9184570
37: -83.5841141, -17.5975761, -83.7232513, -17.4789886, -58.3540115, 58.3341560
38: -58.6140251, 3.0825291, -58.7370834, 3.2203083, -61.2310410, 61.2127991
39: -78.8727493, -11.7193317, -79.0641174, -11.5984087, -65.2093353, 65.2928848
40: -67.5832672, -18.4493637, -67.7166290, -18.3527641, -41.1505928, 41.1443024
41: -55.1245918, -6.9630709, -55.2194023, -6.8467093, -42.2610817, 42.1662750
42: -33.9462624, 6.7287188, -33.9545135, 6.8032131, -37.6716003, 37.5927582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7621609, upper bound: 45.0236082
time: 98.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7979205, upper bound: 45.0236081
time: 41.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -27.9965019, 17.0495720, -28.1253700, 17.0877838, -44.3021927, 44.3819809
1: -13.6029787, 17.0546455, -13.6683102, 17.0794296, -30.6824074, 30.7229557
2: -14.0162554, 21.6400795, -14.0799971, 21.6558647, -35.4300766, 35.4735069
3: -12.8292789, 23.3910027, -12.8894081, 23.4259491, -36.2552261, 36.2804108
4: -21.4960022, 18.4397812, -21.5610924, 18.4512215, -39.9472237, 40.0008736
5: -11.9315147, 22.8078251, -11.9986067, 22.8241348, -34.7556496, 34.8064308
6: -50.6766739, -3.6758847, -50.7807655, -3.5823007, -40.5519180, 40.5629463
7: -16.3283386, 18.4344692, -16.3934460, 18.4327450, -34.7610855, 34.8279152
8: -18.2245140, 21.2873497, -18.3174057, 21.3414383, -39.5659523, 39.6047554
9: -16.6151085, 23.2349300, -16.7023125, 23.2788353, -38.5469894, 38.5816917
10: -24.1876106, 38.4450569, -24.2901917, 38.5299339, -61.7267303, 61.7376976
11: -24.7314529, 17.5239487, -24.7898312, 17.5735359, -42.3049889, 42.3137817
12: -28.6176605, 20.0455017, -28.6440010, 20.1247387, -46.8731384, 46.8132744
13: -32.8566437, 28.7377701, -32.9148178, 28.7952576, -61.6519012, 61.6525879
14: -23.3449554, 39.1464767, -23.5030479, 39.2346878, -59.9518089, 60.0134354
15: -18.8518867, 25.8225746, -18.9378777, 25.8852043, -44.7370911, 44.7604523
16: -32.6472588, 19.8785934, -32.7211151, 19.9094181, -52.5566788, 52.5997086
17: -17.6820641, 38.4339294, -17.7666473, 38.5006561, -55.2025604, 55.2280273
18: -25.7748337, 19.5675850, -25.8389912, 19.6055222, -45.3803558, 45.4065781
19: -26.3836594, 12.3962593, -26.4808712, 12.4784822, -38.8621407, 38.8771286
20: -21.0783806, 20.3600044, -21.1571045, 20.4324303, -41.5108109, 41.5171089
21: -25.6648026, 18.7826614, -25.7787018, 18.8755989, -44.5404015, 44.5613632
22: -22.0973053, 24.4621735, -22.1579704, 24.5216789, -46.6189842, 46.6201439
23: -21.6607285, 17.4200649, -21.7109241, 17.4837227, -39.1444511, 39.1309891
24: -32.0711098, 11.8124332, -32.2020493, 11.8767900, -43.9478989, 44.0144806
25: -18.0888710, 25.3492222, -18.1322765, 25.4041939, -43.4930649, 43.4814987
26: -29.2270870, 26.8539410, -29.2576294, 26.9301491, -56.1572342, 56.1115723
27: -32.0708008, 16.4589272, -32.1465263, 16.5226707, -47.7358971, 47.7303619
28: -21.5141582, 21.6138954, -21.5538063, 21.6811981, -43.1953583, 43.1677017
29: -23.6762695, 22.1740417, -23.7430534, 22.2148514, -45.8911209, 45.9170952
30: -29.5811844, 16.7782192, -29.6859646, 16.8468170, -45.8818550, 45.9199791
31: -26.3148613, 18.9883690, -26.4383774, 19.0773678, -45.3922272, 45.4267464
32: -42.2132721, 8.4292488, -42.2810669, 8.4948330, -47.5616760, 47.5730171
33: -72.3179779, -5.7225914, -72.4589920, -5.6145725, -61.2736053, 61.2989426
34: -56.4939842, -5.5719347, -56.5462875, -5.4852190, -43.6259766, 43.5904694
35: -50.1317291, -0.0396814, -50.2199326, 0.0448046, -48.2526855, 48.2602425
36: -47.7706528, 4.8326721, -47.8305779, 4.9403057, -52.0120087, 51.9707413
37: -83.6206360, -17.5588074, -83.7315979, -17.4661751, -58.3977890, 58.3791542
38: -58.6494789, 3.0946016, -58.7472229, 3.2231226, -61.2782288, 61.2614136
39: -78.9029617, -11.6942902, -79.0694351, -11.5903168, -65.2463531, 65.3136520
40: -67.6149139, -18.3874626, -67.7218781, -18.3313446, -41.1661797, 41.2027702
41: -55.1511459, -6.9373074, -55.2248116, -6.8390026, -42.2705345, 42.2421951
42: -33.9462814, 6.7592897, -33.9575195, 6.8078194, -37.6792374, 37.6563416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9143340, upper bound: 45.0258380
time: 33.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9501261, upper bound: 45.0258380
time: 52.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.0474854, 17.0399628, -28.1440620, 17.0780907, -44.3455582, 44.4110489
1: -13.6285248, 17.0343437, -13.6809826, 17.0655499, -30.6940746, 30.7153263
2: -14.0251675, 21.6024246, -14.0908871, 21.6381721, -35.4231033, 35.4636307
3: -12.8623362, 23.3725929, -12.9035816, 23.4113197, -36.2736549, 36.2761765
4: -21.5118065, 18.4050083, -21.5733261, 18.4367104, -39.9485168, 39.9783325
5: -11.9630699, 22.7980728, -12.0123863, 22.8128700, -34.7759399, 34.8104591
6: -50.6819038, -3.6425381, -50.7764816, -3.5682139, -40.5727959, 40.5643158
7: -16.3371315, 18.3913956, -16.4026413, 18.4116898, -34.7488213, 34.7940369
8: -18.2394276, 21.2397995, -18.3293381, 21.3233719, -39.5627975, 39.5691376
9: -16.6230774, 23.2358761, -16.7042847, 23.2772694, -38.5629578, 38.5870819
10: -24.2150116, 38.4498444, -24.2992401, 38.5288696, -61.7526398, 61.7531204
11: -24.7515850, 17.5717888, -24.7837639, 17.5919609, -42.3435440, 42.3555527
12: -28.5446472, 20.0262222, -28.6174202, 20.1277828, -46.8127556, 46.7668762
13: -32.8550644, 28.7299557, -32.9136314, 28.7936440, -61.6487083, 61.6435852
14: -23.3139286, 39.1462708, -23.4913750, 39.2349892, -59.9415436, 60.0021248
15: -18.8879032, 25.8286400, -18.9523640, 25.8788376, -44.7667389, 44.7810059
16: -32.6836929, 19.8970490, -32.7321167, 19.9070396, -52.5907326, 52.6291656
17: -17.6746521, 38.4497223, -17.7626381, 38.4998169, -55.2073746, 55.2401428
18: -25.7572174, 19.5990868, -25.8333721, 19.6157284, -45.3729477, 45.4324570
19: -26.4169502, 12.4665222, -26.4778004, 12.5022507, -38.9192009, 38.9443207
20: -21.0781479, 20.4019127, -21.1452961, 20.4513474, -41.5294952, 41.5472107
21: -25.6773949, 18.8398209, -25.7681160, 18.8987389, -44.5761337, 44.6079369
22: -22.1051292, 24.4897690, -22.1478519, 24.5337944, -46.6389236, 46.6376190
23: -21.6853428, 17.4810638, -21.7077293, 17.5040588, -39.1893997, 39.1887932
24: -32.0988846, 11.8511581, -32.2007675, 11.8898935, -43.9887772, 44.0519257
25: -18.1182022, 25.3995132, -18.1306629, 25.4216805, -43.5398827, 43.5301743
26: -29.2140217, 26.9042206, -29.2381439, 26.9566994, -56.1707230, 56.1423645
27: -32.0839767, 16.5078621, -32.1426086, 16.5386429, -47.7636833, 47.7943382
28: -21.5355930, 21.6802025, -21.5495262, 21.7039833, -43.2395782, 43.2297287
29: -23.6813583, 22.1940022, -23.7287197, 22.2276917, -45.9090500, 45.9227219
30: -29.6105194, 16.8235397, -29.6842899, 16.8643951, -45.9292603, 45.9579773
31: -26.3329334, 19.0374908, -26.4365292, 19.0939693, -45.4269028, 45.4740219
32: -42.1945915, 8.4245625, -42.2704315, 8.4990587, -47.5539856, 47.5317078
33: -72.3335800, -5.7280273, -72.4607315, -5.6162319, -61.2887726, 61.3051758
34: -56.4852524, -5.5478992, -56.5434647, -5.4776983, -43.6199875, 43.6090240
35: -50.1407433, -0.0196600, -50.2182198, 0.0516577, -48.2425537, 48.2795982
36: -47.7590027, 4.8834543, -47.8149643, 4.9615450, -52.0168457, 51.9913788
37: -83.6612701, -17.5163116, -83.7293549, -17.4520054, -58.4606781, 58.4052544
38: -58.6798630, 3.1764498, -58.7424126, 3.2504787, -61.3285675, 61.3027496
39: -78.9412613, -11.6582880, -79.0704498, -11.5783186, -65.3001938, 65.3522110
40: -67.6239166, -18.4073582, -67.7210693, -18.3389397, -41.2093163, 41.1830215
41: -55.1620064, -6.8889885, -55.2221069, -6.8225155, -42.3237038, 42.2326965
42: -33.9676437, 6.7786646, -33.9567528, 6.8191538, -37.7177963, 37.6439934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.7975682, upper bound: 45.0236104
time: 28.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.8332051, upper bound: 45.0236104
time: 28.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.0749054, 17.0791245, -28.1494598, 17.0905857, -44.3860474, 44.4437904
1: -13.6494713, 17.0799427, -13.6832705, 17.0810890, -30.7305603, 30.7632141
2: -14.0591946, 21.6593704, -14.0939646, 21.6575470, -35.4773827, 35.5142479
3: -12.8799877, 23.4243202, -12.9062338, 23.4291496, -36.3091354, 36.3305550
4: -21.5490265, 18.4564552, -21.5776939, 18.4539337, -40.0029602, 40.0341492
5: -11.9793091, 22.8401432, -12.0143833, 22.8272095, -34.8065186, 34.8545265
6: -50.6971855, -3.6163244, -50.7827682, -3.5628452, -40.5930252, 40.6217270
7: -16.3675404, 18.4576187, -16.4060211, 18.4357033, -34.8032455, 34.8636398
8: -18.2777863, 21.3003387, -18.3341312, 21.3440666, -39.6218529, 39.6344681
9: -16.6956253, 23.2807693, -16.7283516, 23.2821274, -38.6245651, 38.6542168
10: -24.2741718, 38.5076790, -24.3186111, 38.5363045, -61.8121490, 61.8293610
11: -24.7721653, 17.5933895, -24.7926159, 17.5959949, -42.3681602, 42.3860054
12: -28.6607933, 20.0945473, -28.6573696, 20.1336441, -46.9214478, 46.8755913
13: -32.9305801, 28.7842236, -32.9385262, 28.8012676, -61.7318497, 61.7227478
14: -23.4640770, 39.1930542, -23.5409145, 39.2377243, -60.0657883, 60.0995026
15: -18.9190884, 25.8431931, -18.9593849, 25.8887558, -44.8078461, 44.8025780
16: -32.7154617, 19.9305801, -32.7415428, 19.9140282, -52.6294899, 52.6721230
17: -17.7657623, 38.4748459, -17.7930145, 38.5033302, -55.2845230, 55.2958488
18: -25.8072262, 19.6136875, -25.8430157, 19.6200562, -45.4272842, 45.4567032
19: -26.4429226, 12.4706211, -26.4854603, 12.5034409, -38.9463654, 38.9560814
20: -21.1287460, 20.4225540, -21.1613274, 20.4532261, -41.5819702, 41.5838814
21: -25.7256508, 18.8584862, -25.7839508, 18.9010124, -44.6266632, 44.6424370
22: -22.1515656, 24.5082397, -22.1626244, 24.5366516, -46.6882172, 46.6708641
23: -21.7128353, 17.4895477, -21.7147713, 17.5064850, -39.2193222, 39.2043190
24: -32.1405792, 11.8878155, -32.2059441, 11.9018993, -44.0424805, 44.0937576
25: -18.1412735, 25.4110909, -18.1376057, 25.4244423, -43.5657158, 43.5486984
26: -29.2939758, 26.9402504, -29.2633514, 26.9591827, -56.2531586, 56.2036018
27: -32.1317406, 16.5331249, -32.1503754, 16.5473061, -47.8211899, 47.8033218
28: -21.5662956, 21.6866875, -21.5581169, 21.7054787, -43.2717743, 43.2448044
29: -23.7315674, 22.2201710, -23.7466431, 22.2298355, -45.9614029, 45.9668121
30: -29.6255112, 16.8443680, -29.6891365, 16.8682728, -45.9485397, 45.9884605
31: -26.3730526, 19.0614204, -26.4451180, 19.1015816, -45.4746323, 45.5065384
32: -42.2347488, 8.4590836, -42.2838898, 8.5034304, -47.5981941, 47.6079254
33: -72.3743057, -5.6538534, -72.4646759, -5.5922899, -61.3549194, 61.3720627
34: -56.5194397, -5.5228271, -56.5491753, -5.4695425, -43.6685638, 43.6416969
35: -50.1712837, 0.0059729, -50.2240562, 0.0598764, -48.3094940, 48.3110275
36: -47.8183517, 4.9039946, -47.8351746, 4.9642019, -52.0841217, 52.0435791
37: -83.6977844, -17.4775848, -83.7377014, -17.4391727, -58.5044479, 58.4502335
38: -58.7152901, 3.1885242, -58.7525673, 3.2532377, -61.3757782, 61.3514252
39: -78.9714508, -11.6332674, -79.0757675, -11.5702534, -65.3372116, 65.3729858
40: -67.6555710, -18.3454437, -67.7263031, -18.3174973, -41.2249718, 41.2414894
41: -55.1885986, -6.8632393, -55.2275467, -6.8147793, -42.3331375, 42.3085938
42: -33.9676590, 6.8092194, -33.9597549, 6.8237696, -37.7254486, 37.7075844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 765

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9498142, upper bound: 45.0258410
time: 32.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -44.9854897, upper bound: 45.0258412
time: 25.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.1103592, 17.0798264, -27.9597702, 16.8933964, -44.2152939, 44.2632942
1: -13.6605473, 17.0683632, -13.5709047, 16.8850937, -30.5456409, 30.6392670
2: -14.0666332, 21.6303406, -13.9536343, 21.4324932, -35.2564697, 35.3576813
3: -12.8799305, 23.4077568, -12.7955627, 23.2266407, -36.1065712, 36.2033195
4: -21.5485706, 18.4136238, -21.4151764, 18.2612114, -39.8097839, 39.8288002
5: -11.9879608, 22.8078747, -11.9047632, 22.6420784, -34.6300392, 34.7126389
6: -50.7625351, -3.5947161, -50.6017303, -3.6916771, -40.4935150, 40.4314880
7: -16.3784924, 18.4217186, -16.2823143, 18.2218246, -34.6003189, 34.7040329
8: -18.3029556, 21.3038025, -18.1781654, 21.1075363, -39.4104919, 39.4819679
9: -16.6467190, 23.2985325, -16.4630470, 23.0678711, -38.3792114, 38.4267921
10: -24.2242222, 38.5168304, -24.0119991, 38.2371902, -61.4742279, 61.5533829
11: -24.7601776, 17.5657825, -24.6295357, 17.4643230, -42.2245026, 42.1953201
12: -28.5647449, 20.1096802, -28.3220482, 19.9070301, -46.6122665, 46.5785141
13: -32.8774796, 28.7943363, -32.7172279, 28.6222610, -61.4997406, 61.5115662
14: -23.4064274, 39.2308998, -23.0653191, 39.0029488, -59.7984734, 59.6946030
15: -18.9275436, 25.8598061, -18.7868004, 25.7549343, -44.6824799, 44.6466064
16: -32.6972580, 19.9302902, -32.5661964, 19.7081032, -52.4053612, 52.4964867
17: -17.7206726, 38.4947777, -17.4919434, 38.3145523, -55.0645981, 55.0350418
18: -25.8425922, 19.5846500, -25.6214428, 19.5031204, -45.3457108, 45.2060928
19: -26.4865456, 12.4712515, -26.2647476, 12.4051266, -38.8916702, 38.7360001
20: -21.1454468, 20.4261055, -20.9005795, 20.3333473, -41.4787941, 41.3266830
21: -25.7629299, 18.8677197, -25.4926357, 18.7679863, -44.5309143, 44.3603554
22: -22.1651344, 24.5096264, -21.9134903, 24.4465523, -46.6116867, 46.4231186
23: -21.7143116, 17.4786243, -21.5660534, 17.3978748, -39.1121864, 39.0446777
24: -32.2083244, 11.8488884, -31.9540234, 11.7420330, -43.9503555, 43.8029099
25: -18.1507912, 25.3948307, -17.9651031, 25.3245316, -43.4753227, 43.3599319
26: -29.2730503, 26.9142189, -28.9735947, 26.7811871, -56.0542374, 55.8878136
27: -32.1405640, 16.5000725, -31.9177551, 16.4009094, -47.7409897, 47.5587769
28: -21.5603752, 21.6727905, -21.3784008, 21.5987701, -43.1591454, 43.0511932
29: -23.7289066, 22.2073746, -23.4951420, 22.1187706, -45.8476791, 45.7025146
30: -29.6697578, 16.8400536, -29.4868984, 16.7301788, -45.8537560, 45.7752113
31: -26.4430904, 19.0644112, -26.1830864, 18.9830875, -45.4261780, 45.2474976
32: -42.2309647, 8.4849710, -42.0690346, 8.3857641, -47.4454269, 47.3950500
33: -72.4566498, -5.6607780, -72.1727600, -5.8105297, -61.2129440, 61.0777359
34: -56.5524902, -5.5043898, -56.3555756, -5.5771923, -43.5871582, 43.4443283
35: -50.2248955, 0.0252428, -49.9960938, -0.0411901, -48.2357407, 48.0458946
36: -47.8204193, 4.9295244, -47.5647049, 4.8602476, -51.9703751, 51.7818222
37: -83.7549820, -17.4923782, -83.4710236, -17.5939064, -58.3456650, 58.1596069
38: -58.7710876, 3.2116461, -58.4246979, 3.1207943, -61.2877655, 61.0304565
39: -79.0529861, -11.6074829, -78.7652435, -11.6801443, -65.3136749, 65.0927048
40: -67.7111816, -18.3654232, -67.5357666, -18.4681282, -41.1151886, 41.0526237
41: -55.2140083, -6.8565350, -55.0741005, -6.9525452, -42.1530380, 42.1523361
42: -33.9501801, 6.7950172, -33.8757515, 6.6861238, -37.5480270, 37.5727768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0138792, upper bound: 44.6262040
time: 54.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0219138, upper bound: 44.6910279
time: 56.23 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.2362480, 17.1188316, -27.9617882, 16.9010239, -44.3491287, 44.3033714
1: -13.7147083, 17.0826416, -13.5716372, 16.8878498, -30.6025581, 30.6542778
2: -14.1266832, 21.6558189, -13.9543915, 21.4378166, -35.3238754, 35.3846321
3: -12.9178238, 23.4266167, -12.7957439, 23.2284203, -36.1462440, 36.2223587
4: -21.6003571, 18.4343338, -21.4152184, 18.2625599, -39.8629150, 39.8495522
5: -12.0478115, 22.8423576, -11.9051056, 22.6485596, -34.6963730, 34.7474632
6: -50.7942085, -3.5232606, -50.6066971, -3.6915202, -40.5260048, 40.5106583
7: -16.4352589, 18.4415855, -16.2828312, 18.2253189, -34.6605759, 34.7244186
8: -18.3976860, 21.3228168, -18.1789379, 21.1088676, -39.5065536, 39.5017548
9: -16.7013416, 23.3282356, -16.4631252, 23.0735130, -38.4397583, 38.4576721
10: -24.3100681, 38.5744781, -24.0128708, 38.2485352, -61.5778046, 61.6145287
11: -24.7923679, 17.5882168, -24.6318951, 17.4639034, -42.2562714, 42.2201118
12: -28.5800056, 20.1577377, -28.3228340, 19.9083080, -46.6359863, 46.6248474
13: -32.9124298, 28.8328571, -32.7178726, 28.6251564, -61.5375862, 61.5507278
14: -23.5373478, 39.2649841, -23.0670433, 39.0104599, -59.9374390, 59.7336273
15: -18.9700623, 25.8978958, -18.7879601, 25.7563572, -44.7264175, 44.6858559
16: -32.7707634, 19.9809265, -32.5675621, 19.7180691, -52.4888306, 52.5484886
17: -17.8062782, 38.5436440, -17.4928436, 38.3248329, -55.1664009, 55.0885429
18: -25.8768463, 19.5999985, -25.6223984, 19.5015297, -45.3783760, 45.2223969
19: -26.5148029, 12.5009584, -26.2686920, 12.4049625, -38.9197655, 38.7696495
20: -21.1882210, 20.4776993, -20.9082375, 20.3335152, -41.5217361, 41.3859367
21: -25.8039913, 18.9112930, -25.4981747, 18.7676201, -44.5716095, 44.4094696
22: -22.2157593, 24.5625324, -21.9231853, 24.4470673, -46.6628265, 46.4857178
23: -21.7389603, 17.4886093, -21.5683498, 17.3944378, -39.1334000, 39.0569611
24: -32.2434845, 11.8773994, -31.9574814, 11.7418804, -43.9853668, 43.8348808
25: -18.1853256, 25.4373531, -17.9692535, 25.3246708, -43.5099945, 43.4066086
26: -29.3109055, 26.9609776, -28.9785366, 26.7818832, -56.0927887, 55.9395142
27: -32.1898994, 16.5550880, -31.9257851, 16.4008408, -47.7908630, 47.6294022
28: -21.5950546, 21.7317314, -21.3841553, 21.5992107, -43.1942673, 43.1158867
29: -23.7686768, 22.2422199, -23.5021000, 22.1187687, -45.8874435, 45.7443199
30: -29.6954689, 16.8837280, -29.4885101, 16.7301598, -45.8847504, 45.8237228
31: -26.4817734, 19.1072464, -26.1869030, 18.9831181, -45.4648895, 45.2941513
32: -42.2797928, 8.5583954, -42.0781937, 8.3862648, -47.4915924, 47.4761543
33: -72.5195312, -5.5543690, -72.1858444, -5.8094425, -61.2788696, 61.1954041
34: -56.6112480, -5.4155159, -56.3675613, -5.5768728, -43.6456375, 43.5447083
35: -50.2879448, 0.1227961, -50.0088692, -0.0407019, -48.2996445, 48.1564713
36: -47.8915939, 5.0383425, -47.5796089, 4.8607635, -52.0421829, 51.9057617
37: -83.7948914, -17.4475861, -83.4777145, -17.5935287, -58.3880615, 58.2208099
38: -58.8612556, 3.3330050, -58.4429092, 3.1217222, -61.3804321, 61.1764755
39: -79.1303177, -11.5184679, -78.7802734, -11.6793556, -65.3920746, 65.2026138
40: -67.7418823, -18.3141575, -67.5405273, -18.4674168, -41.1478043, 41.1123009
41: -55.2460976, -6.7939472, -55.0788078, -6.9525595, -42.1873932, 42.2220917
42: -33.9626465, 6.8312254, -33.8776550, 6.6860580, -37.5580940, 37.6223373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0138792, upper bound: 44.6619637
time: 45.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0219138, upper bound: 44.7264867
time: 153.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -28.1157722, 17.0923042, -27.9845848, 16.9315834, -44.2478104, 44.3027115
1: -13.6628113, 17.0838757, -13.5917377, 16.9302788, -30.5930901, 30.6756134
2: -14.0697174, 21.6496811, -13.9876270, 21.4887772, -35.3063965, 35.4118042
3: -12.8825722, 23.4255562, -12.8107424, 23.2761536, -36.1587257, 36.2362976
4: -21.5529499, 18.4308586, -21.4522858, 18.3123360, -39.8652878, 39.8831444
5: -11.9899349, 22.8218117, -11.9187307, 22.6825104, -34.6724472, 34.7405434
6: -50.7688293, -3.5893397, -50.6163864, -3.6653867, -40.5509682, 40.4510803
7: -16.3818626, 18.4457130, -16.3112259, 18.2845955, -34.6664581, 34.7569389
8: -18.3077431, 21.3244705, -18.2164650, 21.1676331, -39.4753761, 39.5409355
9: -16.6707649, 23.3033810, -16.5350876, 23.1125679, -38.4459953, 38.4877510
10: -24.2435722, 38.5242577, -24.0709038, 38.2941742, -61.5499268, 61.6124687
11: -24.7690506, 17.5698204, -24.6497555, 17.4860592, -42.2551117, 42.2195740
12: -28.6046734, 20.1155510, -28.4377518, 19.9752693, -46.7208519, 46.6867447
13: -32.9023399, 28.8019447, -32.7923737, 28.6762161, -61.5785561, 61.5943184
14: -23.4559498, 39.2336197, -23.2146187, 39.0494232, -59.8954926, 59.8179741
15: -18.9345627, 25.8697891, -18.8183098, 25.7690296, -44.7035904, 44.6880989
16: -32.7067528, 19.9372635, -32.5973663, 19.7419376, -52.4486923, 52.5346298
17: -17.7510166, 38.4982872, -17.5824356, 38.3396530, -55.1202240, 55.1115685
18: -25.8522797, 19.5889816, -25.6712818, 19.5174351, -45.3697128, 45.2602615
19: -26.4941788, 12.4724426, -26.2903175, 12.4092045, -38.9033813, 38.7627602
20: -21.1614742, 20.4279842, -20.9509010, 20.3537903, -41.5152664, 41.3788834
21: -25.7787495, 18.8699799, -25.5403748, 18.7865486, -44.5653000, 44.4103546
22: -22.1799278, 24.5124664, -21.9588547, 24.4649677, -46.6448975, 46.4713211
23: -21.7213211, 17.4810524, -21.5933075, 17.4063339, -39.1276550, 39.0743599
24: -32.2134933, 11.8609180, -31.9956856, 11.7783585, -43.9918518, 43.8566055
25: -18.1577263, 25.3976269, -17.9880161, 25.3355064, -43.4932327, 43.3856430
26: -29.2981625, 26.9166927, -29.0529251, 26.8115520, -56.1097145, 55.9696198
27: -32.1482735, 16.5087566, -31.9648800, 16.4260883, -47.7497978, 47.6155586
28: -21.5689335, 21.6742840, -21.4085350, 21.6051617, -43.1740952, 43.0828171
29: -23.7461452, 22.2094994, -23.5435429, 22.1432362, -45.8893814, 45.7530441
30: -29.6746254, 16.8439217, -29.5017128, 16.7509193, -45.8841629, 45.7942123
31: -26.4517155, 19.0720081, -26.2230854, 19.0069199, -45.4586334, 45.2950935
32: -42.2443886, 8.4893265, -42.1084480, 8.4202070, -47.5216064, 47.4384613
33: -72.4606018, -5.6368008, -72.2133102, -5.7369394, -61.2791367, 61.1436768
34: -56.5581741, -5.4962234, -56.3893661, -5.5520716, -43.6193199, 43.4924164
35: -50.2307396, 0.0334902, -50.0263329, -0.0157557, -48.2669525, 48.1124649
36: -47.8405952, 4.9322491, -47.6229782, 4.8801699, -52.0213699, 51.8470535
37: -83.7633591, -17.4795494, -83.5072632, -17.5554161, -58.3899918, 58.2034607
38: -58.7812080, 3.2144165, -58.4599304, 3.1332598, -61.3368378, 61.0773773
39: -79.0583649, -11.5994167, -78.7953644, -11.6554737, -65.3340988, 65.1295624
40: -67.7164307, -18.3440933, -67.5672150, -18.4068069, -41.1736450, 41.0677071
41: -55.2193985, -6.8487978, -55.0992050, -6.9271431, -42.2279282, 42.1613312
42: -33.9531555, 6.7996483, -33.8754463, 6.7167196, -37.6115837, 37.5800285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0161152, upper bound: 44.7559468
time: 56.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0241432, upper bound: 44.8209232
time: 48.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -28.2416573, 17.1313076, -27.9865913, 16.9391937, -44.3816376, 44.3427887
1: -13.7169847, 17.0981560, -13.5924397, 16.9330368, -30.6500206, 30.6905956
2: -14.1297684, 21.6751671, -13.9883680, 21.4941101, -35.3737793, 35.4387932
3: -12.9204693, 23.4444160, -12.8108921, 23.2779255, -36.1983948, 36.2553101
4: -21.6047440, 18.4515572, -21.4523125, 18.3137093, -39.9184532, 39.9038696
5: -12.0497932, 22.8563042, -11.9190769, 22.6889801, -34.7387733, 34.7753830
6: -50.8005104, -3.5179019, -50.6213531, -3.6652317, -40.5834808, 40.5302353
7: -16.4386330, 18.4655914, -16.3117313, 18.2880859, -34.7267189, 34.7773209
8: -18.4024696, 21.3435020, -18.2172737, 21.1689453, -39.5714149, 39.5607758
9: -16.7253799, 23.3330936, -16.5351734, 23.1181984, -38.5065498, 38.5186386
10: -24.3294430, 38.5819168, -24.0717773, 38.3055649, -61.6534424, 61.6736298
11: -24.8012161, 17.5922718, -24.6521149, 17.4856396, -42.2868576, 42.2443848
12: -28.6199417, 20.1636124, -28.4385605, 19.9765339, -46.7445946, 46.7330780
13: -32.9372787, 28.8405037, -32.7930222, 28.6791210, -61.6164017, 61.6335258
14: -23.5868645, 39.2677078, -23.2163544, 39.0569077, -60.0344315, 59.8569946
15: -18.9771080, 25.9078674, -18.8194580, 25.7704659, -44.7475739, 44.7273254
16: -32.7802505, 19.9879093, -32.5987396, 19.7518806, -52.5321312, 52.5866470
17: -17.8366165, 38.5471420, -17.5833054, 38.3499146, -55.2219849, 55.1650734
18: -25.8865242, 19.6043282, -25.6722412, 19.5158443, -45.4023666, 45.2765694
19: -26.5224476, 12.5021820, -26.2942543, 12.4090385, -38.9314880, 38.7964363
20: -21.2042446, 20.4795628, -20.9585686, 20.3539505, -41.5581970, 41.4381332
21: -25.8198128, 18.9135551, -25.5459042, 18.7861977, -44.6060104, 44.4594574
22: -22.2305489, 24.5653496, -21.9685326, 24.4654732, -46.6960220, 46.5338821
23: -21.7459717, 17.4910355, -21.5956059, 17.4029007, -39.1488724, 39.0866394
24: -32.2486496, 11.8894119, -31.9991436, 11.7782326, -44.0268822, 43.8885574
25: -18.1922493, 25.4401665, -17.9921513, 25.3356094, -43.5278587, 43.4323196
26: -29.3360348, 26.9634590, -29.0578365, 26.8122540, -56.1482887, 56.0212936
27: -32.1976280, 16.5637703, -31.9728947, 16.4260292, -47.7997093, 47.6861572
28: -21.6036358, 21.7332325, -21.4142933, 21.6056347, -43.2092705, 43.1475258
29: -23.7859344, 22.2443428, -23.5505085, 22.1432343, -45.9291687, 45.7948532
30: -29.7003212, 16.8876133, -29.5033512, 16.7508965, -45.9151611, 45.8427391
31: -26.4903793, 19.1148453, -26.2269135, 19.0069389, -45.4973183, 45.3417587
32: -42.2932472, 8.5627689, -42.1175919, 8.4207544, -47.5677757, 47.5195847
33: -72.5234909, -5.5304060, -72.2264099, -5.7358923, -61.3450928, 61.2613907
34: -56.6169319, -5.4073343, -56.4013023, -5.5517340, -43.6777725, 43.5928078
35: -50.2937469, 0.1310177, -50.0391121, -0.0153179, -48.3308411, 48.2230797
36: -47.9117775, 5.0409908, -47.6378670, 4.8806677, -52.0932541, 51.9709854
37: -83.8032761, -17.4347763, -83.5139542, -17.5550308, -58.4324341, 58.2646332
38: -58.8714027, 3.3358154, -58.4780922, 3.1341629, -61.4295807, 61.2234039
39: -79.1356812, -11.5103741, -78.8103333, -11.6546993, -65.4124908, 65.2394257
40: -67.7471619, -18.2928276, -67.5719757, -18.4061012, -41.2062531, 41.1273880
41: -55.2515182, -6.7862129, -55.1039352, -6.9271555, -42.2623291, 42.2311058
42: -33.9656372, 6.8358469, -33.8773613, 6.7166157, -37.6216278, 37.6295929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 902

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0238539, upper bound: 44.7329630
time: 71.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0251168, upper bound: 44.8574554
time: 46.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -28.1344643, 17.0826263, -28.0381813, 16.9229431, -44.2770996, 44.3471718
1: -13.6755075, 17.0700111, -13.6174078, 16.9104080, -30.5859146, 30.6874199
2: -14.0805826, 21.6320076, -13.9966154, 21.4517727, -35.2972336, 35.4049759
3: -12.8967457, 23.4109459, -12.8462791, 23.2599525, -36.1567001, 36.2572250
4: -21.5651760, 18.4163170, -21.4681702, 18.2779160, -39.8430939, 39.8844872
5: -12.0037308, 22.8109188, -11.9525461, 22.6743393, -34.6780701, 34.7634659
6: -50.7645378, -3.5752802, -50.6222725, -3.6320591, -40.5523338, 40.4726334
7: -16.3910580, 18.4246788, -16.3215237, 18.2449684, -34.6360245, 34.7462006
8: -18.3196697, 21.3064251, -18.2314491, 21.1205444, -39.4402161, 39.5378723
9: -16.6727715, 23.3018188, -16.5436211, 23.1137314, -38.4517593, 38.5043793
10: -24.2526131, 38.5232391, -24.0985889, 38.2997818, -61.5659103, 61.6388359
11: -24.7629814, 17.5882072, -24.6702347, 17.5337448, -42.2967262, 42.2584419
12: -28.5780945, 20.1186180, -28.3651733, 19.9561062, -46.6745720, 46.6268349
13: -32.9011497, 28.8003216, -32.7911758, 28.6687489, -61.5699005, 61.5914993
14: -23.4443054, 39.2339287, -23.1844521, 39.0495148, -59.8845291, 59.8086014
15: -18.9490242, 25.8633633, -18.8540649, 25.7755661, -44.7245903, 44.7174301
16: -32.7176895, 19.9348888, -32.6344337, 19.7600822, -52.4777718, 52.5693207
17: -17.7470436, 38.4974518, -17.5756950, 38.3554802, -55.1324043, 55.1170769
18: -25.8466110, 19.5992126, -25.6539116, 19.5492821, -45.3958931, 45.2531242
19: -26.4911098, 12.4962025, -26.3240013, 12.4794960, -38.9706039, 38.8202057
20: -21.1496639, 20.4468994, -20.9509335, 20.3959312, -41.5455933, 41.3978348
21: -25.7681942, 18.8931351, -25.5534668, 18.8437691, -44.6119614, 44.4466019
22: -22.1697693, 24.5245819, -21.9677277, 24.4925976, -46.6623688, 46.4923096
23: -21.7181129, 17.5013733, -21.6181660, 17.4673367, -39.1854477, 39.1195374
24: -32.2121887, 11.8739901, -32.0235367, 11.8174114, -44.0296021, 43.8975258
25: -18.1560783, 25.4151268, -18.0174809, 25.3864403, -43.5425186, 43.4326096
26: -29.2787743, 26.9432163, -29.0404758, 26.8675232, -56.1462975, 55.9836922
27: -32.1443863, 16.5247307, -31.9786816, 16.4751492, -47.8139458, 47.6440582
28: -21.5646553, 21.6970654, -21.4305058, 21.6715546, -43.2362099, 43.1275711
29: -23.7325039, 22.2223778, -23.5504704, 22.1648846, -45.8973885, 45.7728500
30: -29.6729221, 16.8615150, -29.5312614, 16.7963352, -45.9222565, 45.8419151
31: -26.4498158, 19.0886173, -26.2413101, 19.0561695, -45.5059853, 45.3299255
32: -42.2338028, 8.4935579, -42.0905304, 8.4155922, -47.4803276, 47.4314766
33: -72.4623184, -5.6384430, -72.2290726, -5.7417336, -61.2861252, 61.1589890
34: -56.5553551, -5.4887028, -56.3810883, -5.5281239, -43.6384354, 43.4869118
35: -50.2290421, 0.0403328, -50.0356293, 0.0044975, -48.2865524, 48.1026840
36: -47.8250046, 4.9534607, -47.6124001, 4.9315643, -52.0432129, 51.8537827
37: -83.7610626, -17.4654140, -83.5481415, -17.5126495, -58.4167633, 58.2662048
38: -58.7764397, 3.2417889, -58.4905777, 3.2147179, -61.3777466, 61.1280136
39: -79.0593033, -11.5874405, -78.8336716, -11.6190577, -65.3730011, 65.1834335
40: -67.7156372, -18.3515873, -67.5764236, -18.4260826, -41.1539574, 41.1113625
41: -55.2167168, -6.8323383, -55.1115189, -6.8784580, -42.2194519, 42.2149582
42: -33.9523926, 6.8109837, -33.8971405, 6.7360811, -37.5992622, 37.6189880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=247, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0138799, upper bound: 44.6613904
time: 79.23 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0219157, upper bound: 44.7262997
time: 78.95 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -28.2603569, 17.1216221, -28.0402107, 16.9305630, -44.4109344, 44.3872528
1: -13.7296829, 17.0842972, -13.6181459, 16.9131470, -30.6428299, 30.7024422
2: -14.1406479, 21.6574879, -13.9973602, 21.4571190, -35.3646240, 35.4319420
3: -12.9346504, 23.4298115, -12.8464546, 23.2617416, -36.1963921, 36.2762680
4: -21.6169796, 18.4370441, -21.4682446, 18.2792683, -39.8962479, 39.9052887
5: -12.0636024, 22.8454132, -11.9529057, 22.6808014, -34.7444038, 34.7983170
6: -50.7962036, -3.5038195, -50.6272202, -3.6319132, -40.5847893, 40.5517921
7: -16.4478283, 18.4445496, -16.3220139, 18.2484665, -34.6962967, 34.7665634
8: -18.4144478, 21.3254585, -18.2322273, 21.1218719, -39.5363197, 39.5576859
9: -16.7273903, 23.3315125, -16.5437050, 23.1193581, -38.5123138, 38.5352821
10: -24.3385067, 38.5808792, -24.0994606, 38.3111801, -61.6694794, 61.6999474
11: -24.7951813, 17.6106834, -24.6725922, 17.5333328, -42.3285141, 42.2832756
12: -28.5933762, 20.1666756, -28.3659515, 19.9573689, -46.6983109, 46.6731834
13: -32.9361458, 28.8389091, -32.7918434, 28.6716385, -61.6077843, 61.6307526
14: -23.5752430, 39.2679901, -23.1861992, 39.0570412, -60.0234985, 59.8476372
15: -18.9915695, 25.9014435, -18.8552208, 25.7769756, -44.7685471, 44.7566643
16: -32.7911835, 19.9855404, -32.6358032, 19.7700272, -52.5612106, 52.6213455
17: -17.8326607, 38.5462837, -17.5765648, 38.3657494, -55.2342453, 55.1705818
18: -25.8808956, 19.6145535, -25.6548920, 19.5476570, -45.4285507, 45.2694473
19: -26.5193748, 12.5259476, -26.3279381, 12.4793444, -38.9987183, 38.8538857
20: -21.1924324, 20.4984818, -20.9585915, 20.3960876, -41.5885201, 41.4570732
21: -25.8092346, 18.9367142, -25.5589886, 18.8434181, -44.6526527, 44.4957047
22: -22.2203884, 24.5774803, -21.9774094, 24.4931030, -46.7134933, 46.5548897
23: -21.7427711, 17.5113754, -21.6204720, 17.4639206, -39.2066917, 39.1318474
24: -32.2473602, 11.9025011, -32.0269775, 11.8172922, -44.0646515, 43.9294777
25: -18.1906242, 25.4576378, -18.0216160, 25.3865261, -43.5771484, 43.4792557
26: -29.3166351, 26.9899807, -29.0453968, 26.8682022, -56.1848373, 56.0353775
27: -32.1937256, 16.5797367, -31.9867058, 16.4750862, -47.8638306, 47.7147064
28: -21.5993366, 21.7560310, -21.4362679, 21.6719990, -43.2713356, 43.1922989
29: -23.7722816, 22.2572289, -23.5574284, 22.1648903, -45.9371719, 45.8146591
30: -29.6986313, 16.9052162, -29.5328598, 16.7963142, -45.9532204, 45.8904533
31: -26.4884987, 19.1314831, -26.2451134, 19.0561581, -45.5446548, 45.3765945
32: -42.2826500, 8.5670633, -42.0996666, 8.4160938, -47.5265121, 47.5126724
33: -72.5251846, -5.5320139, -72.2422028, -5.7406569, -61.3520355, 61.2766876
34: -56.6141510, -5.3998232, -56.3930664, -5.5277777, -43.6968956, 43.5873108
35: -50.2920570, 0.1378641, -50.0484085, 0.0049200, -48.3504105, 48.2132721
36: -47.8961792, 5.0622663, -47.6272774, 4.9320717, -52.1150665, 51.9777679
37: -83.8009949, -17.4206238, -83.5548248, -17.5122452, -58.4591522, 58.3274384
38: -58.8665962, 3.3631763, -58.5087395, 3.2156401, -61.4703903, 61.2740250
39: -79.1366043, -11.4983664, -78.8486481, -11.6182823, -65.4513931, 65.2933197
40: -67.7463303, -18.3003483, -67.5811920, -18.4253922, -41.1865692, 41.1710587
41: -55.2488327, -6.7697201, -55.1162148, -6.8784866, -42.2537994, 42.2847176
42: -33.9648743, 6.8471966, -33.8990631, 6.7359886, -37.6093330, 37.6685677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=208, inp2_unstable=209, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=248, inp2_unstable=245, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=19, inp2_unstable=19, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 936
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 186
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1250
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1265
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1266
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 951
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 2016
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 953
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 480
type: B, layer: 1, pos: 1289
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1698

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0138799, upper bound: 44.6970552
time: 54.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 14, lower bound: -45.0219157, upper bound: 44.7616849
time: 42.70 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 99.79 seconds
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7447073, upper bound: 45.0026205
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7802455, upper bound: 45.0026205
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8968111, upper bound: 45.0048403
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9324096, upper bound: 45.0048403
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7800101, upper bound: 45.0026349
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8154808, upper bound: 45.0026349
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9321783, upper bound: 45.0048451
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9677080, upper bound: 45.0048451
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7239320, upper bound: 45.0222767
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7596017, upper bound: 45.0222769
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8760054, upper bound: 45.0244886
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9117086, upper bound: 45.0244884
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7593058, upper bound: 45.0222785
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7948736, upper bound: 45.0222785
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9114412, upper bound: 45.0244908
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9470702, upper bound: 45.0244906
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7829978, upper bound: 45.0039566
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8185839, upper bound: 45.0039566
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9352121, upper bound: 45.0061752
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9708602, upper bound: 45.0061752
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9705638, upper bound: 44.9678631
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0061518, upper bound: 44.9678631
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8183121, upper bound: 45.0039640
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8538154, upper bound: 45.0039640
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9705874, upper bound: 45.0061784
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0061781, upper bound: 45.0061784
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7621609, upper bound: 45.0236082
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7979205, upper bound: 45.0236081
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9143340, upper bound: 45.0258380
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9501261, upper bound: 45.0258380
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.7975682, upper bound: 45.0236104
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.8332051, upper bound: 45.0236104
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9498142, upper bound: 45.0258410
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -44.9854897, upper bound: 45.0258412
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0138792, upper bound: 44.6262040
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0219138, upper bound: 44.6910279
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0138792, upper bound: 44.6619637
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0219138, upper bound: 44.7264867
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0161152, upper bound: 44.7559468
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0241432, upper bound: 44.8209232
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0238539, upper bound: 44.7329630
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0251168, upper bound: 44.8574554
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0138799, upper bound: 44.6613904
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0219157, upper bound: 44.7262997
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0138799, upper bound: 44.6970552
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 99.79
Output dim: 14, lower bound: -45.0219157, upper bound: 44.7616849
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8579542
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0258411, upper bound: 44.8934650
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -44.9487730, upper bound: 45.0244471
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -44.9873345, upper bound: 45.0244905
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -44.9872670, upper bound: 45.0257990
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8380079
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0236103, upper bound: 44.8735212
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0258411, upper bound: 44.9902334
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 99.79
Output dim: 14, lower bound: -45.0258411, upper bound: 45.0258410

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 64.25 + 7193.58 = 7257.84 seconds
