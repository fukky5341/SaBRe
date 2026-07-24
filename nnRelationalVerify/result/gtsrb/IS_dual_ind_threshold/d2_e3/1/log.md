## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 20.1836426535


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=263, inp2_unstable=263, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498)
1: (-11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070)
2: (-9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425)
3: (-9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4056396, 38.4056396)
4: (-16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9559555, 41.9559555)
5: (-7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1119690, 36.1119690)
6: (-38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812)
7: (-11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.6014786, 38.6014748)
8: (-21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7545319, 50.7545319)
9: (-13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186)
10: (-22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662)
11: (-23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790)
12: (-44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2175751, 45.2175751)
13: (-37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5576782, 59.5576782)
14: (-64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507)
15: (-21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186)
16: (-23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021)
17: (-58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0626373, 56.0626373)
18: (-35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650)
19: (-26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775)
20: (-21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115)
21: (-27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035)
22: (-32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925)
23: (-24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843)
24: (-30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705)
25: (-28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095)
26: (-41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693)
27: (-26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029)
28: (-25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052)
29: (-27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4229507, 38.4229507)
30: (-26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750)
31: (-35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546)
32: (-35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8387909, 45.8387985)
33: (-63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3612823, 55.3612900)
34: (-57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5909424, 47.5909424)
35: (-56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8676910, 44.8676949)
36: (-53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4313049, 49.4312973)
37: (-78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8089600, 60.8089523)
38: (-63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6764984, 59.6764984)
39: (-72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0168457, 58.0168381)
40: (-51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460)
41: (-40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611)
42: (-26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 57.47 = 60.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -20.2038465, upper bound: 20.2038465

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1391674, upper bound: 20.1948068
time: 52.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2009809, upper bound: 20.2009814
time: 104.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 156.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 156.32
Output dim: 5, lower bound: -20.1391674, upper bound: 20.1948068
IS_A2, status: Status.UNKNOWN, split count: 1, time: 156.32
Output dim: 5, lower bound: -20.2009809, upper bound: 20.2009814

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -37.6382751, 17.5744839, -37.6791458, 17.5932503, -55.2315254, 55.2536316
1: -11.9135551, 22.4706497, -11.9455767, 22.4777203, -34.3912735, 34.4162254
2: -9.6774540, 25.2769547, -9.7299509, 25.2847767, -34.9622307, 35.0069046
3: -9.5335999, 28.9514198, -9.5997715, 28.9633102, -38.2677841, 38.3231049
4: -16.5670700, 25.3575821, -16.6345634, 25.3676548, -41.8182373, 41.8758774
5: -7.3777080, 29.0151806, -7.4402113, 29.0291824, -35.9812660, 36.0298767
6: -38.2353668, 11.9772367, -38.2508087, 12.0123425, -50.2477112, 50.2280464
7: -11.0621729, 28.6554737, -11.1045895, 28.6647797, -38.5134506, 38.5440750
8: -21.1911068, 29.8577232, -21.2513027, 29.8665390, -50.6292572, 50.6827545
9: -13.7240763, 28.3075924, -13.7477055, 28.3291779, -42.0532532, 42.0552979
10: -22.0972672, 31.9493675, -22.1198311, 31.9969864, -54.0942535, 54.0691986
11: -23.7029495, 14.6085491, -23.7278671, 14.6821079, -38.3850555, 38.3364182
12: -44.2497101, 4.2523060, -44.2626953, 4.3724718, -45.0849152, 44.9752045
13: -37.4660835, 22.2577209, -37.4810257, 22.3149681, -59.4741974, 59.4336472
14: -64.8790207, 2.5434523, -64.9068756, 2.6547756, -67.5337982, 67.4503250
15: -21.7882576, 20.3370819, -21.8599243, 20.3574944, -42.1457520, 42.1970062
16: -23.4331970, 21.6997566, -23.4618492, 21.7365417, -45.1697388, 45.1616058
17: -58.3988037, -1.3222122, -58.4137878, -1.2254486, -55.9444275, 55.8612709
18: -35.8489456, 14.6183376, -35.8651047, 14.6384020, -50.4873466, 50.4834442
19: -26.4344444, 9.4368944, -26.4494438, 9.4757166, -35.9101601, 35.8863373
20: -21.5376530, 15.8358345, -21.5593052, 15.8790989, -37.4167519, 37.3951416
21: -27.2797585, 12.9025078, -27.2987766, 12.9558678, -40.2356262, 40.2012863
22: -32.1002808, 10.6065350, -32.1217651, 10.6265631, -42.7268448, 42.7283020
23: -24.5893326, 13.9974585, -24.6032009, 14.0291805, -38.6185150, 38.6006584
24: -30.7401772, 13.7240047, -30.7610264, 13.7349377, -44.4751129, 44.4850311
25: -28.8940201, 12.8859119, -28.9118767, 12.9175291, -41.8115501, 41.7977905
26: -41.0249710, 16.9784622, -41.0479965, 17.0360298, -58.0610008, 58.0264587
27: -26.0718613, 18.1813164, -26.1095467, 18.1909599, -44.2628212, 44.2908630
28: -25.0722923, 17.2797661, -25.0892601, 17.3111782, -42.3834686, 42.3690262
29: -27.6132011, 10.8896141, -27.6261921, 10.9297428, -38.3614349, 38.3348465
30: -26.8439808, 18.2948647, -26.8647556, 18.3243523, -45.1683350, 45.1596222
31: -35.4059448, 12.0682755, -35.4245644, 12.1122341, -47.5181808, 47.4928398
32: -35.2254143, 10.9485016, -35.2419891, 10.9959841, -45.7640076, 45.7314301
33: -63.6518211, -3.7802582, -63.6967239, -3.7572374, -55.2449265, 55.2706451
34: -57.7887344, -6.3891497, -57.8287811, -6.3685789, -47.4823151, 47.5096283
35: -56.0786934, -4.3652163, -56.0933113, -4.3498335, -44.8005142, 44.8096123
36: -53.4860344, 0.8299932, -53.4984932, 0.8654499, -49.3741913, 49.3503418
37: -78.2643738, -14.3095951, -78.2874146, -14.2721834, -60.7319336, 60.7276230
38: -63.8174400, 0.3482962, -63.8347626, 0.3921337, -59.6074066, 59.5860977
39: -72.1255112, -8.2018290, -72.1506348, -8.1742992, -57.9451675, 57.9471207
40: -51.3515511, -6.2260165, -51.3757629, -6.2104297, -45.1411209, 45.1497459
41: -40.0514183, 12.2311277, -40.0693207, 12.2546978, -52.3061142, 52.3004494
42: -26.1608162, 11.9289856, -26.1762428, 11.9601316, -38.1209488, 38.1052284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=263, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408032
time: 116.99 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
time: 51.22 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -37.7496605, 17.6441803, -37.7056999, 17.6068840, -55.3565445, 55.3498802
1: -12.0046806, 22.5328045, -11.9637852, 22.4827576, -34.4874382, 34.4965897
2: -9.7977486, 25.3766251, -9.7673388, 25.2899170, -35.0876656, 35.1439629
3: -9.6718359, 29.0955124, -9.6488018, 28.9703140, -38.4088974, 38.5162048
4: -16.7155685, 25.5171165, -16.6832924, 25.3741493, -41.9693069, 42.0849915
5: -7.5137315, 29.1633663, -7.4857950, 29.0382023, -36.1189003, 36.2251053
6: -38.3427429, 12.0630569, -38.2607498, 12.0309334, -50.3736763, 50.3238068
7: -11.1983471, 28.7107868, -11.1353788, 28.6704636, -38.6451721, 38.6345024
8: -21.3381748, 29.9821987, -21.2963104, 29.8718033, -50.7731781, 50.8562851
9: -13.7917690, 28.3723221, -13.7641916, 28.3395958, -42.1313629, 42.1365128
10: -22.2193794, 32.0581245, -22.1355019, 32.0246124, -54.2439919, 54.1936264
11: -23.9940300, 14.7619839, -23.7449608, 14.7386093, -38.7326393, 38.5069427
12: -44.5388374, 4.5109806, -44.2714462, 4.4635162, -45.4673615, 45.2245026
13: -37.5825806, 22.4113350, -37.4901237, 22.3561554, -59.6550751, 59.5747223
14: -65.1690979, 2.7637615, -64.9233780, 2.7419424, -67.9110413, 67.6871414
15: -21.9579887, 20.5450916, -21.9130650, 20.3714447, -42.3294334, 42.4581566
16: -23.5866184, 21.7653217, -23.4802132, 21.7523632, -45.3389816, 45.2455368
17: -58.6971550, -1.1192093, -58.4234238, -1.1523800, -56.3461227, 56.0451622
18: -35.9291801, 14.6742992, -35.8714218, 14.6525974, -50.5817795, 50.5457230
19: -26.5694122, 9.5213757, -26.4594536, 9.5041466, -36.0735588, 35.9808273
20: -21.6659145, 15.9235601, -21.5746651, 15.9100914, -37.5760040, 37.4982262
21: -27.4702721, 13.0176783, -27.3115845, 12.9957142, -40.4659882, 40.3292618
22: -32.1817207, 10.6841841, -32.1348877, 10.6351318, -42.8168526, 42.8190727
23: -24.7047901, 14.0704880, -24.6123924, 14.0521317, -38.7569199, 38.6828804
24: -30.8073368, 13.7624130, -30.7700577, 13.7426224, -44.5499573, 44.5324707
25: -28.9824104, 12.9683523, -28.9220619, 12.9403458, -41.9227562, 41.8904152
26: -41.1793060, 17.0977268, -41.0631638, 17.0747356, -58.2540436, 58.1608887
27: -26.1766624, 18.2431259, -26.1304302, 18.1973686, -44.3740311, 44.3735580
28: -25.1709099, 17.3478832, -25.1009960, 17.3330765, -42.5039864, 42.4488792
29: -27.7262878, 10.9846144, -27.6321507, 10.9599876, -38.5078278, 38.4327240
30: -26.9435787, 18.3527737, -26.8775330, 18.3341484, -45.2777252, 45.2303085
31: -35.5505180, 12.1589489, -35.4368515, 12.1447277, -47.6952438, 47.5958023
32: -35.3582649, 11.0549717, -35.2521515, 11.0289192, -45.9405365, 45.8463974
33: -63.7570267, -3.6513286, -63.7271957, -3.7415228, -55.3612137, 55.4746475
34: -57.8891106, -6.2396250, -57.8583527, -6.3554354, -47.5874786, 47.7402878
35: -56.1235847, -4.2851858, -56.1021805, -4.3399420, -44.8513870, 44.9583511
36: -53.5936127, 0.9340706, -53.5063858, 0.8912067, -49.5088348, 49.4496536
37: -78.3905106, -14.2087650, -78.3006973, -14.2459478, -60.8618317, 60.8475418
38: -63.9376335, 0.4755068, -63.8450928, 0.4238009, -59.7624969, 59.7040558
39: -72.2366486, -8.1005487, -72.1661072, -8.1604004, -58.0854797, 58.0568771
40: -51.4378090, -6.1263266, -51.3855286, -6.2022729, -45.2355347, 45.2592010
41: -40.1322098, 12.2943897, -40.0810280, 12.2688379, -52.4010468, 52.3754196
42: -26.2531242, 12.0022173, -26.1874390, 11.9762325, -38.2293549, 38.1896553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=263, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1975767, upper bound: 20.1473699
time: 50.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1975802, upper bound: 20.1975803
time: 84.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 136.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 136.91
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408032
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 136.91
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 136.91
Output dim: 5, lower bound: -20.1975767, upper bound: 20.1473699
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 136.91
Output dim: 5, lower bound: -20.1975802, upper bound: 20.1975803

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -37.6238976, 17.5717812, -37.7161407, 17.6338749, -55.2577744, 55.2879219
1: -11.9061813, 22.4692745, -11.9764175, 22.6150837, -34.5212631, 34.4456940
2: -9.6690607, 25.2752514, -9.7523870, 25.4392891, -35.1083488, 35.0276375
3: -9.5235291, 28.9481926, -9.6107836, 29.1421909, -38.4439392, 38.3301010
4: -16.5588684, 25.3547211, -16.6710434, 25.5484543, -41.9915543, 41.9104309
5: -7.3679924, 29.0126476, -7.4654598, 29.1742115, -36.1205063, 36.0501442
6: -38.2312851, 11.9679012, -38.3216553, 12.0334272, -50.2647133, 50.2895584
7: -11.0512447, 28.6532249, -11.1572933, 28.7710438, -38.6219635, 38.5925331
8: -21.1818848, 29.8553543, -21.2798519, 30.0528069, -50.8215332, 50.6998749
9: -13.7191467, 28.3007946, -13.9019337, 28.3690281, -42.0881729, 42.2027283
10: -22.0923805, 31.9350052, -22.4697495, 32.0206032, -54.1129837, 54.4047546
11: -23.6994419, 14.5842094, -23.9666576, 14.6644497, -38.3638916, 38.5508652
12: -44.2463646, 4.2379723, -44.5744247, 4.4035501, -45.1088562, 45.2797089
13: -37.4613495, 22.2507305, -37.5057755, 22.3859253, -59.5162277, 59.4717484
14: -64.8719788, 2.5295086, -65.2514877, 2.6673260, -67.5393066, 67.7809982
15: -21.7635384, 20.3338089, -21.8661442, 20.4484749, -42.2120132, 42.1999512
16: -23.4266396, 21.6812744, -23.6309872, 21.7344856, -45.1611252, 45.3122635
17: -58.3966408, -1.3316641, -58.6513710, -1.1932564, -55.9643784, 56.1093903
18: -35.8462372, 14.6095448, -36.0249023, 14.6486454, -50.4948807, 50.6344452
19: -26.4316959, 9.4307079, -26.5979996, 9.4847679, -35.9164658, 36.0287094
20: -21.5340862, 15.8290644, -21.7054577, 15.8842869, -37.4183731, 37.5345230
21: -27.2760296, 12.8950281, -27.5115585, 12.9734192, -40.2494507, 40.4065857
22: -32.0968628, 10.5931568, -32.1946564, 10.6519642, -42.7488251, 42.7878113
23: -24.5865746, 13.9896984, -24.7112789, 14.0414753, -38.6280518, 38.7009773
24: -30.7351017, 13.7205868, -30.8023510, 13.7436342, -44.4787369, 44.5229378
25: -28.8907261, 12.8801622, -28.9940224, 12.9433870, -41.8341141, 41.8741837
26: -41.0203133, 16.9632378, -41.2534180, 17.0568390, -58.0771523, 58.2166557
27: -26.0635815, 18.1792259, -26.1579628, 18.2462158, -44.3097992, 44.3371887
28: -25.0694160, 17.2729778, -25.1499176, 17.3211689, -42.3905869, 42.4228973
29: -27.6092815, 10.8790245, -27.7291794, 10.9425306, -38.3712234, 38.4307938
30: -26.8403225, 18.2803764, -26.9555111, 18.3270607, -45.1673813, 45.2358856
31: -35.4024658, 12.0598526, -35.6009140, 12.1159019, -47.5183678, 47.6607666
32: -35.2214050, 10.9437523, -35.3624420, 11.0226173, -45.7726212, 45.8597946
33: -63.6443787, -3.7847199, -63.7177010, -3.5884070, -55.4526520, 55.2867508
34: -57.7837639, -6.3929443, -57.8553047, -6.2383089, -47.6505737, 47.5331039
35: -56.0713196, -4.3674259, -56.1014786, -4.2078638, -44.9915085, 44.8234406
36: -53.4805450, 0.8286018, -53.5195160, 0.9266796, -49.4302750, 49.3711472
37: -78.2587280, -14.3175688, -78.3435898, -14.2466049, -60.7734833, 60.7924271
38: -63.8110657, 0.3453789, -63.8735733, 0.4687018, -59.6786804, 59.6202240
39: -72.1188202, -8.2044821, -72.1976166, -8.0944891, -58.0150604, 58.0184555
40: -51.3468018, -6.2287726, -51.4499359, -6.1498260, -45.1969757, 45.2211647
41: -40.0431213, 12.2286739, -40.1093254, 12.3008461, -52.3439674, 52.3379974
42: -26.1576271, 11.9133177, -26.2440872, 11.9832048, -38.1408310, 38.1574059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
time: 51.03 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
time: 61.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -37.7463150, 17.6431351, -37.6664124, 17.5945320, -55.3408470, 55.3095474
1: -12.0013771, 22.5322189, -11.9247885, 22.4758873, -34.4772644, 34.4570084
2: -9.7933874, 25.3761005, -9.7162437, 25.2838020, -35.0771904, 35.0923462
3: -9.6669350, 29.0946903, -9.5913029, 28.9605980, -38.3944397, 38.4579926
4: -16.7110748, 25.5164108, -16.6317253, 25.3658562, -41.9565735, 42.0333557
5: -7.5088181, 29.1626205, -7.4293513, 29.0295334, -36.1053886, 36.1664124
6: -38.3416367, 12.0616875, -38.2474823, 12.0148754, -50.3565140, 50.3091698
7: -11.1939449, 28.7101498, -11.0856218, 28.6630020, -38.6329727, 38.5812607
8: -21.3334236, 29.9814358, -21.2403316, 29.8628311, -50.7597046, 50.7967682
9: -13.7903814, 28.3687477, -13.7475786, 28.2978020, -42.0881844, 42.1163254
10: -22.2173729, 32.0505981, -22.1120453, 31.9358749, -54.1532478, 54.1626434
11: -23.9928131, 14.7583494, -23.7307148, 14.6973515, -38.6901627, 38.4890633
12: -44.5377884, 4.5025148, -44.2593307, 4.3640623, -45.3662796, 45.2038345
13: -37.5814362, 22.4096966, -37.4765320, 22.3369064, -59.6292648, 59.5576630
14: -65.1663818, 2.7556219, -64.8919601, 2.6464672, -67.8128510, 67.6475830
15: -21.9552002, 20.5437927, -21.8803539, 20.3556633, -42.3108635, 42.4241486
16: -23.5846024, 21.7623215, -23.4565887, 21.7177353, -45.3023376, 45.2189102
17: -58.6961212, -1.1233587, -58.4118042, -1.2010012, -56.2973328, 56.0294495
18: -35.9280853, 14.6713848, -35.8586273, 14.6187382, -50.5468216, 50.5300140
19: -26.5683136, 9.5186405, -26.4464836, 9.4723625, -36.0406761, 35.9651260
20: -21.6643944, 15.9203386, -21.5564518, 15.8721361, -37.5365295, 37.4767914
21: -27.4689960, 13.0136471, -27.2965603, 12.9485531, -40.4175491, 40.3102074
22: -32.1808014, 10.6820259, -32.1239777, 10.6100035, -42.7908058, 42.8060036
23: -24.7037106, 14.0681133, -24.5999107, 14.0244617, -38.7281723, 38.6680222
24: -30.8062706, 13.7618246, -30.7576752, 13.7355776, -44.5418472, 44.5195007
25: -28.9814491, 12.9659395, -28.9107895, 12.9119291, -41.8933792, 41.8767281
26: -41.1779556, 17.0914288, -41.0468521, 17.0008202, -58.1787758, 58.1382828
27: -26.1743202, 18.2424927, -26.1030025, 18.1899967, -44.3643188, 44.3454971
28: -25.1696281, 17.3458900, -25.0859261, 17.3098907, -42.4795189, 42.4318161
29: -27.7255478, 10.9820900, -27.6234303, 10.9305630, -38.4773331, 38.4212341
30: -26.9425068, 18.3503914, -26.8649082, 18.3058949, -45.2484016, 45.2153015
31: -35.5489273, 12.1555042, -35.4182205, 12.1046534, -47.6535797, 47.5737228
32: -35.3571663, 11.0523205, -35.2390976, 10.9978352, -45.9051208, 45.8306732
33: -63.7533951, -3.6529288, -63.6843567, -3.7607589, -55.3384094, 55.4302368
34: -57.8867302, -6.2409716, -57.8301239, -6.3711748, -47.5696945, 47.7137146
35: -56.1213684, -4.2863550, -56.0762482, -4.3533010, -44.8366013, 44.9305763
36: -53.5926628, 0.9333124, -53.4952621, 0.8824015, -49.4986877, 49.4385605
37: -78.3892365, -14.2106276, -78.2858124, -14.2676477, -60.8394623, 60.8296432
38: -63.9357376, 0.4744596, -63.8230667, 0.4126301, -59.7491913, 59.6815643
39: -72.2350464, -8.1017637, -72.1470490, -8.1741152, -58.0653915, 58.0388031
40: -51.4363327, -6.1269979, -51.3679886, -6.2099953, -45.2263374, 45.2409897
41: -40.1305161, 12.2931881, -40.0611267, 12.2550468, -52.3855629, 52.3543167
42: -26.2522526, 12.0002251, -26.1773815, 11.9529819, -38.2052345, 38.1776047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
time: 49.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
time: 63.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -37.7354202, 17.6414623, -37.7429276, 17.6475410, -55.3829613, 55.3843918
1: -11.9973059, 22.5314388, -11.9948788, 22.6201496, -34.6174545, 34.5263176
2: -9.7893581, 25.3749428, -9.7900038, 25.4444542, -35.2338104, 35.1649475
3: -9.6618099, 29.0923386, -9.6599531, 29.1492062, -38.5851059, 38.5234489
4: -16.7074261, 25.5142956, -16.7200336, 25.5549774, -42.1427155, 42.1198425
5: -7.5040383, 29.1608429, -7.5112000, 29.1832390, -36.2581940, 36.2455368
6: -38.3386917, 12.0536766, -38.3318291, 12.0520163, -50.3907089, 50.3855057
7: -11.1875057, 28.7085724, -11.1880264, 28.7766991, -38.7537460, 38.6829758
8: -21.3289967, 29.9798622, -21.3250847, 30.0581112, -50.9655075, 50.8736343
9: -13.7869024, 28.3655739, -13.9184875, 28.3797245, -42.1666260, 42.2840614
10: -22.2144871, 32.0439568, -22.4854298, 32.0489349, -54.2634201, 54.5293884
11: -23.9905415, 14.7376986, -23.9837284, 14.7209558, -38.7114983, 38.7214279
12: -44.5355034, 4.4966860, -44.5832367, 4.4946260, -45.4914703, 45.5290680
13: -37.5779533, 22.4047890, -37.5149193, 22.4261665, -59.6963806, 59.6130447
14: -65.1620178, 2.7497854, -65.2680206, 2.7543850, -67.9164047, 68.0178070
15: -21.9332371, 20.5419178, -21.9186630, 20.4626579, -42.3958969, 42.4605789
16: -23.5800629, 21.7467422, -23.6495476, 21.7504597, -45.3305206, 45.3962898
17: -58.6949654, -1.1285658, -58.6610832, -1.1201439, -56.3660736, 56.2932510
18: -35.9264526, 14.6654510, -36.0315781, 14.6630116, -50.5894623, 50.6970291
19: -26.5666828, 9.5152302, -26.6080704, 9.5135307, -36.0802155, 36.1232986
20: -21.6623764, 15.9167957, -21.7208424, 15.9153328, -37.5777092, 37.6376381
21: -27.4665833, 13.0101976, -27.5244045, 13.0134735, -40.4800568, 40.5346031
22: -32.1783218, 10.6709690, -32.2078781, 10.6612215, -42.8395424, 42.8788452
23: -24.7020302, 14.0627937, -24.7204494, 14.0646439, -38.7666740, 38.7832413
24: -30.8024178, 13.7590494, -30.8116245, 13.7513189, -44.5537376, 44.5706749
25: -28.9791183, 12.9626322, -29.0042305, 12.9663496, -41.9454689, 41.9668617
26: -41.1747055, 17.0825081, -41.2686844, 17.0957184, -58.2704239, 58.3511925
27: -26.1684532, 18.2410507, -26.1788139, 18.2526093, -44.4210625, 44.4198647
28: -25.1680412, 17.3410339, -25.1617088, 17.3433361, -42.5113754, 42.5027428
29: -27.7224121, 10.9739819, -27.7352448, 10.9730282, -38.5179443, 38.5288696
30: -26.9399471, 18.3382702, -26.9683933, 18.3363800, -45.2763290, 45.3066635
31: -35.5470657, 12.1505585, -35.6132469, 12.1485567, -47.6956215, 47.7638054
32: -35.3542938, 11.0501595, -35.3726883, 11.0557060, -45.9492645, 45.9747391
33: -63.7496338, -3.6557670, -63.7482376, -3.5726709, -55.5689697, 55.4908600
34: -57.8841553, -6.2433262, -57.8848610, -6.2251244, -47.7557907, 47.7637482
35: -56.1161804, -4.2873812, -56.1104012, -4.1977882, -45.0424957, 44.9722443
36: -53.5880699, 0.9325771, -53.5274734, 0.9522018, -49.5649948, 49.4705048
37: -78.3849030, -14.2167025, -78.3570099, -14.2198896, -60.9017792, 60.9125671
38: -63.9312515, 0.4725432, -63.8840561, 0.5002236, -59.8337402, 59.7383270
39: -72.2300415, -8.1032896, -72.2132568, -8.0809593, -58.1540298, 58.1281433
40: -51.4332008, -6.1291385, -51.4593811, -6.1415639, -45.2916374, 45.3302422
41: -40.1239624, 12.2919350, -40.1212959, 12.3150539, -52.4390182, 52.4132309
42: -26.2499275, 11.9866085, -26.2555294, 11.9994850, -38.2494125, 38.2421379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=262, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1559530
time: 46.21 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1961887
time: 50.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 99.01 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1559530
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 99.01
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1961887

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -37.6228485, 17.5709095, -37.7159958, 17.6337624, -55.2566109, 55.2869034
1: -11.9058628, 22.4677734, -11.9763899, 22.6148872, -34.5207520, 34.4441643
2: -9.6686707, 25.2739639, -9.7523422, 25.4391060, -35.1077766, 35.0263062
3: -9.5230742, 28.9455109, -9.6107264, 29.1418133, -38.4430771, 38.3199463
4: -16.5583916, 25.3525848, -16.6709805, 25.5481377, -41.9907837, 41.9015541
5: -7.3674693, 29.0104313, -7.4653983, 29.1738873, -36.1196823, 36.0385246
6: -38.2304077, 11.9675026, -38.3215332, 12.0333729, -50.2637787, 50.2890358
7: -11.0508232, 28.6512794, -11.1572170, 28.7707558, -38.6212845, 38.5783195
8: -21.1813087, 29.8533993, -21.2797852, 30.0525265, -50.8206940, 50.6883240
9: -13.7185116, 28.2987900, -13.9018335, 28.3687553, -42.0872650, 42.2006226
10: -22.0915985, 31.9343319, -22.4696503, 32.0205307, -54.1121292, 54.4039841
11: -23.6973877, 14.5838842, -23.9663448, 14.6644058, -38.3617935, 38.5502281
12: -44.2447472, 4.2374544, -44.5742111, 4.4034672, -45.0802612, 45.2789688
13: -37.4608688, 22.2469788, -37.5057144, 22.3853645, -59.5424957, 59.4680328
14: -64.8692017, 2.5292072, -65.2510757, 2.6672888, -67.5364914, 67.7802811
15: -21.7631378, 20.3320293, -21.8660755, 20.4482002, -42.2113380, 42.1981049
16: -23.4257355, 21.6800880, -23.6308613, 21.7343140, -45.1600494, 45.3109512
17: -58.3942986, -1.3325071, -58.6510315, -1.1934061, -55.9177780, 56.1082726
18: -35.8443260, 14.6088047, -36.0246429, 14.6485815, -50.4929085, 50.6334457
19: -26.4293690, 9.4304218, -26.5976562, 9.4847126, -35.9140816, 36.0280762
20: -21.5320053, 15.8288469, -21.7051620, 15.8842516, -37.4162560, 37.5340080
21: -27.2739124, 12.8948488, -27.5112610, 12.9734068, -40.2473183, 40.4061089
22: -32.0941391, 10.5927591, -32.1942635, 10.6519032, -42.7460403, 42.7870216
23: -24.5839462, 13.9893446, -24.7109032, 14.0414257, -38.6253738, 38.7002487
24: -30.7315598, 13.7201347, -30.8018475, 13.7435884, -44.4751472, 44.5219803
25: -28.8876228, 12.8797359, -28.9935589, 12.9433184, -41.8309402, 41.8732948
26: -41.0181160, 16.9628639, -41.2530975, 17.0567322, -58.0748482, 58.2159615
27: -26.0614414, 18.1789894, -26.1576653, 18.2461815, -44.3076248, 44.3366547
28: -25.0670433, 17.2726326, -25.1495743, 17.3211288, -42.3881721, 42.4222069
29: -27.6062889, 10.8787193, -27.7287750, 10.9424915, -38.3640289, 38.4301147
30: -26.8375950, 18.2800064, -26.9551373, 18.3269958, -45.1645889, 45.2351456
31: -35.3994102, 12.0594521, -35.6004868, 12.1158695, -47.5152817, 47.6599388
32: -35.2196426, 10.9434605, -35.3622093, 11.0225830, -45.7705002, 45.8591995
33: -63.6426163, -3.7851348, -63.7174377, -3.5884838, -55.4411621, 55.2860260
34: -57.7823334, -6.3932657, -57.8550644, -6.2383614, -47.6242065, 47.5325165
35: -56.0703354, -4.3677158, -56.1013412, -4.2078972, -44.9701920, 44.8228912
36: -53.4789047, 0.8283167, -53.5192795, 0.9266462, -49.4115906, 49.3705521
37: -78.2564316, -14.3178968, -78.3432617, -14.2466183, -60.7417603, 60.7917252
38: -63.8085403, 0.3447838, -63.8732147, 0.4686408, -59.6506195, 59.6192322
39: -72.1159058, -8.2048464, -72.1972351, -8.0945349, -58.0016251, 58.0176468
40: -51.3455811, -6.2294860, -51.4497414, -6.1499124, -45.1956673, 45.2202568
41: -40.0421867, 12.2283554, -40.1091995, 12.3008156, -52.3430023, 52.3375549
42: -26.1563568, 11.9129715, -26.2439423, 11.9831657, -38.1395226, 38.1569138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1564538
time: 41.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
time: 76.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -37.6584930, 17.6058388, -37.7225113, 17.6337776, -55.2922707, 55.3283501
1: -11.9526653, 22.4536572, -11.9889393, 22.5874825, -34.5401459, 34.4425964
2: -9.7547369, 25.3077259, -9.7839909, 25.4163151, -35.1710510, 35.0917168
3: -9.6134033, 28.9502335, -9.6530418, 29.0889912, -38.4760513, 38.3735771
4: -16.6596565, 25.3996696, -16.7129860, 25.5066490, -42.0468597, 41.9979019
5: -7.4590597, 29.0442562, -7.5036039, 29.1338730, -36.1642685, 36.1210861
6: -38.2960663, 12.0287485, -38.3155670, 12.0443573, -50.3404236, 50.3443146
7: -11.1432791, 28.6083679, -11.1822262, 28.7342052, -38.6670227, 38.5769081
8: -21.2718506, 29.8779354, -21.3164062, 30.0155869, -50.8649139, 50.7614594
9: -13.7286978, 28.2617912, -13.9068727, 28.3352299, -42.0639267, 42.1686630
10: -22.1683121, 32.0042801, -22.4709682, 32.0357895, -54.2041016, 54.4752502
11: -23.8754463, 14.6997147, -23.9382172, 14.7144861, -38.5899315, 38.6379318
12: -44.4472656, 4.4588995, -44.5473328, 4.4877539, -45.3962250, 45.4543076
13: -37.5430527, 22.3044033, -37.5062485, 22.3842049, -59.5923309, 59.4922256
14: -65.0231018, 2.7215672, -65.2129822, 2.7505074, -67.7736053, 67.9345474
15: -21.8946953, 20.4439011, -21.9120846, 20.4227295, -42.3174248, 42.3559875
16: -23.5145607, 21.6881008, -23.6327019, 21.7265739, -45.2411346, 45.3208008
17: -58.5639610, -1.1940899, -58.6106796, -1.1352320, -56.2164841, 56.1749001
18: -35.8323593, 14.6167269, -35.9921417, 14.6531048, -50.4854660, 50.6088676
19: -26.4456406, 9.4776363, -26.5575275, 9.5080223, -35.9536629, 36.0351639
20: -21.5547485, 15.8889322, -21.6762924, 15.9115648, -37.4663124, 37.5652237
21: -27.3542156, 12.9858875, -27.4789162, 13.0106916, -40.3649063, 40.4648056
22: -32.0385399, 10.6324348, -32.1494904, 10.6542196, -42.6927605, 42.7819252
23: -24.5621834, 14.0194492, -24.6618767, 14.0587387, -38.6209221, 38.6813278
24: -30.6168098, 13.7133942, -30.7333832, 13.7441730, -44.3609848, 44.4467773
25: -28.8133545, 12.9178114, -28.9347382, 12.9602709, -41.7736244, 41.8525505
26: -41.0575180, 17.0362759, -41.2205963, 17.0890675, -58.1465836, 58.2568741
27: -26.0670910, 18.2167778, -26.1357307, 18.2475853, -44.3146744, 44.3525085
28: -25.0457306, 17.3005810, -25.1102448, 17.3376350, -42.3833656, 42.4108276
29: -27.5665169, 10.9389849, -27.6702194, 10.9671993, -38.3554459, 38.4286156
30: -26.7987862, 18.3014126, -26.9102211, 18.3301163, -45.1289024, 45.2116318
31: -35.3881721, 12.1017494, -35.5467072, 12.1415005, -47.5296707, 47.6484566
32: -35.2938309, 11.0297604, -35.3487282, 11.0496941, -45.8810806, 45.9283905
33: -63.7027512, -3.6850243, -63.7302094, -3.5799918, -55.5150909, 55.4405441
34: -57.8122482, -6.2820435, -57.8549805, -6.2325335, -47.6759186, 47.6932678
35: -56.0712242, -4.3186312, -56.0917778, -4.2034235, -44.9930420, 44.9165268
36: -53.5321198, 0.9088001, -53.5034142, 0.9479036, -49.5067596, 49.4217300
37: -78.2647476, -14.2481279, -78.3071213, -14.2251081, -60.7752686, 60.8292999
38: -63.8504906, 0.4286551, -63.8500862, 0.4903469, -59.7418594, 59.6584320
39: -72.1487579, -8.1184826, -72.1802826, -8.0860062, -58.0692444, 58.0783615
40: -51.3742981, -6.1540074, -51.4374161, -6.1489787, -45.2253189, 45.2834091
41: -40.0879173, 12.2710686, -40.1076355, 12.3091640, -52.3970795, 52.3787041
42: -26.2076416, 11.9688530, -26.2399120, 11.9941578, -38.2017975, 38.2087631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1901985, upper bound: 20.0942161
time: 84.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1015177
time: 50.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -37.7343712, 17.6405945, -37.7427750, 17.6474190, -55.3817902, 55.3833694
1: -11.9969692, 22.5299530, -11.9948034, 22.6199322, -34.6169014, 34.5247574
2: -9.7890005, 25.3736458, -9.7899723, 25.4442692, -35.2332687, 35.1636200
3: -9.6613817, 29.0896416, -9.6598949, 29.1488266, -38.5842590, 38.5132713
4: -16.7069416, 25.5121498, -16.7199707, 25.5546570, -42.1419220, 42.1109810
5: -7.5035086, 29.1586266, -7.5111284, 29.1829128, -36.2574005, 36.2339478
6: -38.3378372, 12.0532579, -38.3317108, 12.0519667, -50.3898048, 50.3849678
7: -11.1870527, 28.7066460, -11.1879530, 28.7764244, -38.7530670, 38.6687546
8: -21.3284245, 29.9779167, -21.3249931, 30.0578346, -50.9646759, 50.8620911
9: -13.7862911, 28.3635387, -13.9183893, 28.3794594, -42.1657486, 42.2819290
10: -22.2137508, 32.0432053, -22.4853134, 32.0488281, -54.2625809, 54.5285187
11: -23.9884872, 14.7373695, -23.9834499, 14.7208967, -38.7093849, 38.7208176
12: -44.5338974, 4.4961672, -44.5830154, 4.4945507, -45.4628448, 45.5283051
13: -37.5774384, 22.4010353, -37.5148621, 22.4256325, -59.7226639, 59.6093445
14: -65.1592712, 2.7495575, -65.2676086, 2.7543507, -67.9136200, 68.0171661
15: -21.9327888, 20.5401268, -21.9186096, 20.4623871, -42.3951759, 42.4587364
16: -23.5791817, 21.7455502, -23.6494141, 21.7502823, -45.3294640, 45.3949661
17: -58.6926422, -1.1294775, -58.6607437, -1.1202850, -56.3194656, 56.2921104
18: -35.9245529, 14.6647873, -36.0313225, 14.6629639, -50.5875168, 50.6961098
19: -26.5643482, 9.5149279, -26.6077347, 9.5134811, -36.0778275, 36.1226616
20: -21.6602974, 15.9165955, -21.7205276, 15.9152889, -37.5755844, 37.6371231
21: -27.4644547, 13.0100441, -27.5240974, 13.0134754, -40.4779282, 40.5341415
22: -32.1756477, 10.6705914, -32.2075119, 10.6611423, -42.8367920, 42.8781052
23: -24.6994267, 14.0624075, -24.7200813, 14.0645962, -38.7640228, 38.7824898
24: -30.7988777, 13.7586098, -30.8111115, 13.7512665, -44.5501442, 44.5697212
25: -28.9760113, 12.9622345, -29.0038109, 12.9662914, -41.9423027, 41.9660454
26: -41.1724854, 17.0820904, -41.2683716, 17.0956573, -58.2681427, 58.3504639
27: -26.1663361, 18.2407761, -26.1784992, 18.2525654, -44.4188995, 44.4192734
28: -25.1656590, 17.3406696, -25.1613579, 17.3432941, -42.5089531, 42.5020294
29: -27.7194118, 10.9736576, -27.7347908, 10.9729834, -38.5107498, 38.5280800
30: -26.9372025, 18.3379002, -26.9680023, 18.3363380, -45.2735405, 45.3059006
31: -35.5440063, 12.1501217, -35.6127930, 12.1484766, -47.6924820, 47.7629166
32: -35.3525925, 11.0498600, -35.3724136, 11.0556774, -45.9471741, 45.9741211
33: -63.7477646, -3.6561670, -63.7479782, -3.5727015, -55.5574875, 55.4901123
34: -57.8826904, -6.2437277, -57.8846512, -6.2251644, -47.7294006, 47.7631149
35: -56.1151924, -4.2877045, -56.1102562, -4.1978216, -45.0212631, 44.9716873
36: -53.5864601, 0.9323511, -53.5272293, 0.9521399, -49.5462799, 49.4699173
37: -78.3825912, -14.2170763, -78.3566742, -14.2199459, -60.8701019, 60.9118958
38: -63.9287491, 0.4719477, -63.8837051, 0.5001659, -59.8056946, 59.7373276
39: -72.2270889, -8.1036100, -72.2128677, -8.0809908, -58.1406174, 58.1273193
40: -51.4319916, -6.1298027, -51.4592133, -6.1416445, -45.2903481, 45.3294106
41: -40.1230278, 12.2916145, -40.1211624, 12.3150005, -52.4380264, 52.4127769
42: -26.2486992, 11.9862547, -26.2553253, 11.9994211, -38.2481194, 38.2415810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=262, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1901985, upper bound: 20.1344525
time: 55.53 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1417534
time: 49.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 107.54 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1564538
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1901985, upper bound: 20.0942161
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1015177
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1901985, upper bound: 20.1344525
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 107.54
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1417534

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -37.6228485, 17.5709095, -37.7867966, 17.6848125, -55.3076630, 55.3577042
1: -11.9058628, 22.4677734, -12.0357819, 22.6700439, -34.5759048, 34.5035553
2: -9.6686707, 25.2739639, -9.8205233, 25.5310669, -35.1997375, 35.0944862
3: -9.5230742, 28.9455109, -9.6829815, 29.2742615, -38.5743027, 38.3939323
4: -16.5583916, 25.3525848, -16.7526932, 25.6977596, -42.1408386, 41.9831543
5: -7.3674693, 29.0104313, -7.5391469, 29.3082829, -36.2553101, 36.1104164
6: -38.2304077, 11.9675026, -38.4147110, 12.0841227, -50.3145294, 50.3822136
7: -11.0508232, 28.6512794, -11.2508221, 28.8168888, -38.6710129, 38.6710815
8: -21.1813087, 29.8533993, -21.3669872, 30.1684875, -50.9405441, 50.7729416
9: -13.7185116, 28.2987900, -13.9461069, 28.4118214, -42.1303329, 42.2448959
10: -22.0915985, 31.9343319, -22.5695667, 32.0834732, -54.1718903, 54.5038986
11: -23.6973877, 14.5838842, -24.2341614, 14.7443562, -38.4417419, 38.8180466
12: -44.2447472, 4.2374544, -44.8505630, 4.5418110, -45.2154999, 45.5573807
13: -37.4608688, 22.2469788, -37.6071320, 22.4810181, -59.6326141, 59.5897903
14: -64.8692017, 2.5292072, -65.5131226, 2.7758169, -67.6450195, 68.0423279
15: -21.7631378, 20.3320293, -21.9636269, 20.6385899, -42.4017258, 42.2956543
16: -23.4257355, 21.6800880, -23.7562504, 21.7630768, -45.1888123, 45.4363403
17: -58.3942986, -1.3325071, -58.9345932, -1.0873919, -56.0204926, 56.4201813
18: -35.8443260, 14.6088047, -36.0886116, 14.6845112, -50.5288391, 50.6974182
19: -26.4293690, 9.4304218, -26.7180824, 9.5308828, -35.9602509, 36.1485062
20: -21.5320053, 15.8288469, -21.8120480, 15.9288301, -37.4608345, 37.6408958
21: -27.2739124, 12.8948488, -27.6831455, 13.0356703, -40.3095818, 40.5779953
22: -32.0941391, 10.5927591, -32.2544861, 10.7099104, -42.8040504, 42.8472443
23: -24.5839462, 13.9893446, -24.8131599, 14.0830669, -38.6670151, 38.8025055
24: -30.7315598, 13.7201347, -30.8485203, 13.7711220, -44.5026817, 44.5686569
25: -28.8876228, 12.8797359, -29.0644646, 12.9943085, -41.8819313, 41.9441986
26: -41.0181160, 16.9628639, -41.3845024, 17.1181946, -58.1363106, 58.3473663
27: -26.0614414, 18.1789894, -26.2249813, 18.2984066, -44.3598480, 44.4039688
28: -25.0670433, 17.2726326, -25.2314911, 17.3581791, -42.4252243, 42.5041237
29: -27.6062889, 10.8787193, -27.8295803, 10.9974451, -38.4194336, 38.5329857
30: -26.8375950, 18.2800064, -27.0345058, 18.3547707, -45.1923676, 45.3145142
31: -35.3994102, 12.0594521, -35.7272949, 12.1628742, -47.5622864, 47.7867470
32: -35.2196426, 10.9434605, -35.4788132, 11.0814133, -45.8287125, 45.9806900
33: -63.6426163, -3.7851348, -63.7775116, -3.4826384, -55.5744171, 55.3426895
34: -57.7823334, -6.3932657, -57.9151917, -6.1096716, -47.7818298, 47.5856934
35: -56.0703354, -4.3677158, -56.1314545, -4.1429834, -45.0578918, 44.8476982
36: -53.4789047, 0.8283167, -53.6143951, 0.9949369, -49.4815979, 49.4650040
37: -78.2564316, -14.3178968, -78.4472580, -14.1827717, -60.7965851, 60.8841248
38: -63.8085403, 0.3447838, -63.9748917, 0.5519233, -59.7355194, 59.7284698
39: -72.1159058, -8.2048464, -72.2838516, -8.0217142, -58.0761871, 58.1235123
40: -51.3455811, -6.2294860, -51.5115967, -6.0662932, -45.2792892, 45.2821121
41: -40.0421867, 12.2283554, -40.1726303, 12.3402004, -52.3823853, 52.4009857
42: -26.1563568, 11.9129715, -26.3218689, 12.0252142, -38.1815720, 38.2348404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1466596
time: 266.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1881893
time: 46.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -37.6584930, 17.6058388, -37.6545868, 17.6013470, -55.2598419, 55.2604256
1: -11.9526653, 22.4536572, -11.9382629, 22.5753002, -34.5279655, 34.3919220
2: -9.7547369, 25.3077259, -9.6934128, 25.4033127, -35.1580505, 35.0011368
3: -9.6134033, 28.9502335, -9.5375290, 29.0700970, -38.4639053, 38.2558899
4: -16.6596565, 25.3996696, -16.5963001, 25.4900208, -42.0345306, 41.8805161
5: -7.4590597, 29.0442562, -7.3952103, 29.1108456, -36.1469727, 36.0122986
6: -38.2960663, 12.0287485, -38.2896690, 11.9907169, -50.2867813, 50.3184166
7: -11.1432791, 28.6083679, -11.1089058, 28.7192173, -38.6587448, 38.5055542
8: -21.2718506, 29.8779354, -21.2101917, 30.0014267, -50.8587494, 50.6534958
9: -13.7286978, 28.2617912, -13.8667946, 28.3027954, -42.0314941, 42.1285858
10: -22.1683121, 32.0042801, -22.4328156, 31.9594612, -54.1277733, 54.4370956
11: -23.8754463, 14.6997147, -23.8963070, 14.5843525, -38.4598007, 38.5960236
12: -44.4472656, 4.4588995, -44.5255661, 4.2762136, -45.1821136, 45.4500809
13: -37.5430527, 22.3044033, -37.4820862, 22.2869911, -59.4937134, 59.4825211
14: -65.0231018, 2.7215672, -65.1685944, 2.5522118, -67.5753174, 67.8901596
15: -21.8946953, 20.4439011, -21.7877407, 20.3878803, -42.2825775, 42.2316437
16: -23.5145607, 21.6881008, -23.5854111, 21.6735802, -45.1881409, 45.2735138
17: -58.5639610, -1.1940899, -58.5858688, -1.3052101, -56.0434418, 56.1768303
18: -35.8323593, 14.6167269, -35.9692993, 14.6184263, -50.4507866, 50.5860252
19: -26.4456406, 9.4776363, -26.5323658, 9.4402065, -35.8858490, 36.0100021
20: -21.5547485, 15.8889322, -21.6392479, 15.8371458, -37.3918953, 37.5281792
21: -27.3542156, 12.9858875, -27.4469967, 12.9170952, -40.2713089, 40.4328842
22: -32.0385399, 10.6324348, -32.1146774, 10.6246891, -42.6632309, 42.7471123
23: -24.5621834, 14.0194492, -24.6388302, 14.0035877, -38.5657730, 38.6582794
24: -30.6168098, 13.7133942, -30.7032318, 13.7254915, -44.3423004, 44.4166260
25: -28.8133545, 12.9178114, -28.9066525, 12.9056034, -41.7189560, 41.8244629
26: -41.0575180, 17.0362759, -41.1822510, 16.9927101, -58.0502281, 58.2185287
27: -26.0670910, 18.2167778, -26.0772705, 18.2315788, -44.2986679, 44.2940483
28: -25.0457306, 17.3005810, -25.0815125, 17.2838936, -42.3296242, 42.3820953
29: -27.5665169, 10.9389849, -27.6511459, 10.8964405, -38.2842865, 38.4123306
30: -26.7987862, 18.3014126, -26.8765030, 18.2912388, -45.0900269, 45.1779175
31: -35.3881721, 12.1017494, -35.5156517, 12.0645924, -47.4527664, 47.6174011
32: -35.2938309, 11.0297604, -35.3219147, 10.9690285, -45.7930145, 45.9041824
33: -63.7027512, -3.6850243, -63.6547890, -3.6186485, -55.4813766, 55.3442764
34: -57.8122482, -6.2820435, -57.7853508, -6.2663336, -47.6509476, 47.5933685
35: -56.0712242, -4.3186312, -56.0681686, -4.2289200, -44.9761124, 44.8461037
36: -53.5321198, 0.9088001, -53.4828873, 0.8868904, -49.4428558, 49.4159393
37: -78.2647476, -14.2481279, -78.2705688, -14.2893410, -60.7330322, 60.7699127
38: -63.8504906, 0.4286551, -63.8221970, 0.4151111, -59.6757202, 59.6460495
39: -72.1487579, -8.1184826, -72.1393204, -8.1265736, -58.0361557, 58.0426254
40: -51.3742981, -6.1540074, -51.4039078, -6.1727252, -45.2015724, 45.2499008
41: -40.0879173, 12.2710686, -40.0774460, 12.2714920, -52.3594093, 52.3485146
42: -26.2076416, 11.9688530, -26.2128410, 11.9469023, -38.1545448, 38.1816940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0506689
time: 43.13 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0922015
time: 57.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -37.7343712, 17.6405945, -37.6748619, 17.6150055, -55.3493767, 55.3154564
1: -11.9969692, 22.5299530, -11.9441519, 22.6077919, -34.6047592, 34.4741058
2: -9.7890005, 25.3736458, -9.6993380, 25.4312859, -35.2202873, 35.0729828
3: -9.6613817, 29.0896416, -9.5444393, 29.1299133, -38.5721054, 38.3955460
4: -16.7069416, 25.5121498, -16.6032677, 25.5380363, -42.1295853, 41.9935760
5: -7.5035086, 29.1586266, -7.4027500, 29.1599121, -36.2400856, 36.1251717
6: -38.3378372, 12.0532579, -38.3058472, 11.9983492, -50.3361855, 50.3591042
7: -11.1870527, 28.7066460, -11.1146221, 28.7614441, -38.7448502, 38.5973663
8: -21.3284245, 29.9779167, -21.2187996, 30.0436974, -50.9585114, 50.7541580
9: -13.7862911, 28.3635387, -13.8782482, 28.3470058, -42.1332970, 42.2417870
10: -22.2137508, 32.0432053, -22.4470978, 31.9724464, -54.1861954, 54.4903030
11: -23.9884872, 14.7373695, -23.9415245, 14.5907173, -38.5792046, 38.6788940
12: -44.5338974, 4.4961672, -44.5612411, 4.2829781, -45.2487335, 45.5240860
13: -37.5774384, 22.4010353, -37.4906960, 22.3284607, -59.6240387, 59.5995941
14: -65.1592712, 2.7495575, -65.2232513, 2.5559902, -67.7152634, 67.9728088
15: -21.9327888, 20.5401268, -21.7942772, 20.4275703, -42.3603592, 42.3344040
16: -23.5791817, 21.7455502, -23.6020660, 21.6973152, -45.2764969, 45.3476181
17: -58.6926422, -1.1294775, -58.6360245, -1.2902279, -56.1463928, 56.2940369
18: -35.9245529, 14.6647873, -36.0084610, 14.6282024, -50.5527573, 50.6732483
19: -26.5643482, 9.5149279, -26.5825825, 9.4456472, -36.0099945, 36.0975113
20: -21.6602974, 15.9165955, -21.6835461, 15.8408470, -37.5011444, 37.6001434
21: -27.4644547, 13.0100441, -27.4922104, 12.9198771, -40.3843307, 40.5022545
22: -32.1756477, 10.6705914, -32.1727180, 10.6316395, -42.8072891, 42.8433075
23: -24.6994267, 14.0624075, -24.6970234, 14.0094776, -38.7089043, 38.7594299
24: -30.7988777, 13.7586098, -30.7809505, 13.7326126, -44.5314903, 44.5395584
25: -28.9760113, 12.9622345, -28.9756966, 12.9116220, -41.8876343, 41.9379311
26: -41.1724854, 17.0820904, -41.2300797, 16.9992218, -58.1717072, 58.3121719
27: -26.1663361, 18.2407761, -26.1200523, 18.2365284, -44.4028625, 44.3608284
28: -25.1656590, 17.3406696, -25.1326370, 17.2895126, -42.4551697, 42.4733047
29: -27.7194118, 10.9736576, -27.7157288, 10.9022074, -38.4395599, 38.5117531
30: -26.9372025, 18.3379002, -26.9342690, 18.2974014, -45.2346039, 45.2721710
31: -35.5440063, 12.1501217, -35.5817795, 12.0716152, -47.6156235, 47.7319031
32: -35.3525925, 11.0498600, -35.3456116, 10.9750118, -45.8591309, 45.9499130
33: -63.7477646, -3.6561670, -63.6725197, -3.6114550, -55.5237579, 55.3938446
34: -57.8826904, -6.2437277, -57.8150215, -6.2589760, -47.7044678, 47.6632156
35: -56.1151924, -4.2877045, -56.0866547, -4.2232771, -45.0043030, 44.9012489
36: -53.5864601, 0.9323511, -53.5067329, 0.8911953, -49.4823608, 49.4641266
37: -78.3825912, -14.2170763, -78.3201447, -14.2841597, -60.8279419, 60.8525238
38: -63.9287491, 0.4719477, -63.8558426, 0.4249568, -59.7395477, 59.7250137
39: -72.2270889, -8.1036100, -72.1719055, -8.1215878, -58.1075287, 58.0916138
40: -51.4319916, -6.1298027, -51.4256897, -6.1654286, -45.2665634, 45.2958870
41: -40.1230278, 12.2916145, -40.0909996, 12.2773399, -52.4003677, 52.3826141
42: -26.2486992, 11.9862547, -26.2282791, 11.9521275, -38.2008286, 38.2145348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=261, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0909109
time: 59.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.1324431
time: 48.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 110.18 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1466596
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1881893
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0506689
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0922015
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0909109
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 110.18
Output dim: 5, lower bound: -20.1881894, upper bound: 20.1324431

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -37.7987366, 17.5782852, -37.7805595, 17.6785488, -55.4772873, 55.3588448
1: -12.0369883, 22.4716187, -12.0329199, 22.6672764, -34.7042656, 34.5045395
2: -9.7812557, 25.2787895, -9.8172131, 25.5283833, -35.3096390, 35.0960007
3: -9.6037416, 28.9544086, -9.6799507, 29.2720947, -38.6535263, 38.3978767
4: -16.6823292, 25.3578892, -16.7483940, 25.6952534, -42.2622910, 41.9830208
5: -7.4536366, 29.0177231, -7.5353618, 29.3051834, -36.3385353, 36.1112595
6: -38.2478027, 12.0175629, -38.4119568, 12.0786896, -50.3264923, 50.4295197
7: -11.1841412, 28.6545620, -11.2468605, 28.8138695, -38.8004608, 38.6682205
8: -21.3389511, 29.8654232, -21.3628979, 30.1656151, -51.0946808, 50.7742615
9: -13.8391237, 28.3023815, -13.9430218, 28.4079552, -42.2470779, 42.2454033
10: -22.2110481, 31.9481697, -22.5661812, 32.0783615, -54.2894096, 54.5143509
11: -23.7471275, 14.6332521, -24.2296295, 14.7409420, -38.4880676, 38.8628807
12: -44.2779312, 4.2905350, -44.8486786, 4.5373259, -45.2623978, 45.5996552
13: -37.5288544, 22.2695885, -37.6017227, 22.4763184, -59.7320175, 59.5668182
14: -64.9637299, 2.5442123, -65.5052948, 2.7707729, -67.7345047, 68.0495071
15: -21.8328018, 20.3619690, -21.9599209, 20.6361618, -42.4689636, 42.3218918
16: -23.5566101, 21.6860085, -23.7521057, 21.7589073, -45.3155174, 45.4381142
17: -58.4745216, -1.3145199, -58.9291916, -1.0906801, -56.1282654, 56.4222260
18: -35.8702545, 14.6920452, -36.0843124, 14.6819229, -50.5521774, 50.7763596
19: -26.4488106, 9.5290346, -26.7139778, 9.5283928, -35.9772034, 36.2430115
20: -21.5430260, 15.9090900, -21.8072319, 15.9258890, -37.4689140, 37.7163239
21: -27.3050499, 12.9839182, -27.6781693, 13.0327511, -40.3377991, 40.6620865
22: -32.1141090, 10.7026262, -32.2494888, 10.7071972, -42.8213043, 42.9521141
23: -24.5967388, 14.0613098, -24.8090515, 14.0800381, -38.6767769, 38.8703613
24: -30.7421417, 13.7923145, -30.8438301, 13.7676678, -44.5098114, 44.6361465
25: -28.8974400, 13.0000477, -29.0601654, 12.9913902, -41.8888321, 42.0602112
26: -41.0398712, 17.0595074, -41.3780060, 17.1144810, -58.1543503, 58.4375153
27: -26.0817738, 18.2658081, -26.2201309, 18.2953396, -44.3771133, 44.4859390
28: -25.0771675, 17.3916130, -25.2267189, 17.3549461, -42.4321136, 42.6183319
29: -27.6312294, 10.9625921, -27.8250179, 10.9946060, -38.4412689, 38.6116943
30: -26.8549252, 18.3393307, -27.0296154, 18.3514194, -45.2063446, 45.3689461
31: -35.4213486, 12.1833687, -35.7225571, 12.1594934, -47.5808411, 47.9059258
32: -35.2382126, 10.9897537, -35.4759903, 11.0784407, -45.8463440, 46.0195160
33: -63.6617241, -3.7077904, -63.7720947, -3.4859271, -55.5839844, 55.4154129
34: -57.7903023, -6.2832365, -57.9093323, -6.1122084, -47.7792206, 47.6896591
35: -56.0807190, -4.2578430, -56.1269073, -4.1455917, -45.0541916, 44.9539032
36: -53.4886322, 0.9614601, -53.6094551, 0.9930134, -49.4769363, 49.5962372
37: -78.2756577, -14.2884903, -78.4425735, -14.1853762, -60.8040924, 60.9071198
38: -63.8227463, 0.5209761, -63.9670029, 0.5487995, -59.7235641, 59.8973236
39: -72.1357422, -8.1480246, -72.2759018, -8.0248280, -58.0851440, 58.1713104
40: -51.3953018, -6.2068677, -51.5070763, -6.0753317, -45.3199692, 45.3002090
41: -40.0565796, 12.2603035, -40.1696014, 12.3362808, -52.3928604, 52.4299049
42: -26.1676064, 11.9338264, -26.3191872, 12.0194674, -38.1870728, 38.2530136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0922020, upper bound: 20.1881886
time: 67.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0922020, upper bound: 20.1881894
time: 69.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -37.6174278, 17.5313969, -37.6346893, 17.5653152, -55.1827431, 55.1660843
1: -11.9353199, 22.3968964, -11.9298401, 22.5475140, -34.4828339, 34.3267365
2: -9.7393265, 25.2442741, -9.6858788, 25.3721733, -35.1114998, 34.9301529
3: -9.6007175, 28.9068985, -9.5313454, 29.0490150, -38.4281845, 38.2029991
4: -16.6390343, 25.3548164, -16.5863800, 25.4682159, -41.9909286, 41.8234863
5: -7.4448781, 28.9847794, -7.3883162, 29.0816536, -36.1023026, 35.9432449
6: -38.2734222, 12.0009260, -38.2786522, 11.9772663, -50.2506866, 50.2795792
7: -11.1272345, 28.5345383, -11.1010838, 28.6835308, -38.6057892, 38.4220009
8: -21.2476616, 29.8159389, -21.1984272, 29.9714470, -50.8034973, 50.5775604
9: -13.7062149, 28.1946068, -13.8559752, 28.2702274, -41.9764404, 42.0505829
10: -22.1481190, 31.9391594, -22.4229736, 31.9279404, -54.0760574, 54.3621330
11: -23.8274078, 14.6810589, -23.8727245, 14.5753489, -38.4027557, 38.5537834
12: -44.4335632, 4.4286823, -44.5188980, 4.2613983, -45.1488800, 45.4078064
13: -37.5256653, 22.2393837, -37.4736786, 22.2552261, -59.4249573, 59.3760376
14: -64.9879303, 2.6564703, -65.1515961, 2.5205889, -67.5085220, 67.8080673
15: -21.8681755, 20.4285412, -21.7749233, 20.3803825, -42.2485580, 42.2034645
16: -23.4824257, 21.5983009, -23.5698586, 21.6299782, -45.1124039, 45.1681595
17: -58.5397148, -1.2372084, -58.5742760, -1.3261690, -55.9918518, 56.1129265
18: -35.7840958, 14.6010113, -35.9455948, 14.6108246, -50.3949203, 50.5466080
19: -26.3631382, 9.4687166, -26.4923153, 9.4358482, -35.7989883, 35.9610329
20: -21.4806862, 15.8750448, -21.6033707, 15.8303566, -37.3110428, 37.4784164
21: -27.2717800, 12.9744244, -27.4069080, 12.9114857, -40.1832657, 40.3813324
22: -31.9216042, 10.6185112, -32.0580902, 10.6177950, -42.5393982, 42.6766014
23: -24.4969597, 14.0053606, -24.6069450, 13.9967957, -38.4937553, 38.6123047
24: -30.5304966, 13.6985579, -30.6612148, 13.7182550, -44.2487526, 44.3597717
25: -28.7102318, 12.9008074, -28.8566742, 12.8973627, -41.6075935, 41.7574806
26: -40.9696312, 17.0228844, -41.1396866, 16.9861679, -57.9557991, 58.1625710
27: -25.9810352, 18.2055779, -26.0354557, 18.2261925, -44.2072296, 44.2410355
28: -24.9492149, 17.2875481, -25.0347481, 17.2775078, -42.2267227, 42.3222961
29: -27.4609051, 10.9248619, -27.6000290, 10.8895378, -38.1689301, 38.3455963
30: -26.7436199, 18.2818851, -26.8493900, 18.2817116, -45.0253296, 45.1312752
31: -35.2889824, 12.0848446, -35.4674797, 12.0564251, -47.3454056, 47.5523224
32: -35.2609367, 11.0133743, -35.3059769, 10.9611206, -45.7485809, 45.8686447
33: -63.6458435, -3.7052994, -63.6270790, -3.6284633, -55.4144821, 55.2958145
34: -57.7360306, -6.2968416, -57.7482414, -6.2734194, -47.5641479, 47.5382767
35: -55.9870911, -4.3323307, -56.0270920, -4.2355328, -44.8828659, 44.7906036
36: -53.4404144, 0.8988123, -53.4385185, 0.8821173, -49.3432617, 49.3595352
37: -78.2086411, -14.2616673, -78.2431641, -14.2959328, -60.6589813, 60.7215118
38: -63.7230225, 0.4041042, -63.7604599, 0.4033484, -59.5290298, 59.5548401
39: -72.0835724, -8.1351719, -72.1074753, -8.1346865, -57.9589996, 57.9910889
40: -51.3267479, -6.1778545, -51.3808289, -6.1843262, -45.1424217, 45.2029724
41: -40.0528107, 12.2576332, -40.0603600, 12.2650118, -52.3178215, 52.3179932
42: -26.1840782, 11.9500027, -26.2013168, 11.9377699, -38.1218491, 38.1513214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 761

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0261225
time: 55.49 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0482679
time: 53.49 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -37.8344116, 17.6133003, -37.6483688, 17.5950871, -55.4294968, 55.2616692
1: -12.0837364, 22.4574795, -11.9354172, 22.5725498, -34.6562881, 34.3928986
2: -9.8673038, 25.3125229, -9.6900797, 25.4006214, -35.2679253, 35.0026016
3: -9.6940756, 28.9591179, -9.5345078, 29.0679131, -38.5430832, 38.2597847
4: -16.7836151, 25.4049454, -16.5920067, 25.4875488, -42.1558838, 41.8803864
5: -7.5451961, 29.0516090, -7.3914309, 29.1077309, -36.2302017, 36.0131912
6: -38.3135414, 12.0787907, -38.2869377, 11.9852772, -50.2988205, 50.3657303
7: -11.2764769, 28.6116180, -11.1049690, 28.7162018, -38.7880859, 38.5027008
8: -21.4293938, 29.8899708, -21.2060699, 29.9985619, -51.0128250, 50.6548462
9: -13.8491774, 28.2654934, -13.8636751, 28.2989540, -42.1481323, 42.1291695
10: -22.2877693, 32.0181198, -22.4294472, 31.9543648, -54.2421341, 54.4475670
11: -23.9254646, 14.7491350, -23.8917618, 14.5809498, -38.5064163, 38.6408958
12: -44.4806442, 4.5119791, -44.5236816, 4.2716951, -45.2291336, 45.4922562
13: -37.6109467, 22.3270321, -37.4766502, 22.2822762, -59.5931702, 59.4596558
14: -65.1178055, 2.7365904, -65.1607132, 2.5471210, -67.6649246, 67.8973007
15: -21.9644661, 20.4737663, -21.7840652, 20.3854179, -42.3498840, 42.2578316
16: -23.6454239, 21.6941814, -23.5812454, 21.6694202, -45.3148422, 45.2754288
17: -58.6446266, -1.1760998, -58.5805054, -1.3084946, -56.1514740, 56.1791267
18: -35.8583221, 14.6999245, -35.9650307, 14.6158543, -50.4741745, 50.6649551
19: -26.4651775, 9.5762148, -26.5282497, 9.4377222, -35.9029007, 36.1044655
20: -21.5657806, 15.9691992, -21.6344337, 15.8342047, -37.3999863, 37.6036339
21: -27.3854084, 13.0749435, -27.4420319, 12.9141750, -40.2995834, 40.5169754
22: -32.0586700, 10.7422323, -32.1096649, 10.6219749, -42.6806450, 42.8518982
23: -24.5750141, 14.0914345, -24.6347198, 14.0005713, -38.5755844, 38.7261543
24: -30.6274490, 13.7855425, -30.6985378, 13.7220020, -44.3494492, 44.4840813
25: -28.8232059, 13.0380497, -28.9023495, 12.9026995, -41.7259064, 41.9403992
26: -41.0793037, 17.1329746, -41.1758041, 16.9889774, -58.0682831, 58.3087769
27: -26.0874157, 18.3035278, -26.0724373, 18.2285080, -44.3159256, 44.3759651
28: -25.0558891, 17.4195728, -25.0767727, 17.2806473, -42.3365364, 42.4963455
29: -27.5915680, 11.0228424, -27.6465797, 10.8936167, -38.3062057, 38.4910660
30: -26.8161850, 18.3607063, -26.8716679, 18.2878761, -45.1040611, 45.2323761
31: -35.4102364, 12.2256012, -35.5109062, 12.0612602, -47.4714966, 47.7365074
32: -35.3123207, 11.0760727, -35.3190765, 10.9660454, -45.8107224, 45.9429855
33: -63.7218857, -3.6077852, -63.6493645, -3.6219692, -55.4909668, 55.4169388
34: -57.8203163, -6.1720190, -57.7795105, -6.2688646, -47.6483612, 47.6973267
35: -56.0815735, -4.2086601, -56.0636063, -4.2315350, -44.9723587, 44.9523163
36: -53.5419312, 1.0419245, -53.4780045, 0.8849411, -49.4382019, 49.5472183
37: -78.2840576, -14.2187748, -78.2659302, -14.2919683, -60.7405090, 60.7928696
38: -63.8648682, 0.6048131, -63.8142624, 0.4120197, -59.6639023, 59.8149490
39: -72.1686554, -8.0617666, -72.1313629, -8.1297121, -58.0450134, 58.0902863
40: -51.4240417, -6.1312981, -51.3993912, -6.1817636, -45.2422791, 45.2680931
41: -40.1023407, 12.3030052, -40.0744095, 12.2675419, -52.3698807, 52.3774147
42: -26.2188873, 11.9896812, -26.2102089, 11.9411564, -38.1600418, 38.1998901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 761

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0676634
time: 66.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0898158
time: 133.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -37.6933441, 17.5661087, -37.6549225, 17.5789719, -55.2723160, 55.2210312
1: -11.9796572, 22.4731922, -11.9357624, 22.5799751, -34.5596313, 34.4089546
2: -9.7736244, 25.3101711, -9.6918554, 25.4001446, -35.1737671, 35.0020256
3: -9.6487122, 29.0462799, -9.5381870, 29.1088696, -38.5363922, 38.3426781
4: -16.6863441, 25.4672737, -16.5933914, 25.5162277, -42.0860519, 41.9365425
5: -7.4892902, 29.0990334, -7.3958006, 29.1306953, -36.1953888, 36.0560493
6: -38.3151550, 12.0255623, -38.2947655, 11.9848366, -50.2999916, 50.3203278
7: -11.1710320, 28.6327858, -11.1068325, 28.7257557, -38.6918411, 38.5138321
8: -21.3043346, 29.9159508, -21.2070541, 30.0137196, -50.9033356, 50.6782455
9: -13.7638922, 28.2963448, -13.8674059, 28.3144035, -42.0782967, 42.1637497
10: -22.1935749, 31.9781075, -22.4372807, 31.9409447, -54.1345215, 54.4153900
11: -23.9404297, 14.7187586, -23.9179497, 14.5817413, -38.5221710, 38.6367073
12: -44.5201645, 4.4659567, -44.5545731, 4.2681637, -45.2155151, 45.4818344
13: -37.5601196, 22.3360538, -37.4823112, 22.2967167, -59.5552902, 59.4930725
14: -65.1241455, 2.6844254, -65.2062607, 2.5243921, -67.6485367, 67.8906860
15: -21.9063053, 20.5247936, -21.7814312, 20.4200745, -42.3263779, 42.3062248
16: -23.5471420, 21.6557045, -23.5865364, 21.6537056, -45.2008476, 45.2422409
17: -58.6683540, -1.1725121, -58.6243782, -1.3112164, -56.0947723, 56.2300644
18: -35.8762665, 14.6491203, -35.9847755, 14.6206131, -50.4968796, 50.6338959
19: -26.4818554, 9.5060577, -26.5425205, 9.4413118, -35.9231682, 36.0485764
20: -21.5862522, 15.9026623, -21.6476345, 15.8340597, -37.4203110, 37.5502968
21: -27.3819752, 12.9985971, -27.4521122, 12.9142838, -40.2962570, 40.4507103
22: -32.0585670, 10.6566744, -32.1161041, 10.6247425, -42.6833115, 42.7727776
23: -24.6341801, 14.0483370, -24.6651630, 14.0026731, -38.6368523, 38.7135010
24: -30.7125072, 13.7438192, -30.7389603, 13.7253351, -44.4378433, 44.4827805
25: -28.8728561, 12.9451990, -28.9257126, 12.9033613, -41.7762184, 41.8709106
26: -41.0845184, 17.0686855, -41.1874542, 16.9927177, -58.0772362, 58.2561417
27: -26.0801888, 18.2296333, -26.0782032, 18.2311134, -44.3113022, 44.3078384
28: -25.0691566, 17.3276787, -25.0858574, 17.2831554, -42.3523102, 42.4135361
29: -27.6137428, 10.9595633, -27.6646080, 10.8953142, -38.3241730, 38.4451141
30: -26.8819580, 18.3184414, -26.9071350, 18.2878971, -45.1698532, 45.2255783
31: -35.4448166, 12.1333237, -35.5335770, 12.0634203, -47.5082359, 47.6669006
32: -35.3197212, 11.0334644, -35.3296852, 10.9670877, -45.8146515, 45.9143448
33: -63.6907959, -3.6763854, -63.6447906, -3.6212215, -55.4568405, 55.3453979
34: -57.8064613, -6.2585268, -57.7779465, -6.2660952, -47.6176605, 47.6081924
35: -56.0310211, -4.3013964, -56.0455437, -4.2299738, -44.9110184, 44.8457222
36: -53.4947014, 0.9223261, -53.4623337, 0.8863659, -49.3828125, 49.4077377
37: -78.3264160, -14.2305984, -78.2927475, -14.2907352, -60.7539368, 60.8041229
38: -63.8011703, 0.4475164, -63.7941246, 0.4131742, -59.5928345, 59.6338348
39: -72.1616669, -8.1203299, -72.1400452, -8.1296415, -58.0304489, 58.0401077
40: -51.3845139, -6.1536856, -51.4026222, -6.1770296, -45.2074852, 45.2489357
41: -40.0879745, 12.2781706, -40.0739136, 12.2708569, -52.3588333, 52.3520851
42: -26.2251110, 11.9673929, -26.2167645, 11.9430437, -38.1681557, 38.1841583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1479475, upper bound: 20.0909106
time: 53.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1142022, upper bound: 20.0909109
time: 50.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -37.9102287, 17.6480064, -37.6686211, 17.6087265, -55.5189552, 55.3166275
1: -12.1280937, 22.5337753, -11.9413013, 22.6050034, -34.7330971, 34.4750748
2: -9.9015532, 25.3784275, -9.6960526, 25.4285984, -35.3301506, 35.0744781
3: -9.7420444, 29.0985565, -9.5413895, 29.1277485, -38.6513290, 38.3995323
4: -16.8308735, 25.5174255, -16.5989685, 25.5355606, -42.2509689, 41.9934311
5: -7.5896406, 29.1659775, -7.3989644, 29.1567917, -36.3233261, 36.1260452
6: -38.3552933, 12.1033592, -38.3030586, 11.9928551, -50.3481483, 50.4064178
7: -11.3202534, 28.7098942, -11.1106558, 28.7584286, -38.8741302, 38.5945206
8: -21.4860535, 29.9899712, -21.2147026, 30.0408192, -51.1126251, 50.7555084
9: -13.9067955, 28.3672218, -13.8751144, 28.3431606, -42.2499542, 42.2423363
10: -22.3331928, 32.0571365, -22.4437389, 31.9673462, -54.3005371, 54.5008774
11: -24.0384045, 14.7867737, -23.9369850, 14.5872841, -38.6256866, 38.7237587
12: -44.5672150, 4.5492773, -44.5593224, 4.2785063, -45.2957001, 45.5662842
13: -37.6453743, 22.4236069, -37.4852448, 22.3237572, -59.7235031, 59.5767365
14: -65.2538757, 2.7645626, -65.2154160, 2.5509510, -67.8048248, 67.9799805
15: -22.0025272, 20.5699348, -21.7905540, 20.4251137, -42.4276428, 42.3604889
16: -23.7101078, 21.7516270, -23.5979042, 21.6931305, -45.4032364, 45.3495331
17: -58.7731743, -1.1114588, -58.6306534, -1.2934952, -56.2542419, 56.2962570
18: -35.9504967, 14.7479820, -36.0041885, 14.6256638, -50.5761604, 50.7521706
19: -26.5838871, 9.6135473, -26.5784817, 9.4431658, -36.0270538, 36.1920280
20: -21.6713142, 15.9968081, -21.6787262, 15.8379393, -37.5092545, 37.6755333
21: -27.4955711, 13.0991297, -27.4872494, 12.9169416, -40.4125137, 40.5863800
22: -32.1957283, 10.7803860, -32.1676865, 10.6289339, -42.8246613, 42.9480743
23: -24.7122135, 14.1343765, -24.6929207, 14.0064392, -38.7186508, 38.8272972
24: -30.8094902, 13.8307333, -30.7762794, 13.7291164, -44.5386047, 44.6070137
25: -28.9858112, 13.0824385, -28.9713783, 12.9087191, -41.8945312, 42.0538177
26: -41.1942253, 17.1788006, -41.2236023, 16.9955177, -58.1897430, 58.4024048
27: -26.1866379, 18.3275471, -26.1152039, 18.2334652, -44.4201050, 44.4427490
28: -25.1757908, 17.4596882, -25.1278648, 17.2863083, -42.4620972, 42.5875549
29: -27.7444477, 11.0575132, -27.7111588, 10.8993807, -38.4615021, 38.5905151
30: -26.9544945, 18.3972340, -26.9294281, 18.2940483, -45.2485428, 45.3266602
31: -35.5660133, 12.2740402, -35.5770226, 12.0682583, -47.6342697, 47.8510628
32: -35.3711166, 11.0961628, -35.3427582, 10.9720116, -45.8767548, 45.9887695
33: -63.7670708, -3.5788827, -63.6670837, -3.6147356, -55.5335083, 55.4665909
34: -57.8907242, -6.1336708, -57.8092194, -6.2615309, -47.7018509, 47.7672272
35: -56.1255989, -4.1777210, -56.0821037, -4.2259407, -45.0006180, 45.0074730
36: -53.5962410, 1.0654612, -53.5018234, 0.8892107, -49.4777679, 49.5954056
37: -78.4020615, -14.1876850, -78.3154755, -14.2867584, -60.8356323, 60.8754578
38: -63.9430466, 0.6481538, -63.8479195, 0.4218173, -59.7277374, 59.8938980
39: -72.2470169, -8.0469036, -72.1639404, -8.1246872, -58.1167068, 58.1393127
40: -51.4817543, -6.1071873, -51.4211464, -6.1744475, -45.3073082, 45.3139572
41: -40.1375847, 12.3235645, -40.0880051, 12.2733803, -52.4109650, 52.4115677
42: -26.2599545, 12.0070858, -26.2256279, 11.9464302, -38.2063828, 38.2327118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=261, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1142022, upper bound: 20.1324429
time: 72.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.1479475, upper bound: 20.1324432
time: 58.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 133.69 seconds
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.0922020, upper bound: 20.1881886
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.0922020, upper bound: 20.1881894
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0261225
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0482679
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0676634
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1520775, upper bound: 20.0898158
IS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1479475, upper bound: 20.0909106
IS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1142022, upper bound: 20.0909109
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1142022, upper bound: 20.1324429
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 133.69
Output dim: 5, lower bound: -20.1479475, upper bound: 20.1324432

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -37.7987366, 17.5782852, -37.7037582, 17.6430817, -55.4418182, 55.2820435
1: -12.0369883, 22.4716187, -11.9882641, 22.5896835, -34.6266708, 34.4598846
2: -9.7812557, 25.2787895, -9.7826767, 25.4613590, -35.2426147, 35.0614662
3: -9.6037416, 28.9544086, -9.6316204, 29.1303444, -38.5108719, 38.3569756
4: -16.6823292, 25.3578892, -16.7007217, 25.5809422, -42.1474838, 41.9423981
5: -7.4536366, 29.0177231, -7.4904690, 29.1889248, -36.2216759, 36.0763741
6: -38.2478027, 12.0175629, -38.3694801, 12.0538197, -50.3016205, 50.3870430
7: -11.1841412, 28.6545620, -11.2026730, 28.7139568, -38.7004089, 38.6362419
8: -21.3389511, 29.8654232, -21.3057785, 30.0639191, -50.9911499, 50.7265091
9: -13.8391237, 28.3023815, -13.8849411, 28.3044739, -42.1435966, 42.1873245
10: -22.2110481, 31.9481697, -22.5201473, 32.0387230, -54.2461853, 54.4683151
11: -23.7471275, 14.6332521, -24.1148815, 14.7029552, -38.4500809, 38.7481346
12: -44.2779312, 4.2905350, -44.7606888, 4.4995852, -45.2500076, 45.5117493
13: -37.5288544, 22.2695885, -37.5668297, 22.3764992, -59.5874329, 59.5208130
14: -64.9637299, 2.5442123, -65.3666992, 2.7425499, -67.7062836, 67.9109116
15: -21.8328018, 20.3619690, -21.9214439, 20.5383492, -42.3711510, 42.2834129
16: -23.5566101, 21.6860085, -23.6868229, 21.7004089, -45.2570190, 45.3728333
17: -58.4745216, -1.3145199, -58.7985687, -1.1560669, -56.1058731, 56.2881355
18: -35.8702545, 14.6920452, -35.9905472, 14.6333046, -50.5035591, 50.6825943
19: -26.4488106, 9.5290346, -26.5932293, 9.4907742, -35.9395828, 36.1222649
20: -21.5430260, 15.9090900, -21.6999264, 15.8981056, -37.4411316, 37.6090164
21: -27.3050499, 12.9839182, -27.5661602, 13.0084152, -40.3134651, 40.5500793
22: -32.1141090, 10.7026262, -32.1102142, 10.6687260, -42.7828369, 42.8128395
23: -24.5967388, 14.0613098, -24.6695271, 14.0367489, -38.6334877, 38.7308350
24: -30.7421417, 13.7923145, -30.6587257, 13.7220507, -44.4641914, 44.4510422
25: -28.8974400, 13.0000477, -28.8948689, 12.9466219, -41.8440628, 41.8949165
26: -41.0398712, 17.0595074, -41.2611504, 17.0683250, -58.1081963, 58.3206558
27: -26.0817738, 18.2658081, -26.1190987, 18.2711468, -44.3529205, 44.3849068
28: -25.0771675, 17.3916130, -25.1047668, 17.3145370, -42.3917046, 42.4963799
29: -27.6312294, 10.9625921, -27.6696339, 10.9596577, -38.4105606, 38.4555359
30: -26.8549252, 18.3393307, -26.8889618, 18.3145599, -45.1694870, 45.2282944
31: -35.4213486, 12.1833687, -35.5640869, 12.1106930, -47.5320435, 47.7474556
32: -35.2382126, 10.9897537, -35.4157333, 11.0580416, -45.8248596, 45.9579773
33: -63.6617241, -3.7077904, -63.7254295, -3.5151024, -55.5620804, 55.3695831
34: -57.7903023, -6.2832365, -57.8376770, -6.1508255, -47.7639008, 47.6177063
35: -56.0807190, -4.2578430, -56.0820694, -4.1767721, -45.0388107, 44.9109344
36: -53.4886322, 0.9614601, -53.5536842, 0.9692259, -49.4683990, 49.5427551
37: -78.2756577, -14.2884903, -78.3227158, -14.2166948, -60.8000336, 60.7865982
38: -63.8227463, 0.5209761, -63.8865891, 0.5049725, -59.7032471, 59.8160553
39: -72.1357422, -8.1480246, -72.1950378, -8.0399342, -58.0776596, 58.0925293
40: -51.3953018, -6.2068677, -51.4483871, -6.1000962, -45.2952042, 45.2415199
41: -40.0565796, 12.2603035, -40.1336136, 12.3154631, -52.3720436, 52.3939171
42: -26.1676064, 11.9338264, -26.2771072, 12.0018110, -38.1694183, 38.2109337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=260, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858043
time: 61.50 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0898163, upper bound: 20.1858038
time: 61.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -37.7987366, 17.5782852, -37.7796326, 17.6778107, -55.4765472, 55.3579178
1: -12.0369883, 22.4716187, -12.0326385, 22.6660099, -34.7029991, 34.5042572
2: -9.7812557, 25.2787895, -9.8168993, 25.5272789, -35.3085327, 35.0956879
3: -9.6037416, 28.9544086, -9.6795769, 29.2697868, -38.6437988, 38.3974762
4: -16.6823292, 25.3578892, -16.7479916, 25.6934128, -42.2537766, 41.9826355
5: -7.4536366, 29.0177231, -7.5349154, 29.3032818, -36.3273468, 36.1108170
6: -38.2478027, 12.0175629, -38.4112167, 12.0783443, -50.3261490, 50.4287796
7: -11.1841412, 28.6545620, -11.2464752, 28.8122196, -38.7865601, 38.6678810
8: -21.3389511, 29.8654232, -21.3624001, 30.1639557, -51.0835037, 50.7737732
9: -13.8391237, 28.3023815, -13.9424868, 28.4062195, -42.2453423, 42.2448692
10: -22.2110481, 31.9481697, -22.5655651, 32.0777359, -54.2887840, 54.5137329
11: -23.7471275, 14.6332521, -24.2278843, 14.7406654, -38.4877930, 38.8611374
12: -44.2779312, 4.2905350, -44.8472824, 4.5368729, -45.2619171, 45.5713882
13: -37.5288544, 22.2695885, -37.6012650, 22.4731445, -59.7296600, 59.5944595
14: -64.9637299, 2.5442123, -65.5029297, 2.7704782, -67.7342072, 68.0471420
15: -21.8328018, 20.3619690, -21.9595547, 20.6346073, -42.4674072, 42.3215256
16: -23.5566101, 21.6860085, -23.7513657, 21.7578773, -45.3144875, 45.4373741
17: -58.4745216, -1.3145199, -58.9272003, -1.0914345, -56.1274948, 56.3760223
18: -35.8702545, 14.6920452, -36.0826912, 14.6813745, -50.5516281, 50.7747345
19: -26.4488106, 9.5290346, -26.7119827, 9.5281410, -35.9769516, 36.2410164
20: -21.5430260, 15.9090900, -21.8054810, 15.9257050, -37.4687309, 37.7145691
21: -27.3050499, 12.9839182, -27.6763515, 13.0326138, -40.3376617, 40.6602707
22: -32.1141090, 10.7026262, -32.2472038, 10.7068691, -42.8209763, 42.9498291
23: -24.5967388, 14.0613098, -24.8067989, 14.0797443, -38.6764832, 38.8681107
24: -30.7421417, 13.7923145, -30.8407841, 13.7672939, -44.5094376, 44.6330986
25: -28.8974400, 13.0000477, -29.0575047, 12.9910269, -41.8884659, 42.0575523
26: -41.0398712, 17.0595074, -41.3760910, 17.1141624, -58.1540337, 58.4356003
27: -26.0817738, 18.2658081, -26.2183094, 18.2951488, -44.3769226, 44.4841156
28: -25.0771675, 17.3916130, -25.2247143, 17.3546638, -42.4318314, 42.6163254
29: -27.6312294, 10.9625921, -27.8224678, 10.9943342, -38.4410019, 38.6050034
30: -26.8549252, 18.3393307, -27.0272865, 18.3511009, -45.2060242, 45.3666153
31: -35.4213486, 12.1833687, -35.7199173, 12.1591358, -47.5804825, 47.9032860
32: -35.2382126, 10.9897537, -35.4745216, 11.0781641, -45.8460388, 46.0176926
33: -63.6617241, -3.7077904, -63.7705307, -3.4863005, -55.5836182, 55.4043884
34: -57.7903023, -6.2832365, -57.9081154, -6.1125498, -47.7788544, 47.6635513
35: -56.0807190, -4.2578430, -56.1260757, -4.1458626, -45.0538940, 44.9329643
36: -53.4886322, 0.9614601, -53.6081161, 0.9927397, -49.4766998, 49.5779114
37: -78.2756577, -14.2884903, -78.4405899, -14.1856794, -60.8037872, 60.8758621
38: -63.8227463, 0.5209761, -63.9648705, 0.5482750, -59.7230759, 59.8698120
39: -72.1357422, -8.1480246, -72.2734222, -8.0250998, -58.0848541, 58.1584244
40: -51.3953018, -6.2068677, -51.5060310, -6.0759006, -45.3194008, 45.2991638
41: -40.0565796, 12.2603035, -40.1688461, 12.3359890, -52.3925705, 52.4291496
42: -26.1676064, 11.9338264, -26.3181667, 12.0191870, -38.1867943, 38.2519913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=260, inp2_unstable=260, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 761

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858046
time: 72.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858046
time: 53.30 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 128.16 seconds
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 128.16
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858043
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 128.16
Output dim: 5, lower bound: -20.0898163, upper bound: 20.1858038
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 128.16
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858046
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 128.16
Output dim: 5, lower bound: -20.0676639, upper bound: 20.1858046

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -37.6791039, 17.4726448, -37.6528091, 17.5924911, -55.2715950, 55.1254539
1: -11.9194660, 22.4295940, -11.9375372, 22.5797367, -34.4992027, 34.3671303
2: -9.6672783, 25.2319012, -9.7288647, 25.4530945, -35.1203728, 34.9607658
3: -9.4965191, 28.9105759, -9.5814981, 29.1204758, -38.3936844, 38.2595520
4: -16.5230865, 25.3051434, -16.6251678, 25.5711823, -41.9783478, 41.8138390
5: -7.3548560, 28.9733868, -7.4453611, 29.1804867, -36.1124191, 35.9853096
6: -38.1968384, 11.9081535, -38.3550568, 12.0013847, -50.1982231, 50.2632103
7: -11.0735912, 28.6190281, -11.1564980, 28.7059536, -38.5824661, 38.5544281
8: -21.1053963, 29.7749214, -21.1922283, 30.0503139, -50.7457581, 50.5250092
9: -13.7489834, 28.2668915, -13.8420219, 28.2952957, -42.0442810, 42.1089134
10: -22.1275253, 31.8989620, -22.4860764, 32.0181389, -54.1268616, 54.3850403
11: -23.6181087, 14.4773674, -24.0922737, 14.6270409, -38.2451477, 38.5696411
12: -44.2357941, 4.1640110, -44.7524338, 4.4410849, -45.1491928, 45.3784027
13: -37.3950462, 22.1600494, -37.5043221, 22.3452091, -59.4178619, 59.3391190
14: -64.8858643, 2.4864159, -65.3332214, 2.7172403, -67.6031036, 67.8196411
15: -21.6798820, 20.2947941, -21.8501625, 20.5180740, -42.1979561, 42.1449585
16: -23.4770203, 21.6306343, -23.6607037, 21.6739330, -45.1509552, 45.2913361
17: -58.3651733, -1.4077797, -58.7759285, -1.2000389, -55.9426651, 56.1722183
18: -35.7636108, 14.5071869, -35.9716644, 14.5424786, -50.3060913, 50.4788513
19: -26.3813324, 9.4599667, -26.5775242, 9.4566231, -35.8379555, 36.0374908
20: -21.4823837, 15.8357573, -21.6804581, 15.8637447, -37.3461304, 37.5162163
21: -27.2116985, 12.8855219, -27.5447636, 12.9609604, -40.1726608, 40.4302864
22: -32.0453262, 10.6362438, -32.0930405, 10.6371346, -42.6824608, 42.7292862
23: -24.5307617, 13.9519033, -24.6539936, 13.9844074, -38.5151672, 38.6058960
24: -30.6502743, 13.6798840, -30.6373901, 13.6674004, -44.3176727, 44.3172760
25: -28.8428383, 12.9255991, -28.8787479, 12.9115410, -41.7543793, 41.8043480
26: -40.9420090, 16.9350338, -41.2391930, 17.0074463, -57.9494553, 58.1742249
27: -26.0061455, 18.1984806, -26.0927086, 18.2391701, -44.2453156, 44.2911911
28: -25.0066833, 17.2811737, -25.0873871, 17.2610950, -42.2677765, 42.3685608
29: -27.5507050, 10.8731623, -27.6522408, 10.9160738, -38.2851791, 38.3479233
30: -26.7482071, 18.1750107, -26.8638535, 18.2350731, -44.9832802, 45.0388641
31: -35.3348846, 12.0659962, -35.5429764, 12.0532703, -47.3881531, 47.6089706
32: -35.1912384, 10.8982716, -35.4006042, 11.0175266, -45.7366257, 45.8492584
33: -63.6216774, -3.7878609, -63.7039185, -3.5498204, -55.4953918, 55.2829514
34: -57.7072678, -6.4287672, -57.8205452, -6.2139521, -47.6205292, 47.4547043
35: -56.0136986, -4.3751106, -56.0660629, -4.2270012, -44.9228058, 44.7788010
36: -53.4479141, 0.8886642, -53.5440712, 0.9380083, -49.3907166, 49.4505615
37: -78.1966858, -14.4064960, -78.3042679, -14.2702246, -60.6665039, 60.6485748
38: -63.7664452, 0.4351649, -63.8700523, 0.4721513, -59.6068726, 59.6989059
39: -72.0642395, -8.2073078, -72.1652679, -8.0493183, -57.9939270, 57.9879303
40: -51.3360138, -6.3049974, -51.4254379, -6.1397157, -45.1962967, 45.1204414
41: -40.0075455, 12.1686020, -40.1187019, 12.2718849, -52.2794304, 52.2873039
42: -26.1265049, 11.8900127, -26.2598820, 11.9838629, -38.1103668, 38.1498947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=260, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1329423
time: 51.55 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0655541, upper bound: 20.1837211
time: 45.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -37.7929764, 17.5754089, -37.7014236, 17.6418514, -55.4348297, 55.2768326
1: -12.0345192, 22.4707413, -11.9872494, 22.5893211, -34.6238403, 34.4579926
2: -9.7784653, 25.2776642, -9.7815342, 25.4608955, -35.2393608, 35.0591965
3: -9.6009922, 28.9526863, -9.6304398, 29.1296272, -38.5066071, 38.3572731
4: -16.6786633, 25.3558273, -16.6992359, 25.5801506, -42.1429367, 41.9389267
5: -7.4509511, 29.0169945, -7.4893751, 29.1886539, -36.2162247, 36.0743790
6: -38.2461281, 12.0149260, -38.3687630, 12.0527630, -50.2988892, 50.3836899
7: -11.1814499, 28.6531982, -11.2015533, 28.7134075, -38.6937332, 38.6337662
8: -21.3333664, 29.8632870, -21.3034439, 30.0630474, -50.9772186, 50.7220993
9: -13.8367586, 28.3018379, -13.8839712, 28.3042393, -42.1409988, 42.1858101
10: -22.2085953, 31.9467564, -22.5191708, 32.0381622, -54.2418823, 54.4659271
11: -23.7451019, 14.6296387, -24.1140137, 14.7014894, -38.4465904, 38.7436523
12: -44.2770119, 4.2869415, -44.7602921, 4.4981136, -45.2472382, 45.5020370
13: -37.5257530, 22.2673740, -37.5656166, 22.3755798, -59.5804291, 59.5298462
14: -64.9568558, 2.5424433, -65.3639221, 2.7418232, -67.6986771, 67.9063644
15: -21.8291683, 20.3597240, -21.9199600, 20.5374298, -42.3666000, 42.2796860
16: -23.5523491, 21.6813450, -23.6850739, 21.6984711, -45.2508202, 45.3664169
17: -58.4723396, -1.3171072, -58.7976913, -1.1571703, -56.1027832, 56.2842674
18: -35.8689613, 14.6874542, -35.9899902, 14.6314354, -50.5003967, 50.6774445
19: -26.4463768, 9.5274086, -26.5922432, 9.4901056, -35.9364815, 36.1196518
20: -21.5414581, 15.9071264, -21.6992378, 15.8972111, -37.4386673, 37.6063652
21: -27.3029690, 12.9817152, -27.5653305, 13.0074863, -40.3104553, 40.5470467
22: -32.1127281, 10.7010078, -32.1096344, 10.6680365, -42.7807655, 42.8106422
23: -24.5945187, 14.0587883, -24.6686020, 14.0356941, -38.6302109, 38.7273903
24: -30.7403011, 13.7895098, -30.6579590, 13.7208805, -44.4611816, 44.4474678
25: -28.8956127, 12.9982557, -28.8941345, 12.9458637, -41.8414764, 41.8923912
26: -41.0383301, 17.0555229, -41.2605591, 17.0667267, -58.1050568, 58.3160820
27: -26.0799980, 18.2642002, -26.1183491, 18.2705078, -44.3505058, 44.3825493
28: -25.0751019, 17.3890438, -25.1039257, 17.3135147, -42.3886185, 42.4929695
29: -27.6287918, 10.9605217, -27.6686382, 10.9587898, -38.4072189, 38.4518776
30: -26.8530388, 18.3354225, -26.8881588, 18.3129845, -45.1660233, 45.2235794
31: -35.4178009, 12.1805096, -35.5626602, 12.1095266, -47.5273285, 47.7431717
32: -35.2368965, 10.9877472, -35.4152107, 11.0572844, -45.8226547, 45.9483719
33: -63.6600761, -3.7100105, -63.7247696, -3.5161424, -55.5548325, 55.3724289
34: -57.7891464, -6.2864771, -57.8371887, -6.1521730, -47.7612152, 47.5950089
35: -56.0792732, -4.2604284, -56.0814781, -4.1778584, -45.0362930, 44.8921585
36: -53.4878998, 0.9598637, -53.5533829, 0.9685650, -49.4665451, 49.5336151
37: -78.2725983, -14.2914753, -78.3214798, -14.2179117, -60.7956848, 60.7791290
38: -63.8216057, 0.5191417, -63.8861427, 0.5042119, -59.7004929, 59.8072281
39: -72.1333923, -8.1491032, -72.1940918, -8.0403833, -58.0716934, 58.1058350
40: -51.3933449, -6.2093654, -51.4475670, -6.1013145, -45.2920303, 45.2382011
41: -40.0548134, 12.2579403, -40.1329002, 12.3144989, -52.3693123, 52.3908386
42: -26.1655617, 11.9289064, -26.2762661, 11.9998159, -38.1653786, 38.2051735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=260, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837216
time: 47.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0877306, upper bound: 20.1837211
time: 54.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -37.6791039, 17.4726448, -37.7286453, 17.6272469, -55.3063507, 55.2012901
1: -11.9194660, 22.4295940, -11.9819021, 22.6560421, -34.5755081, 34.4114952
2: -9.6672783, 25.2319012, -9.7631149, 25.5190315, -35.1863098, 34.9950180
3: -9.4965191, 28.9105759, -9.6294384, 29.2599087, -38.5266418, 38.3000870
4: -16.5230865, 25.3051434, -16.6724415, 25.6836548, -42.0846481, 41.8540726
5: -7.3548560, 28.9733868, -7.4898386, 29.2948284, -36.2181015, 36.0197945
6: -38.1968384, 11.9081535, -38.3967667, 12.0259190, -50.2227554, 50.3049202
7: -11.0735912, 28.6190281, -11.2002907, 28.8042221, -38.6686172, 38.5860367
8: -21.1053963, 29.7749214, -21.2488689, 30.1503696, -50.8381424, 50.5722656
9: -13.7489834, 28.2668915, -13.8995762, 28.3970604, -42.1460419, 42.1664658
10: -22.1275253, 31.8989620, -22.5314617, 32.0570984, -54.1752167, 54.4304237
11: -23.6181087, 14.4773674, -24.2052593, 14.6647444, -38.2828522, 38.6826248
12: -44.2357941, 4.1640110, -44.8390350, 4.4783077, -45.1610718, 45.4380341
13: -37.3950462, 22.1600494, -37.5386810, 22.4418716, -59.5601349, 59.4127655
14: -64.8858643, 2.4864159, -65.4694748, 2.7451630, -67.6310272, 67.9558868
15: -21.6798820, 20.2947941, -21.8882599, 20.6143417, -42.2942238, 42.1830521
16: -23.4770203, 21.6306343, -23.7252331, 21.7314301, -45.2084503, 45.3558655
17: -58.3651733, -1.4077797, -58.9046173, -1.1353874, -55.9642487, 56.2600708
18: -35.7636108, 14.5071869, -36.0638695, 14.5904980, -50.3541107, 50.5710564
19: -26.3813324, 9.4599667, -26.6962643, 9.4939995, -35.8753319, 36.1562309
20: -21.4823837, 15.8357573, -21.7860527, 15.8913336, -37.3737183, 37.6218109
21: -27.2116985, 12.8855219, -27.6549683, 12.9851189, -40.1968155, 40.5404892
22: -32.0453262, 10.6362438, -32.2301140, 10.6752853, -42.7206116, 42.8663559
23: -24.5307617, 13.9519033, -24.7912750, 14.0273905, -38.5581512, 38.7431793
24: -30.6502743, 13.6798840, -30.8194962, 13.7126026, -44.3628769, 44.4993820
25: -28.8428383, 12.9255991, -29.0413818, 12.9559593, -41.7987976, 41.9669800
26: -40.9420090, 16.9350338, -41.3541336, 17.0532017, -57.9952087, 58.2891693
27: -26.0061455, 18.1984806, -26.1919365, 18.2631664, -44.2693100, 44.3904190
28: -25.0066833, 17.2811737, -25.2073631, 17.3012218, -42.3079071, 42.4885368
29: -27.5507050, 10.8731623, -27.8051033, 10.9507256, -38.3155899, 38.4973755
30: -26.7482071, 18.1750107, -27.0021954, 18.2715912, -45.0197983, 45.1772079
31: -35.3348846, 12.0659962, -35.6987991, 12.1017885, -47.4366722, 47.7647934
32: -35.1912384, 10.8982716, -35.4593353, 11.0375767, -45.7577667, 45.9089966
33: -63.6216774, -3.7878609, -63.7490234, -3.5209465, -55.5167542, 55.3177872
34: -57.7072678, -6.4287672, -57.8910332, -6.1756897, -47.6354752, 47.5005875
35: -56.0136986, -4.3751106, -56.1100540, -4.1960812, -44.9376984, 44.8008423
36: -53.4479141, 0.8886642, -53.5983925, 0.9615650, -49.3989792, 49.4857559
37: -78.1966858, -14.4064960, -78.4220886, -14.2391672, -60.6702576, 60.7378616
38: -63.7664452, 0.4351649, -63.9483147, 0.5155206, -59.6266403, 59.7526550
39: -72.0642395, -8.2073078, -72.2436218, -8.0345154, -58.0010834, 58.0538559
40: -51.3360138, -6.3049974, -51.4830856, -6.1155424, -45.2204704, 45.1780891
41: -40.0075455, 12.1686020, -40.1538849, 12.2923489, -52.2998962, 52.3224869
42: -26.1265049, 11.8900127, -26.3009396, 12.0012054, -38.1277084, 38.1909523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=260, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837214
time: 43.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0655541, upper bound: 20.1837214
time: 56.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -37.7929764, 17.5754089, -37.7772980, 17.6765556, -55.4695320, 55.3527069
1: -12.0345192, 22.4707413, -12.0316153, 22.6656418, -34.7001610, 34.5023575
2: -9.7784653, 25.2776642, -9.8157730, 25.5268135, -35.3052788, 35.0934372
3: -9.6009922, 28.9526863, -9.6784315, 29.2690525, -38.6395645, 38.3977966
4: -16.6786633, 25.3558273, -16.7464676, 25.6926041, -42.2492294, 41.9791107
5: -7.4509511, 29.0169945, -7.5338378, 29.3029785, -36.3218918, 36.1088600
6: -38.2461281, 12.0149260, -38.4105148, 12.0772705, -50.3233986, 50.4254417
7: -11.1814499, 28.6531982, -11.2453737, 28.8116760, -38.7799149, 38.6653786
8: -21.3333664, 29.8632870, -21.3600960, 30.1630650, -51.0695877, 50.7693634
9: -13.8367586, 28.3018379, -13.9415121, 28.4059811, -42.2427406, 42.2433510
10: -22.2085953, 31.9467564, -22.5645485, 32.0771523, -54.2857475, 54.5113068
11: -23.7451019, 14.6296387, -24.2270012, 14.7391987, -38.4842987, 38.8566399
12: -44.2770119, 4.2869415, -44.8469086, 4.5353794, -45.2591858, 45.5616531
13: -37.5257530, 22.2673740, -37.5999641, 22.4722252, -59.7226334, 59.6035461
14: -64.9568558, 2.5424433, -65.5001373, 2.7697840, -67.7266388, 68.0425797
15: -21.8291683, 20.3597240, -21.9580593, 20.6336784, -42.4628448, 42.3177834
16: -23.5523491, 21.6813450, -23.7496014, 21.7559566, -45.3083038, 45.4309464
17: -58.4723396, -1.3171072, -58.9263229, -1.0925570, -56.1243744, 56.3721046
18: -35.8689613, 14.6874542, -36.0821915, 14.6794758, -50.5484390, 50.7696457
19: -26.4463768, 9.5274086, -26.7109852, 9.5274849, -35.9738617, 36.2383957
20: -21.5414581, 15.9071264, -21.8048077, 15.9248056, -37.4662628, 37.7119331
21: -27.3029690, 12.9817152, -27.6755104, 13.0317116, -40.3346786, 40.6572266
22: -32.1127281, 10.7010078, -32.2466202, 10.7061844, -42.8189125, 42.9476280
23: -24.5945187, 14.0587883, -24.8058815, 14.0786839, -38.6732025, 38.8646698
24: -30.7403011, 13.7895098, -30.8400402, 13.7661228, -44.5064240, 44.6295509
25: -28.8956127, 12.9982557, -29.0567627, 12.9902897, -41.8859024, 42.0550194
26: -41.0383301, 17.0555229, -41.3754578, 17.1125107, -58.1508408, 58.4309807
27: -26.0799980, 18.2642002, -26.2175465, 18.2945004, -44.3744965, 44.4817467
28: -25.0751019, 17.3890438, -25.2238884, 17.3535843, -42.4286880, 42.6129303
29: -27.6287918, 10.9605217, -27.8214836, 10.9934702, -38.4376526, 38.6013374
30: -26.8530388, 18.3354225, -27.0264797, 18.3495064, -45.2025452, 45.3619003
31: -35.4178009, 12.1805096, -35.7184982, 12.1579800, -47.5757828, 47.8990097
32: -35.2368965, 10.9877472, -35.4739838, 11.0773506, -45.8438416, 46.0081100
33: -63.6600761, -3.7100105, -63.7698479, -3.4873033, -55.5762939, 55.4072418
34: -57.7891464, -6.2864771, -57.9076309, -6.1138659, -47.7761459, 47.6408691
35: -56.0792732, -4.2604284, -56.1254578, -4.1469440, -45.0513306, 44.9141922
36: -53.4878998, 0.9598637, -53.6077576, 0.9921007, -49.4748688, 49.5687637
37: -78.2725983, -14.2914753, -78.4393387, -14.1869183, -60.7994080, 60.8684311
38: -63.8216057, 0.5191417, -63.9643936, 0.5474811, -59.7202911, 59.8609772
39: -72.1333923, -8.1491032, -72.2724762, -8.0255547, -58.0788727, 58.1717148
40: -51.3933449, -6.2093654, -51.5051994, -6.0771427, -45.3162003, 45.2958336
41: -40.0548134, 12.2579403, -40.1681366, 12.3350210, -52.3898354, 52.4260788
42: -26.1655617, 11.9289064, -26.3173256, 12.0171766, -38.1827393, 38.2462311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=260, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 935
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1374
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1293
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 734

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837214
time: 48.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0877306, upper bound: 20.1837214
time: 65.62 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 116.48 seconds
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1329423
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0655541, upper bound: 20.1837211
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837216
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0877306, upper bound: 20.1837211
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837214
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0655541, upper bound: 20.1837214
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837214
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 116.48
Output dim: 5, lower bound: -20.0877306, upper bound: 20.1837214

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -37.6752434, 17.4696121, -37.8669510, 17.6142616, -55.2895050, 55.3365631
1: -11.9175930, 22.4274864, -12.0957689, 22.5875378, -34.5051308, 34.5232544
2: -9.6651554, 25.2297173, -9.8656940, 25.4581966, -35.1233521, 35.0954132
3: -9.4949436, 28.9087200, -9.7020359, 29.1308918, -38.4017105, 38.3785973
4: -16.5202141, 25.3028336, -16.7920704, 25.5773811, -41.9816208, 41.9785461
5: -7.3530455, 28.9712296, -7.5818467, 29.1874084, -36.1171722, 36.1196289
6: -38.1934891, 11.9045000, -38.3728485, 12.1187162, -50.3122063, 50.2773476
7: -11.0713863, 28.6162872, -11.3350277, 28.7109318, -38.5854568, 38.7308197
8: -21.1031647, 29.7722244, -21.4065475, 30.0648499, -50.7565689, 50.7371445
9: -13.7470407, 28.2644806, -13.9887915, 28.3018646, -42.0489044, 42.2532730
10: -22.1255856, 31.8959389, -22.6406937, 32.0395660, -54.1454926, 54.5366325
11: -23.6151924, 14.4753551, -24.1443214, 14.6838980, -38.2990913, 38.6196747
12: -44.2309303, 4.1616211, -44.7715836, 4.5910740, -45.2914658, 45.3949051
13: -37.3916321, 22.1580219, -37.5579300, 22.3795643, -59.4206619, 59.4388580
14: -64.8802948, 2.4816532, -65.4780884, 2.7358103, -67.6161041, 67.9597397
15: -21.6777554, 20.2905464, -21.9603615, 20.5359936, -42.2137489, 42.2509079
16: -23.4745331, 21.6274242, -23.8150406, 21.6893921, -45.1639252, 45.4424667
17: -58.3613281, -1.4096432, -58.8726044, -1.1723957, -55.9523010, 56.3146477
18: -35.7614517, 14.5057535, -36.0031586, 14.6462631, -50.4077148, 50.5089111
19: -26.3788013, 9.4585285, -26.6031532, 9.5531731, -35.9319763, 36.0616837
20: -21.4795055, 15.8339758, -21.7031269, 15.9529734, -37.4324799, 37.5371017
21: -27.2086754, 12.8839397, -27.5865746, 13.0513344, -40.2600098, 40.4705124
22: -32.0419159, 10.6350527, -32.1210175, 10.7371931, -42.7791100, 42.7560692
23: -24.5285816, 13.9500389, -24.6731644, 14.0612431, -38.5898247, 38.6232033
24: -30.6466465, 13.6783657, -30.6530323, 13.7272434, -44.3738899, 44.3313980
25: -28.8396339, 12.9241419, -28.8946953, 13.0230789, -41.8627129, 41.8188362
26: -40.9383354, 16.9329929, -41.2700615, 17.1353931, -58.0737305, 58.2030563
27: -26.0037308, 18.1971264, -26.1193523, 18.2998505, -44.3035812, 44.3164787
28: -25.0039520, 17.2793541, -25.1060257, 17.3762856, -42.3802376, 42.3853798
29: -27.5472698, 10.8717117, -27.6816483, 10.9912596, -38.3570862, 38.3767548
30: -26.7457314, 18.1732159, -26.8909225, 18.2752552, -45.0209885, 45.0641403
31: -35.3319473, 12.0642567, -35.5737915, 12.1802015, -47.5121498, 47.6380463
32: -35.1872444, 10.8958101, -35.4146347, 11.1177464, -45.8315048, 45.8604736
33: -63.6182518, -3.7895083, -63.7300453, -3.4050131, -55.6369476, 55.3066406
34: -57.7043495, -6.4303493, -57.8295441, -6.0793724, -47.7548904, 47.4607010
35: -56.0101357, -4.3763371, -56.0755920, -4.0786724, -45.0685959, 44.7848625
36: -53.4441795, 0.8875360, -53.5563889, 1.1241693, -49.5721512, 49.4572067
37: -78.1927948, -14.4081583, -78.3290024, -14.1499367, -60.7820892, 60.6708450
38: -63.7612076, 0.4333506, -63.8918228, 0.7227802, -59.8496475, 59.7092896
39: -72.0599670, -8.2086840, -72.1924362, -7.9267912, -58.1098709, 58.0129013
40: -51.3335724, -6.3075571, -51.4608002, -6.0624566, -45.2711143, 45.1532440
41: -40.0054932, 12.1655579, -40.1388779, 12.3651505, -52.3706436, 52.3044357
42: -26.1240005, 11.8860483, -26.2802620, 12.0634117, -38.1874123, 38.1663094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=259, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0148906, upper bound: 20.1837216
time: 56.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0148906, upper bound: 20.1329418
time: 66.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -37.7902412, 17.5699673, -37.6897888, 17.6190338, -55.4092751, 55.2597580
1: -12.0334206, 22.4651375, -11.9825172, 22.5658512, -34.5992737, 34.4476547
2: -9.7774820, 25.2716122, -9.7773933, 25.4355736, -35.2130547, 35.0490036
3: -9.6002302, 28.9483795, -9.6272411, 29.1116905, -38.4873505, 38.3496552
4: -16.6771603, 25.3511295, -16.6928883, 25.5604095, -42.1213379, 41.9278412
5: -7.4499669, 29.0118637, -7.4852018, 29.1672440, -36.1934814, 36.0650063
6: -38.2417488, 12.0131474, -38.3507576, 12.0451622, -50.2869110, 50.3639069
7: -11.1804628, 28.6461449, -11.1973591, 28.6837063, -38.6631012, 38.6224518
8: -21.3317432, 29.8562336, -21.2966213, 30.0334206, -50.9456024, 50.7081146
9: -13.8354206, 28.2954845, -13.8782816, 28.2776604, -42.1130829, 42.1737671
10: -22.2072105, 31.9405785, -22.5133114, 32.0122414, -54.2134705, 54.4538879
11: -23.7424011, 14.6284428, -24.1028080, 14.6964512, -38.4388504, 38.7312508
12: -44.2742424, 4.2847099, -44.7476044, 4.4887753, -45.2344055, 45.4869461
13: -37.5246201, 22.2646446, -37.5608978, 22.3640270, -59.5578918, 59.5188599
14: -64.9542313, 2.5350571, -65.3528748, 2.7105551, -67.6647873, 67.8879318
15: -21.8271255, 20.3589535, -21.9113007, 20.5342731, -42.3613968, 42.2702560
16: -23.5506382, 21.6729393, -23.6778736, 21.6631660, -45.2138062, 45.3508148
17: -58.4705048, -1.3197298, -58.7897835, -1.1679897, -56.0822296, 56.2721596
18: -35.8646965, 14.6866207, -35.9722519, 14.6278954, -50.4925919, 50.6588745
19: -26.4407616, 9.5270424, -26.5688190, 9.4886408, -35.9294014, 36.0958633
20: -21.5353031, 15.9061813, -21.6735191, 15.8932343, -37.4285355, 37.5797005
21: -27.2971745, 12.9808893, -27.5410824, 13.0041103, -40.3012848, 40.5219727
22: -32.1048088, 10.7003021, -32.0765076, 10.6649446, -42.7697525, 42.7768097
23: -24.5905933, 14.0577469, -24.6521587, 14.0314016, -38.6219940, 38.7099075
24: -30.7359543, 13.7886181, -30.6401176, 13.7172298, -44.4531860, 44.4287338
25: -28.8882561, 12.9972200, -28.8633461, 12.9415283, -41.8297844, 41.8605652
26: -41.0301590, 17.0546284, -41.2263641, 17.0629883, -58.0931473, 58.2809906
27: -26.0764866, 18.2634773, -26.1037006, 18.2674198, -44.3439064, 44.3671799
28: -25.0690632, 17.3880539, -25.0785866, 17.3093796, -42.3784409, 42.4666405
29: -27.6224136, 10.9596615, -27.6420650, 10.9553328, -38.3971634, 38.4238739
30: -26.8510036, 18.3339806, -26.8798046, 18.3069439, -45.1579475, 45.2137833
31: -35.4108887, 12.1796494, -35.5337296, 12.1057997, -47.5166893, 47.7133789
32: -35.2330093, 10.9865417, -35.3989944, 11.0519657, -45.8132935, 45.9306488
33: -63.6525383, -3.7109737, -63.6934280, -3.5203295, -55.5428543, 55.3403091
34: -57.7818184, -6.2873487, -57.8066254, -6.1558685, -47.7498169, 47.5631332
35: -56.0704346, -4.2610235, -56.0443611, -4.1805334, -45.0245667, 44.8540344
36: -53.4772530, 0.9592333, -53.5088387, 0.9656820, -49.4527740, 49.4873352
37: -78.2641449, -14.2923183, -78.2859802, -14.2214203, -60.7825317, 60.7405090
38: -63.8058929, 0.5176792, -63.8199997, 0.4980211, -59.6782990, 59.7388916
39: -72.1252365, -8.1498308, -72.1604156, -8.0434618, -58.0600433, 58.0711136
40: -51.3902054, -6.2104321, -51.4343948, -6.1056337, -45.2845726, 45.2239609
41: -40.0501862, 12.2568760, -40.1136017, 12.3099194, -52.3601074, 52.3704758
42: -26.1618614, 11.9274807, -26.2607327, 11.9939690, -38.1558304, 38.1882133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=259, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9973382, upper bound: 20.1837216
time: 90.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9973382, upper bound: 20.1334171
time: 53.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -37.7891045, 17.5723572, -37.9155884, 17.6636257, -55.4527283, 55.4879456
1: -12.0326281, 22.4686432, -12.1454554, 22.5971451, -34.6297722, 34.6140976
2: -9.7763596, 25.2754784, -9.9183302, 25.4659882, -35.2423477, 35.1938095
3: -9.5994186, 28.9507999, -9.7510233, 29.1400452, -38.5146713, 38.4763184
4: -16.6757603, 25.3535080, -16.8661499, 25.5863304, -42.1462326, 42.1035576
5: -7.4491601, 29.0148411, -7.6258707, 29.1955566, -36.2209663, 36.2087135
6: -38.2427940, 12.0112896, -38.3865814, 12.1700754, -50.4128685, 50.3978729
7: -11.1792698, 28.6504936, -11.3801003, 28.7183895, -38.6967239, 38.8101578
8: -21.3311539, 29.8605824, -21.5177841, 30.0775928, -50.9880447, 50.9342728
9: -13.8348227, 28.2994595, -14.0307226, 28.3107967, -42.1456184, 42.3301811
10: -22.2066307, 31.9437695, -22.6738033, 32.0595779, -54.2605286, 54.6175728
11: -23.7422161, 14.6276484, -24.1660633, 14.7583771, -38.5005951, 38.7937126
12: -44.2721672, 4.2844954, -44.7794952, 4.6481361, -45.3895721, 45.5185394
13: -37.5222588, 22.2654305, -37.6192360, 22.4100075, -59.5831375, 59.6296692
14: -64.9513397, 2.5376921, -65.5087585, 2.7603521, -67.7116928, 68.0464478
15: -21.8270416, 20.3554993, -22.0302753, 20.5553932, -42.3824348, 42.3857727
16: -23.5498753, 21.6781425, -23.8393383, 21.7139282, -45.2638016, 45.5174789
17: -58.4684868, -1.3189497, -58.8944321, -1.1295700, -56.1124039, 56.4267387
18: -35.8667908, 14.6860304, -36.0214920, 14.7352648, -50.6020546, 50.7075233
19: -26.4438324, 9.5259724, -26.6178875, 9.5866795, -36.0305099, 36.1438599
20: -21.5385551, 15.9053383, -21.7219124, 15.9864616, -37.5250168, 37.6272507
21: -27.2999458, 12.9800968, -27.6072159, 13.0979137, -40.3978577, 40.5873108
22: -32.1093063, 10.6998291, -32.1375732, 10.7680969, -42.8774033, 42.8374023
23: -24.5923462, 14.0569057, -24.6878185, 14.1125917, -38.7049370, 38.7447243
24: -30.7366486, 13.7880077, -30.6736450, 13.7807751, -44.5174255, 44.4616547
25: -28.8924026, 12.9967918, -28.9100990, 13.0574160, -41.9498177, 41.9068909
26: -41.0346527, 17.0534973, -41.2914124, 17.1946754, -58.2293282, 58.3449097
27: -26.0775719, 18.2628746, -26.1450272, 18.3311920, -44.4087639, 44.4079018
28: -25.0723686, 17.3872032, -25.1225739, 17.4286804, -42.5010490, 42.5097771
29: -27.6253624, 10.9590893, -27.6981144, 11.0339909, -38.4791336, 38.4807701
30: -26.8505402, 18.3336430, -26.9152946, 18.3532047, -45.2037430, 45.2489395
31: -35.4148712, 12.1787300, -35.5935440, 12.2364292, -47.6512985, 47.7722740
32: -35.2328644, 10.9853153, -35.4292297, 11.1574421, -45.9175262, 45.9596024
33: -63.6567154, -3.7116003, -63.7509155, -3.3712983, -55.6963501, 55.3960800
34: -57.7862206, -6.2880478, -57.8462448, -6.0175676, -47.8955765, 47.6010056
35: -56.0757217, -4.2616291, -56.0909729, -4.0294933, -45.1820908, 44.8982544
36: -53.4841576, 0.9587479, -53.5657349, 1.1547384, -49.6479950, 49.5402908
37: -78.2687759, -14.2931795, -78.3462372, -14.0976639, -60.9112854, 60.8014069
38: -63.8164368, 0.5172968, -63.9079552, 0.7547641, -59.9432755, 59.8176193
39: -72.1291199, -8.1504192, -72.2212372, -7.9178467, -58.1876297, 58.1308136
40: -51.3909607, -6.2119370, -51.4829292, -6.0239892, -45.3669701, 45.2709923
41: -40.0527802, 12.2549400, -40.1531448, 12.4078503, -52.4606323, 52.4080849
42: -26.1630745, 11.9249315, -26.2966690, 12.0793753, -38.2424507, 38.2215996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=259, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0370681, upper bound: 20.1837211
time: 52.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -20.0370681, upper bound: 20.1334172
time: 56.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -37.6763458, 17.4671783, -37.7170639, 17.6044197, -55.2807655, 55.1842422
1: -11.9183531, 22.4239807, -11.9771481, 22.6325703, -34.5509224, 34.4011307
2: -9.6663103, 25.2258720, -9.7589693, 25.4937019, -35.1600113, 34.9848404
3: -9.4957666, 28.9062805, -9.6261883, 29.2419720, -38.5073318, 38.2924805
4: -16.5216103, 25.3004551, -16.6660633, 25.6639214, -42.0630035, 41.8430061
5: -7.3538523, 28.9682732, -7.4856620, 29.2734165, -36.1953506, 36.0104218
6: -38.1924744, 11.9063625, -38.3787270, 12.0183582, -50.2108307, 50.2850876
7: -11.0725956, 28.6119442, -11.1961174, 28.7745476, -38.6380005, 38.5747375
8: -21.1037807, 29.7678566, -21.2420826, 30.1207447, -50.8065033, 50.5582581
9: -13.7476368, 28.2605362, -13.8939295, 28.3704872, -42.1181259, 42.1544647
10: -22.1261272, 31.8927727, -22.5256424, 32.0311508, -54.1468201, 54.4184151
11: -23.6154003, 14.4761496, -24.1940689, 14.6597071, -38.2751083, 38.6702194
12: -44.2330055, 4.1617813, -44.8263054, 4.4690113, -45.1483002, 45.4229126
13: -37.3939705, 22.1572189, -37.5340462, 22.4302921, -59.5376282, 59.4016876
14: -64.8832245, 2.4789276, -65.4584122, 2.7139597, -67.5971832, 67.9373398
15: -21.6778030, 20.2940483, -21.8796349, 20.6111660, -42.2889709, 42.1736832
16: -23.4753036, 21.6222191, -23.7180328, 21.6961269, -45.1714325, 45.3402519
17: -58.3633118, -1.4104004, -58.8967400, -1.1461773, -55.9436340, 56.2480049
18: -35.7593536, 14.5063667, -36.0461502, 14.5869799, -50.3463326, 50.5525169
19: -26.3757591, 9.4596348, -26.6728611, 9.4925232, -35.8682823, 36.1324959
20: -21.4762440, 15.8347950, -21.7602997, 15.8873758, -37.3636208, 37.5950928
21: -27.2059135, 12.8847170, -27.6307201, 12.9817390, -40.1876526, 40.5154381
22: -32.0374298, 10.6355009, -32.1969757, 10.6722145, -42.7096443, 42.8324776
23: -24.5268211, 13.9508553, -24.7748108, 14.0230980, -38.5499191, 38.7256660
24: -30.6459541, 13.6790276, -30.8016529, 13.7089386, -44.3548927, 44.4806824
25: -28.8354912, 12.9245720, -29.0105934, 12.9515924, -41.7870827, 41.9351654
26: -40.9338379, 16.9341240, -41.3199310, 17.0494652, -57.9833031, 58.2540550
27: -26.0026360, 18.1977558, -26.1772804, 18.2600613, -44.2626953, 44.3750381
28: -25.0006409, 17.2801971, -25.1819992, 17.2971001, -42.2977409, 42.4621964
29: -27.5443211, 10.8723269, -27.7785015, 10.9472828, -38.3056030, 38.4693909
30: -26.7461777, 18.1735497, -26.9938278, 18.2655792, -45.0117569, 45.1673775
31: -35.3279686, 12.0651073, -35.6698799, 12.0980206, -47.4259872, 47.7349854
32: -35.1873970, 10.8970251, -35.4431190, 11.0323105, -45.7484436, 45.8912659
33: -63.6141129, -3.7888856, -63.7176361, -3.5251780, -55.5047302, 55.2856522
34: -57.6999626, -6.4296398, -57.8605003, -6.1793613, -47.6240997, 47.4687195
35: -56.0048637, -4.3757410, -56.0729828, -4.1987848, -44.9259720, 44.7626801
36: -53.4373398, 0.8880014, -53.5538712, 0.9586792, -49.3851547, 49.4394455
37: -78.1881485, -14.4073133, -78.3866272, -14.2426281, -60.6571198, 60.6992569
38: -63.7506523, 0.4337125, -63.8821640, 0.5093446, -59.6044388, 59.6842651
39: -72.0561371, -8.2080345, -72.2098770, -8.0375633, -57.9894104, 58.0191345
40: -51.3328590, -6.3060102, -51.4699402, -6.1198587, -45.2130013, 45.1639290
41: -40.0029144, 12.1674919, -40.1345596, 12.2877579, -52.2906723, 52.3020515
42: -26.1228294, 11.8886061, -26.2854099, 11.9954033, -38.1182327, 38.1740150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=259, inp2_unstable=259, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=310, inp2_unstable=310, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=36, inp2_unstable=36, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1374
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9751647, upper bound: 20.1837219
time: 49.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -19.9751647, upper bound: 20.1334180
time: 54.97 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 106.47 seconds
IS_A1_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 106.47
Output dim: 5, lower bound: -20.0148906, upper bound: 20.1837216
IS_A1_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 106.47
Output dim: 5, lower bound: -20.0148906, upper bound: 20.1329418
IS_A1_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 106.47
Output dim: 5, lower bound: -19.9973382, upper bound: 20.1837216
IS_A1_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 106.47
Output dim: 5, lower bound: -19.9973382, upper bound: 20.1334171
IS_A1_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 106.47
Output dim: 5, lower bound: -20.0370681, upper bound: 20.1837211
IS_A1_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 106.47
Output dim: 5, lower bound: -20.0370681, upper bound: 20.1334172
IS_A1_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 106.47
Output dim: 5, lower bound: -19.9751647, upper bound: 20.1837219
IS_A1_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 106.47
Output dim: 5, lower bound: -19.9751647, upper bound: 20.1334180
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 106.47
Output dim: 5, lower bound: -20.0655541, upper bound: 20.1837214
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 106.47
Output dim: 5, lower bound: -20.0258153, upper bound: 20.1837214
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 106.47
Output dim: 5, lower bound: -20.0877306, upper bound: 20.1837214

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 60.26 + 3541.48 = 3601.74 seconds
