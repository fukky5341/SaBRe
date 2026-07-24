## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.31 + 58.07 = 60.38 seconds
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1391674, upper bound: 20.1948068
time: 53.40 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2009809, upper bound: 20.2009814
time: 105.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 159.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 159.10
Output dim: 5, lower bound: -20.1391674, upper bound: 20.1948068
IS_A2, status: Status.UNKNOWN, split count: 1, time: 159.10
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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408032
time: 120.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
time: 52.75 seconds

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

Time for backsubstitution: 1.73 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1975767, upper bound: 20.1473699
time: 52.08 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1975802, upper bound: 20.1975803
time: 86.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 139.90 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 139.90
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1408032
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 139.90
Output dim: 5, lower bound: -20.1358454, upper bound: 20.1915911
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 139.90
Output dim: 5, lower bound: -20.1975767, upper bound: 20.1473699
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 139.90
Output dim: 5, lower bound: -20.1975802, upper bound: 20.1975803

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -37.6348991, 17.5734482, -37.6398697, 17.5808487, -55.2157478, 55.2133179
1: -11.9102364, 22.4700489, -11.9065895, 22.4708328, -34.3810692, 34.3766403
2: -9.6730747, 25.2764626, -9.6788120, 25.2786465, -34.9517212, 34.9552765
3: -9.5286560, 28.9505825, -9.5422392, 28.9536076, -38.2533112, 38.2648773
4: -16.5625648, 25.3568478, -16.5829773, 25.3593712, -41.8054352, 41.8242111
5: -7.3728123, 29.0144482, -7.3837676, 29.0204887, -35.9676971, 35.9711990
6: -38.2342529, 11.9759045, -38.2374878, 11.9962902, -50.2305450, 50.2133942
7: -11.0577698, 28.6548443, -11.0548325, 28.6573257, -38.5011978, 38.4907951
8: -21.1863270, 29.8569355, -21.1952801, 29.8575706, -50.6157227, 50.6231842
9: -13.7226782, 28.3040009, -13.7310276, 28.2873344, -42.0100136, 42.0350266
10: -22.0952492, 31.9417934, -22.0962887, 31.9082088, -54.0034561, 54.0380821
11: -23.7017403, 14.6049042, -23.7135677, 14.6409101, -38.3426514, 38.3184738
12: -44.2486649, 4.2438183, -44.2505302, 4.2730246, -44.9837494, 44.9545135
13: -37.4649506, 22.2561188, -37.4673996, 22.2956924, -59.4484558, 59.4165268
14: -64.8763657, 2.5353127, -64.8753738, 2.5593042, -67.4356689, 67.4106903
15: -21.7854919, 20.3357258, -21.8273621, 20.3416767, -42.1271667, 42.1630859
16: -23.4311810, 21.6967850, -23.4381542, 21.7020054, -45.1331863, 45.1349411
17: -58.3978577, -1.3263712, -58.4021606, -1.2740622, -55.8957520, 55.8455620
18: -35.8478470, 14.6154289, -35.8522568, 14.6045170, -50.4523621, 50.4676857
19: -26.4333420, 9.4341640, -26.4364414, 9.4438868, -35.8772278, 35.8706055
20: -21.5360889, 15.8326159, -21.5410786, 15.8411674, -37.3772583, 37.3736954
21: -27.2784691, 12.8984890, -27.2837181, 12.9087086, -40.1871796, 40.1822052
22: -32.0993652, 10.6044035, -32.1108398, 10.6013975, -42.7007637, 42.7152443
23: -24.5882721, 13.9950752, -24.5907059, 14.0014925, -38.5897636, 38.5857811
24: -30.7390995, 13.7234392, -30.7486420, 13.7279310, -44.4670296, 44.4720802
25: -28.8930397, 12.8835049, -28.9006004, 12.8891401, -41.7821808, 41.7841034
26: -41.0235748, 16.9721317, -41.0315895, 16.9621181, -57.9856949, 58.0037231
27: -26.0695190, 18.1806850, -26.0821342, 18.1835823, -44.2531013, 44.2628174
28: -25.0710049, 17.2777481, -25.0741425, 17.2879601, -42.3589630, 42.3518906
29: -27.6124668, 10.8870754, -27.6174545, 10.9003410, -38.3309631, 38.3233795
30: -26.8428860, 18.2924652, -26.8521004, 18.2961884, -45.1390762, 45.1445656
31: -35.4043617, 12.0648775, -35.4058990, 12.0721998, -47.4765625, 47.4707756
32: -35.2242889, 10.9458637, -35.2288971, 10.9648714, -45.7285309, 45.7157288
33: -63.6481857, -3.7818642, -63.6539078, -3.7764497, -55.2221680, 55.2262421
34: -57.7863464, -6.3904848, -57.8006020, -6.3842926, -47.4645233, 47.4831238
35: -56.0765305, -4.3663788, -56.0673981, -4.3632259, -44.7856445, 44.7818527
36: -53.4851303, 0.8292580, -53.4874077, 0.8565788, -49.3640289, 49.3393250
37: -78.2631302, -14.3114719, -78.2724991, -14.2939463, -60.7096710, 60.7097549
38: -63.8155594, 0.3473425, -63.8127403, 0.3809395, -59.5940781, 59.5636368
39: -72.1238556, -8.2030039, -72.1315689, -8.1879759, -57.9251556, 57.9290390
40: -51.3500633, -6.2266936, -51.3582993, -6.2181749, -45.1318893, 45.1316071
41: -40.0497284, 12.2299528, -40.0494423, 12.2408400, -52.2905693, 52.2793961
42: -26.1599522, 11.9269800, -26.1661377, 11.9368668, -38.0968170, 38.0931168

Time for backsubstitution: 1.73 seconds

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
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1340
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
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1413
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1408032
time: 42.68 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1408032
time: 54.04 seconds

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

Time for backsubstitution: 1.82 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
time: 51.96 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
time: 63.20 seconds

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

Time for backsubstitution: 1.81 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
time: 50.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
time: 65.49 seconds

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

Time for backsubstitution: 1.82 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1559530
time: 47.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1961887
time: 51.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 101.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1408032
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1408032
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.0851744, upper bound: 20.1473699
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1559530
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 101.35
Output dim: 5, lower bound: -20.1961887, upper bound: 20.1961887

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -37.5989189, 17.5620937, -37.6398697, 17.5808487, -55.1797676, 55.2019653
1: -11.8745222, 22.4637051, -11.9065895, 22.4708328, -34.3453560, 34.3702927
2: -9.6261940, 25.2707977, -9.6788120, 25.2786465, -34.9048386, 34.9496078
3: -9.4760361, 28.9416599, -9.5422392, 28.9536076, -38.2007523, 38.2561035
4: -16.5154419, 25.3492374, -16.5829773, 25.3593712, -41.7589111, 41.8165894
5: -7.3212476, 29.0064621, -7.3837676, 29.0204887, -35.9146500, 35.9632721
6: -38.2220116, 11.9612103, -38.2374878, 11.9962902, -50.2182999, 50.1987000
7: -11.0123682, 28.6480026, -11.0548325, 28.6573257, -38.4530869, 38.4837608
8: -21.1349449, 29.8487186, -21.1952801, 29.8575706, -50.5618591, 50.6154022
9: -13.7073612, 28.2657280, -13.7310276, 28.2873344, -41.9946976, 41.9967575
10: -22.0736523, 31.8604679, -22.0962887, 31.9082088, -53.9818611, 53.9567566
11: -23.6886024, 14.5673428, -23.7135677, 14.6409101, -38.3295135, 38.2809105
12: -44.2374840, 4.1527662, -44.2505302, 4.2730246, -44.9725952, 44.8628845
13: -37.4524574, 22.2384987, -37.4673996, 22.2956924, -59.4347382, 59.3941879
14: -64.8475037, 2.4479618, -64.8753738, 2.5593042, -67.4068069, 67.3233337
15: -21.7557564, 20.3212051, -21.8273621, 20.3416767, -42.0974350, 42.1485672
16: -23.4094410, 21.6651802, -23.4381542, 21.7020054, -45.1114464, 45.1033325
17: -58.3871994, -1.3708696, -58.4021606, -1.2740622, -55.8851166, 55.8019600
18: -35.8360634, 14.5843544, -35.8522568, 14.6045170, -50.4405823, 50.4366112
19: -26.4213905, 9.4049911, -26.4364414, 9.4438868, -35.8652763, 35.8414307
20: -21.5193825, 15.7978592, -21.5410786, 15.8411674, -37.3605499, 37.3389359
21: -27.2646561, 12.8552914, -27.2837181, 12.9087086, -40.1733627, 40.1390076
22: -32.0893784, 10.5813560, -32.1108398, 10.6013975, -42.6907768, 42.6921959
23: -24.5767708, 13.9697037, -24.5907059, 14.0014925, -38.5782623, 38.5604095
24: -30.7277985, 13.7169561, -30.7486420, 13.7279310, -44.4557304, 44.4655991
25: -28.8826866, 12.8575745, -28.9006004, 12.8891401, -41.7718277, 41.7581749
26: -41.0085449, 16.9045677, -41.0315895, 16.9621181, -57.9706650, 57.9361572
27: -26.0444717, 18.1738853, -26.0821342, 18.1835823, -44.2280540, 44.2560196
28: -25.0571518, 17.2565022, -25.0741425, 17.2879601, -42.3451118, 42.3306427
29: -27.6044350, 10.8601875, -27.6174545, 10.9003410, -38.3227997, 38.2962265
30: -26.8312435, 18.2666969, -26.8521004, 18.2961884, -45.1274338, 45.1187973
31: -35.3872070, 12.0281811, -35.4058990, 12.0721998, -47.4594078, 47.4340820
32: -35.2123184, 10.9173946, -35.2288971, 10.9648714, -45.7168121, 45.6842957
33: -63.6089973, -3.7995172, -63.6539078, -3.7764497, -55.1830750, 55.2086639
34: -57.7606049, -6.4049263, -57.8006020, -6.3842926, -47.4415436, 47.4687500
35: -56.0528374, -4.3786774, -56.0673981, -4.3632259, -44.7612381, 44.7702980
36: -53.4749641, 0.8211174, -53.4874077, 0.8565788, -49.3546448, 49.3308258
37: -78.2494507, -14.3313522, -78.2724991, -14.2939463, -60.6948853, 60.6907043
38: -63.7954254, 0.3370485, -63.8127403, 0.3809395, -59.5744095, 59.5531921
39: -72.1064301, -8.2155256, -72.1315689, -8.1879759, -57.9101715, 57.9121323
40: -51.3340683, -6.2337394, -51.3582993, -6.2181749, -45.1158943, 45.1245613
41: -40.0315475, 12.2172813, -40.0494423, 12.2408400, -52.2723885, 52.2667236
42: -26.1506424, 11.9056902, -26.1661377, 11.9368668, -38.0875092, 38.0718269

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1787
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
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
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

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1069563
time: 63.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1408032
time: 44.74 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -37.6750107, 17.6151257, -37.6398697, 17.5808487, -55.2558594, 55.2549973
1: -11.9441967, 22.6080017, -11.9065895, 22.4708328, -34.4150314, 34.5145912
2: -9.6993847, 25.4314804, -9.6788120, 25.2786465, -34.9780312, 35.1102905
3: -9.5444984, 29.1303082, -9.5422392, 28.9536076, -38.2685165, 38.4520569
4: -16.6033516, 25.5383396, -16.5829773, 25.3593712, -41.8472519, 42.0064278
5: -7.4028144, 29.1602077, -7.3837676, 29.0204887, -35.9961319, 36.1186447
6: -38.3059692, 11.9983959, -38.2374878, 11.9962902, -50.3022614, 50.2358856
7: -11.1146984, 28.7617245, -11.0548325, 28.6573257, -38.5580521, 38.6050034
8: -21.2188911, 30.0439796, -21.1952801, 29.8575706, -50.6449509, 50.8118553
9: -13.8783083, 28.3473034, -13.7310276, 28.2873344, -42.1656418, 42.0783310
10: -22.4472141, 31.9725647, -22.0962887, 31.9082088, -54.3554230, 54.0688553
11: -23.9418316, 14.5907631, -23.7135677, 14.6409101, -38.5827408, 38.3043289
12: -44.5614738, 4.2830524, -44.2505302, 4.2730246, -45.3006592, 44.9915314
13: -37.4907913, 22.3289948, -37.4673996, 22.2956924, -59.4778900, 59.4719086
14: -65.2236633, 2.5560112, -64.8753738, 2.5593042, -67.7829666, 67.4313812
15: -21.7943459, 20.4278011, -21.8273621, 20.3416767, -42.1360245, 42.2551651
16: -23.6021652, 21.6974926, -23.4381542, 21.7020054, -45.3041687, 45.1356468
17: -58.6363564, -1.2900829, -58.4021606, -1.2740622, -56.1540680, 55.8729630
18: -36.0087433, 14.6283054, -35.8522568, 14.6045170, -50.6132584, 50.4805603
19: -26.5829277, 9.4457111, -26.4364414, 9.4438868, -36.0268135, 35.8821526
20: -21.6838379, 15.8408957, -21.5410786, 15.8411674, -37.5250053, 37.3819733
21: -27.4925137, 12.9198990, -27.2837181, 12.9087086, -40.4012222, 40.2036171
22: -32.1731262, 10.6316996, -32.1108398, 10.6013975, -42.7745247, 42.7425385
23: -24.6974163, 14.0095100, -24.5907059, 14.0014925, -38.6989098, 38.6002159
24: -30.7814617, 13.7326517, -30.7486420, 13.7279310, -44.5093918, 44.4812927
25: -28.9761238, 12.9116631, -28.9006004, 12.8891401, -41.8652649, 41.8122635
26: -41.2303772, 16.9992981, -41.0315895, 16.9621181, -58.1924973, 58.0308876
27: -26.1203671, 18.2365627, -26.0821342, 18.1835823, -44.3039474, 44.3186951
28: -25.1329784, 17.2895679, -25.0741425, 17.2879601, -42.4209366, 42.3637085
29: -27.7161636, 10.9022465, -27.6174545, 10.9003410, -38.4367065, 38.3397408
30: -26.9346581, 18.2974510, -26.8521004, 18.2961884, -45.2308464, 45.1495514
31: -35.5822296, 12.0716705, -35.4058990, 12.0721998, -47.6544304, 47.4775696
32: -35.3458748, 10.9750500, -35.2288971, 10.9648714, -45.8484039, 45.7367096
33: -63.6728134, -3.6114049, -63.6539078, -3.7764497, -55.2459717, 55.4430618
34: -57.8152237, -6.2588978, -57.8006020, -6.3842926, -47.4914093, 47.6605072
35: -56.0867882, -4.2232494, -56.0673981, -4.3632259, -44.8049545, 44.9793015
36: -53.5069580, 0.8912048, -53.4874077, 0.8565788, -49.3847961, 49.4021606
37: -78.3204651, -14.2841091, -78.2724991, -14.2939463, -60.7820892, 60.7581863
38: -63.8562012, 0.4250445, -63.8127403, 0.3809395, -59.6327133, 59.6427612
39: -72.1723099, -8.1215353, -72.1315689, -8.1879759, -57.9805298, 58.0124969
40: -51.4258842, -6.1652985, -51.3582993, -6.2181749, -45.2077103, 45.1930008
41: -40.0911179, 12.2773972, -40.0494423, 12.2408400, -52.3319588, 52.3268394
42: -26.2284737, 11.9522057, -26.1661377, 11.9368668, -38.1653404, 38.1183434

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
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
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1431
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
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1787
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
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 752
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
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
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1069563
time: 54.22 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1408032
time: 51.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -37.5470581, 17.5360661, -37.6957588, 17.6200867, -55.1671448, 55.2318268
1: -11.8615799, 22.3914604, -11.9705048, 22.5824280, -34.4440079, 34.3619652
2: -9.6344433, 25.2080078, -9.7463770, 25.4111309, -35.0455742, 34.9543839
3: -9.4750595, 28.8060226, -9.6038685, 29.0819721, -38.3348465, 38.1801872
4: -16.5111485, 25.2401047, -16.6640091, 25.5001450, -41.8957214, 41.7884598
5: -7.3230343, 28.8960075, -7.4578633, 29.1248436, -36.0265579, 35.9256020
6: -38.1886635, 11.9430285, -38.3053818, 12.0257635, -50.2144279, 50.2484093
7: -11.0070934, 28.5529919, -11.1514950, 28.7285385, -38.5352859, 38.4864197
8: -21.1247005, 29.7533703, -21.2711525, 30.0102654, -50.7209930, 50.5876770
9: -13.6609411, 28.1970787, -13.8904104, 28.3245239, -41.9854660, 42.0874901
10: -22.0462856, 31.8955116, -22.4552879, 32.0074997, -54.0537872, 54.3507996
11: -23.5842628, 14.5463181, -23.9211597, 14.6580210, -38.2422829, 38.4674759
12: -44.1580505, 4.2001410, -44.5385513, 4.3967419, -45.0136032, 45.2049713
13: -37.4265251, 22.1502914, -37.4970932, 22.3439007, -59.4121704, 59.3509598
14: -64.7329102, 2.5013342, -65.1964417, 2.6634483, -67.3963623, 67.6977768
15: -21.7250710, 20.2357216, -21.8595448, 20.4085388, -42.1336098, 42.0952682
16: -23.3612366, 21.6226273, -23.6141796, 21.7106075, -45.0718460, 45.2368088
17: -58.2656174, -1.3970776, -58.6009789, -1.2083397, -55.8146515, 55.9910622
18: -35.7520370, 14.5607700, -35.9854202, 14.6387596, -50.3907967, 50.5461884
19: -26.3105984, 9.3931255, -26.5474281, 9.4792805, -35.7898788, 35.9405518
20: -21.4263115, 15.8012714, -21.6609001, 15.8805199, -37.3068314, 37.4621735
21: -27.1635704, 12.8707256, -27.4660664, 12.9706478, -40.1342163, 40.3367920
22: -31.9569473, 10.5546589, -32.1362381, 10.6449528, -42.6018982, 42.6908951
23: -24.4466610, 13.9464588, -24.6527061, 14.0355568, -38.4822159, 38.5991669
24: -30.5494766, 13.6749153, -30.7241287, 13.7365017, -44.2859802, 44.3990440
25: -28.7248840, 12.8354998, -28.9245300, 12.9373274, -41.6622124, 41.7600288
26: -40.9030304, 16.9170876, -41.2053223, 17.0502033, -57.9532318, 58.1224098
27: -25.9621773, 18.1550064, -26.1148930, 18.2412167, -44.2033920, 44.2698975
28: -24.9470329, 17.2324963, -25.0984688, 17.3154907, -42.2625237, 42.3309631
29: -27.4532928, 10.8441143, -27.6641769, 10.9367313, -38.2086182, 38.3306389
30: -26.6990509, 18.2435036, -26.8973579, 18.3208389, -45.0198898, 45.1408615
31: -35.2434998, 12.0110655, -35.5343399, 12.1088409, -47.3523407, 47.5454063
32: -35.1608696, 10.9233503, -35.3385010, 11.0166283, -45.7043381, 45.8134155
33: -63.5976830, -3.8139691, -63.6997452, -3.5957155, -55.3990173, 55.2365494
34: -57.7118759, -6.4315271, -57.8253860, -6.2457609, -47.5706329, 47.4626694
35: -56.0264015, -4.3986034, -56.0828285, -4.2135105, -44.9420547, 44.7678146
36: -53.4244957, 0.8048449, -53.4954605, 0.9223671, -49.3720398, 49.3224030
37: -78.1385422, -14.3488865, -78.2937393, -14.2518425, -60.6467590, 60.7091522
38: -63.7303162, 0.3015327, -63.8395844, 0.4588184, -59.5868073, 59.5404587
39: -72.0375290, -8.2196798, -72.1646271, -8.0995770, -57.9302826, 57.9686584
40: -51.2880325, -6.2536554, -51.4279785, -6.1572118, -45.1308212, 45.1743240
41: -40.0070839, 12.2078838, -40.0956421, 12.2949724, -52.3020554, 52.3035278
42: -26.1153736, 11.8955994, -26.2285194, 11.9778929, -38.0932655, 38.1241188

Time for backsubstitution: 1.76 seconds

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
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1340
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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1162141
time: 46.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
time: 46.37 seconds

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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1564538
time: 42.28 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
time: 77.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -37.7103615, 17.6318359, -37.6664124, 17.5945320, -55.3048935, 55.2982483
1: -11.9657211, 22.5259247, -11.9247885, 22.4758873, -34.4416084, 34.4507141
2: -9.7466259, 25.3705330, -9.7162437, 25.2838020, -35.0304260, 35.0867767
3: -9.6143894, 29.0858650, -9.5913029, 28.9605980, -38.3419647, 38.4493027
4: -16.6643181, 25.5088539, -16.6317253, 25.3658562, -41.9104385, 42.0257721
5: -7.4572959, 29.1546898, -7.4293513, 29.0295334, -36.0524254, 36.1585884
6: -38.3294716, 12.0469923, -38.2474823, 12.0148754, -50.3443451, 50.2944756
7: -11.1485672, 28.7033806, -11.0856218, 28.6630020, -38.5848999, 38.5743103
8: -21.2822361, 29.9732609, -21.2403316, 29.8628311, -50.7059097, 50.7890167
9: -13.7752028, 28.3305779, -13.7475786, 28.2978020, -42.0730057, 42.0781555
10: -22.1959763, 31.9696083, -22.1120453, 31.9358749, -54.1318512, 54.0816536
11: -23.9798851, 14.7207966, -23.7307148, 14.6973515, -38.6772385, 38.4515114
12: -44.5266953, 4.4116354, -44.2593307, 4.3640623, -45.3552170, 45.1123428
13: -37.5690613, 22.3918762, -37.4765320, 22.3369064, -59.6155624, 59.5352173
14: -65.1376801, 2.6683207, -64.8919601, 2.6464672, -67.7841492, 67.5602798
15: -21.9252663, 20.5295944, -21.8803539, 20.3556633, -42.2809296, 42.4099503
16: -23.5628967, 21.7303352, -23.4565887, 21.7177353, -45.2806320, 45.1869240
17: -58.6855392, -1.1677999, -58.4118042, -1.2010012, -56.2867584, 55.9857216
18: -35.9162064, 14.6403675, -35.8586273, 14.6187382, -50.5349426, 50.4989929
19: -26.5564671, 9.4895496, -26.4464836, 9.4723625, -36.0288315, 35.9360352
20: -21.6477699, 15.8856583, -21.5564518, 15.8721361, -37.5199051, 37.4421082
21: -27.4552746, 12.9705515, -27.2965603, 12.9485531, -40.4038277, 40.2671127
22: -32.1707878, 10.6590414, -32.1239777, 10.6100035, -42.7807922, 42.7830200
23: -24.6922855, 14.0428152, -24.5999107, 14.0244617, -38.7167473, 38.6427269
24: -30.7950249, 13.7554350, -30.7576752, 13.7355776, -44.5306015, 44.5131111
25: -28.9712372, 12.9401140, -28.9107895, 12.9119291, -41.8831673, 41.8509026
26: -41.1629562, 17.0237999, -41.0468521, 17.0008202, -58.1637764, 58.0706520
27: -26.1496220, 18.2357693, -26.1030025, 18.1899967, -44.3396187, 44.3387718
28: -25.1558552, 17.3247604, -25.0859261, 17.3098907, -42.4657440, 42.4106865
29: -27.7175636, 10.9551792, -27.6234303, 10.9305630, -38.4692078, 38.3943024
30: -26.9309635, 18.3245258, -26.8649082, 18.3058949, -45.2368584, 45.1894341
31: -35.5318146, 12.1188564, -35.4182205, 12.1046534, -47.6364670, 47.5370789
32: -35.3452721, 11.0239277, -35.2390976, 10.9978352, -45.8934479, 45.7993240
33: -63.7141953, -3.6704903, -63.6843567, -3.7607589, -55.2993469, 55.4127731
34: -57.8608856, -6.2553644, -57.8301239, -6.3711748, -47.5466690, 47.6993790
35: -56.0976105, -4.2985907, -56.0762482, -4.3533010, -44.8122025, 44.9190979
36: -53.5825005, 0.9249563, -53.4952621, 0.8824015, -49.4893188, 49.4298706
37: -78.3756561, -14.2304573, -78.2858124, -14.2676477, -60.8247223, 60.8106003
38: -63.9154739, 0.4641418, -63.8230667, 0.4126301, -59.7294769, 59.6709900
39: -72.2177048, -8.1143332, -72.1470490, -8.1741152, -58.0506821, 58.0220718
40: -51.4203644, -6.1341543, -51.3679886, -6.2099953, -45.2103691, 45.2338333
41: -40.1123810, 12.2805481, -40.0611267, 12.2550468, -52.3674278, 52.3416748
42: -26.2430267, 11.9789267, -26.1773815, 11.9529819, -38.1960068, 38.1563072

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
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
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1554
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
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
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
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 940
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
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
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
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
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 850

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1651165, upper bound: 20.0851744
time: 55.81 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.0940342
time: 69.50 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -37.7869415, 17.6849670, -37.6664124, 17.5945320, -55.3814735, 55.3513794
1: -12.0358181, 22.6702728, -11.9247885, 22.4758873, -34.5117035, 34.5950623
2: -9.8205633, 25.5312386, -9.7162437, 25.2838020, -35.1043663, 35.2474823
3: -9.6830425, 29.2746544, -9.5913029, 28.9605980, -38.4101944, 38.6452637
4: -16.7527771, 25.6980553, -16.6317253, 25.3658562, -41.9991150, 42.2156906
5: -7.5392194, 29.3086128, -7.4293513, 29.0295334, -36.1341209, 36.3141174
6: -38.4148369, 12.0841856, -38.2474823, 12.0148754, -50.4297104, 50.3316689
7: -11.2508888, 28.8171768, -11.0856218, 28.6630020, -38.6895676, 38.6955795
8: -21.3670597, 30.1687698, -21.2403316, 29.8628311, -50.7894287, 50.9857178
9: -13.9462061, 28.4120865, -13.7475786, 28.2978020, -42.2440071, 42.1596642
10: -22.5696926, 32.0835609, -22.1120453, 31.9358749, -54.5055695, 54.1930847
11: -24.2344532, 14.7444267, -23.7307148, 14.6973515, -38.9318047, 38.4751434
12: -44.8508148, 4.5418711, -44.2593307, 4.3640623, -45.6832962, 45.2408829
13: -37.6071968, 22.4815483, -37.4765320, 22.3369064, -59.6585159, 59.6121521
14: -65.5135117, 2.7758503, -64.8919601, 2.6464672, -68.1599808, 67.6678085
15: -21.9636936, 20.6388512, -21.8803539, 20.3556633, -42.3193588, 42.5192032
16: -23.7563953, 21.7632580, -23.4565887, 21.7177353, -45.4741287, 45.2198486
17: -58.9349403, -1.0872889, -58.4118042, -1.2010012, -56.5557098, 56.0567894
18: -36.0888824, 14.6846046, -35.8586273, 14.6187382, -50.7076187, 50.5432320
19: -26.7184391, 9.5309238, -26.4464836, 9.4723625, -36.1908035, 35.9774094
20: -21.8123398, 15.9288893, -21.5564518, 15.8721361, -37.6844749, 37.4853401
21: -27.6834335, 13.0356693, -27.2965603, 12.9485531, -40.6319885, 40.3322296
22: -32.2549133, 10.7099876, -32.1239777, 10.6100035, -42.8649178, 42.8339653
23: -24.8135300, 14.0831165, -24.5999107, 14.0244617, -38.8379898, 38.6830292
24: -30.8490391, 13.7711840, -30.7576752, 13.7355776, -44.5846176, 44.5288582
25: -29.0649223, 12.9943647, -28.9107895, 12.9119291, -41.9768524, 41.9051552
26: -41.3847961, 17.1182690, -41.0468521, 17.0008202, -58.3856163, 58.1651230
27: -26.2252922, 18.2984486, -26.1030025, 18.1899967, -44.4152908, 44.4014511
28: -25.2318401, 17.3582363, -25.0859261, 17.3098907, -42.5417328, 42.4441605
29: -27.8300304, 10.9975071, -27.6234303, 10.9305630, -38.5838089, 38.4380493
30: -27.0348873, 18.3548126, -26.8649082, 18.3058949, -45.3407822, 45.2197189
31: -35.7277298, 12.1629467, -35.4182205, 12.1046534, -47.8323822, 47.5811691
32: -35.4790726, 11.0814457, -35.2390976, 10.9978352, -46.0252686, 45.8517532
33: -63.7777939, -3.4826007, -63.6843567, -3.7607589, -55.3619995, 55.6469040
34: -57.9153976, -6.1095772, -57.8301239, -6.3711748, -47.5965652, 47.8909149
35: -56.1316032, -4.1429443, -56.0762482, -4.3533010, -44.8557968, 45.1283073
36: -53.6146507, 0.9949770, -53.4952621, 0.8824015, -49.5196533, 49.5013428
37: -78.4475555, -14.1827555, -78.2858124, -14.2676477, -60.9140625, 60.8780746
38: -63.9752769, 0.5520248, -63.8230667, 0.4126301, -59.7873077, 59.7607117
39: -72.2843094, -8.0216751, -72.1470490, -8.1741152, -58.1228867, 58.1211395
40: -51.5117798, -6.0661950, -51.3679886, -6.2099953, -45.3017845, 45.3017921
41: -40.1727638, 12.3402367, -40.0611267, 12.2550468, -52.4278107, 52.4013634
42: -26.3220310, 12.0252666, -26.1773815, 11.9529819, -38.2750130, 38.2026482

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1759
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
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1554
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
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1540
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
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 940
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1358
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
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1600
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

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1651165, upper bound: 20.0851744
time: 84.76 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1093657, upper bound: 20.0940351
time: 58.31 seconds

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

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1901985, upper bound: 20.0942161
time: 85.55 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1015177
time: 50.66 seconds

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

Time for backsubstitution: 1.83 seconds

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
time: 57.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1417534
time: 50.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 109.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1069563
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1408032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1069563
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.1408032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1162141
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1499619
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1564538
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1901986
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1651165, upper bound: 20.0851744
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.0940342
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1651165, upper bound: 20.0851744
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1093657, upper bound: 20.0940351
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1901985, upper bound: 20.0942161
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1015177
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1901985, upper bound: 20.1344525
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 109.95
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1417534

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -37.5989189, 17.5620937, -37.5989189, 17.5620937, -55.1610107, 55.1610107
1: -11.8745222, 22.4637051, -11.8745222, 22.4637051, -34.3382263, 34.3382263
2: -9.6261940, 25.2707977, -9.6261940, 25.2707977, -34.8969917, 34.8969917
3: -9.4760361, 28.9416599, -9.4760361, 28.9416599, -38.1885757, 38.1885757
4: -16.5154419, 25.3492374, -16.5154419, 25.3492374, -41.7486267, 41.7486267
5: -7.3212476, 29.0064621, -7.3212476, 29.0064621, -35.9005203, 35.9005203
6: -38.2220116, 11.9612103, -38.2220116, 11.9612103, -50.1832199, 50.1832199
7: -11.0123682, 28.6480026, -11.0123682, 28.6480026, -38.4432983, 38.4432983
8: -21.1349449, 29.8487186, -21.1349449, 29.8487186, -50.5531921, 50.5531921
9: -13.7073612, 28.2657280, -13.7073612, 28.2657280, -41.9730911, 41.9730911
10: -22.0736523, 31.8604679, -22.0736523, 31.8604679, -53.9341202, 53.9341202
11: -23.6886024, 14.5673428, -23.6886024, 14.5673428, -38.2559433, 38.2559433
12: -44.2374840, 4.1527662, -44.2374840, 4.1527662, -44.8498840, 44.8498917
13: -37.4524574, 22.2384987, -37.4524574, 22.2384987, -59.3779068, 59.3779144
14: -64.8475037, 2.4479618, -64.8475037, 2.4479618, -67.2954636, 67.2954636
15: -21.7557564, 20.3212051, -21.7557564, 20.3212051, -42.0769615, 42.0769615
16: -23.4094410, 21.6651802, -23.4094410, 21.6651802, -45.0746231, 45.0746231
17: -58.3871994, -1.3708696, -58.3871994, -1.3708696, -55.7857666, 55.7857590
18: -35.8360634, 14.5843544, -35.8360634, 14.5843544, -50.4204178, 50.4204178
19: -26.4213905, 9.4049911, -26.4213905, 9.4049911, -35.8263817, 35.8263817
20: -21.5193825, 15.7978592, -21.5193825, 15.7978592, -37.3172417, 37.3172417
21: -27.2646561, 12.8552914, -27.2646561, 12.8552914, -40.1199493, 40.1199493
22: -32.0893784, 10.5813560, -32.0893784, 10.5813560, -42.6707344, 42.6707344
23: -24.5767708, 13.9697037, -24.5767708, 13.9697037, -38.5464745, 38.5464745
24: -30.7277985, 13.7169561, -30.7277985, 13.7169561, -44.4447556, 44.4447556
25: -28.8826866, 12.8575745, -28.8826866, 12.8575745, -41.7402611, 41.7402611
26: -41.0085449, 16.9045677, -41.0085449, 16.9045677, -57.9131126, 57.9131126
27: -26.0444717, 18.1738853, -26.0444717, 18.1738853, -44.2183571, 44.2183571
28: -25.0571518, 17.2565022, -25.0571518, 17.2565022, -42.3136520, 42.3136520
29: -27.6044350, 10.8601875, -27.6044350, 10.8601875, -38.2829285, 38.2829285
30: -26.8312435, 18.2666969, -26.8312435, 18.2666969, -45.0979385, 45.0979385
31: -35.3872070, 12.0281811, -35.3872070, 12.0281811, -47.4153900, 47.4153900
32: -35.2123184, 10.9173946, -35.2123184, 10.9173946, -45.6678772, 45.6678848
33: -63.6089973, -3.7995172, -63.6089973, -3.7995172, -55.1589890, 55.1589890
34: -57.7606049, -6.4049263, -57.7606049, -6.4049263, -47.4208450, 47.4208603
35: -56.0528374, -4.3786774, -56.0528374, -4.3786774, -44.7434082, 44.7434082
36: -53.4749641, 0.8211174, -53.4749641, 0.8211174, -49.3182297, 49.3182449
37: -78.2494507, -14.3313522, -78.2494507, -14.3313522, -60.6653748, 60.6653671
38: -63.7954254, 0.3370485, -63.7954254, 0.3370485, -59.5350647, 59.5350647
39: -72.1064301, -8.2155256, -72.1064301, -8.2155256, -57.8866806, 57.8866959
40: -51.3340683, -6.2337394, -51.3340683, -6.2337394, -45.1003304, 45.1003304
41: -40.0315475, 12.2172813, -40.0315475, 12.2172813, -52.2488289, 52.2488289
42: -26.1506424, 11.9056902, -26.1506424, 11.9056902, -38.0563316, 38.0563316

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
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
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1638
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
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1451
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
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
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
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0899973
time: 55.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1302383
time: 69.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -37.5989189, 17.5620937, -37.7103615, 17.6318359, -55.2307549, 55.2724533
1: -11.8745222, 22.4637051, -11.9657211, 22.5259247, -34.4004478, 34.4294281
2: -9.6261940, 25.2707977, -9.7466259, 25.3705330, -34.9967270, 35.0174255
3: -9.4760361, 28.9416599, -9.6143894, 29.0858650, -38.3318939, 38.3298111
4: -16.5154419, 25.3492374, -16.6643181, 25.5088539, -41.9088364, 41.8980484
5: -7.3212476, 29.0064621, -7.4572959, 29.1546898, -36.0500908, 36.0351219
6: -38.2220116, 11.9612103, -38.3294716, 12.0469923, -50.2690048, 50.2906799
7: -11.0123682, 28.6480026, -11.1485672, 28.7033806, -38.5027084, 38.5766220
8: -21.1349449, 29.8487186, -21.2822361, 29.9732609, -50.6814423, 50.6997299
9: -13.7073612, 28.2657280, -13.7752028, 28.3305779, -42.0379410, 42.0409317
10: -22.0736523, 31.8604679, -22.1959763, 31.9696083, -54.0432587, 54.0564423
11: -23.6886024, 14.5673428, -23.9798851, 14.7207966, -38.4094009, 38.5472260
12: -44.2374840, 4.1527662, -44.5266953, 4.4116354, -45.1080627, 45.1412430
13: -37.4524574, 22.2384987, -37.5690613, 22.3918762, -59.5255432, 59.5162048
14: -64.8475037, 2.4479618, -65.1376801, 2.6683207, -67.5158234, 67.5856400
15: -21.7557564, 20.3212051, -21.9252663, 20.5295944, -42.2853508, 42.2464714
16: -23.4094410, 21.6651802, -23.5628967, 21.7303352, -45.1397781, 45.2280769
17: -58.3871994, -1.3708696, -58.6855392, -1.1677999, -55.9876709, 56.1138687
18: -35.8360634, 14.5843544, -35.9162064, 14.6403675, -50.4764328, 50.5005608
19: -26.4213905, 9.4049911, -26.5564671, 9.4895496, -35.9109421, 35.9614563
20: -21.5193825, 15.7978592, -21.6477699, 15.8856583, -37.4050407, 37.4456291
21: -27.2646561, 12.8552914, -27.4552746, 12.9705515, -40.2352066, 40.3105659
22: -32.0893784, 10.5813560, -32.1707878, 10.6590414, -42.7484207, 42.7521439
23: -24.5767708, 13.9697037, -24.6922855, 14.0428152, -38.6195869, 38.6619873
24: -30.7277985, 13.7169561, -30.7950249, 13.7554350, -44.4832344, 44.5119820
25: -28.8826866, 12.8575745, -28.9712372, 12.9401140, -41.8227997, 41.8288116
26: -41.0085449, 16.9045677, -41.1629562, 17.0237999, -58.0323448, 58.0675240
27: -26.0444717, 18.1738853, -26.1496220, 18.2357693, -44.2802429, 44.3235092
28: -25.0571518, 17.2565022, -25.1558552, 17.3247604, -42.3819122, 42.4123573
29: -27.6044350, 10.8601875, -27.7175636, 10.9551792, -38.3780365, 38.3984222
30: -26.8312435, 18.2666969, -26.9309635, 18.3245258, -45.1557693, 45.1976624
31: -35.3872070, 12.0281811, -35.5318146, 12.1188564, -47.5060654, 47.5599976
32: -35.2123184, 10.9173946, -35.3452721, 11.0239277, -45.7751007, 45.8055725
33: -63.6089973, -3.7995172, -63.7141953, -3.6704903, -55.3165131, 55.2655411
34: -57.7606049, -6.4049263, -57.8608856, -6.2553644, -47.5994568, 47.5217590
35: -56.0528374, -4.3786774, -56.0976105, -4.2985907, -44.8487091, 44.7952461
36: -53.4749641, 0.8211174, -53.5825005, 0.9249563, -49.4243317, 49.4252014
37: -78.2494507, -14.3313522, -78.3756561, -14.2304573, -60.7514496, 60.7811737
38: -63.7954254, 0.3370485, -63.9154739, 0.4641418, -59.6589661, 59.6629639
39: -72.1064301, -8.2155256, -72.2177048, -8.1143332, -57.9865799, 58.0161514
40: -51.3340683, -6.2337394, -51.4203644, -6.1341543, -45.1999130, 45.1866264
41: -40.0315475, 12.2172813, -40.1123810, 12.2805481, -52.3120956, 52.3296623
42: -26.1506424, 11.9056902, -26.2430267, 11.9789267, -38.1295700, 38.1487160

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1787
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
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1638
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
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1451
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
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1359
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
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
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
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1234881
time: 78.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1637246
time: 54.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -37.6750107, 17.6151257, -37.5989189, 17.5620937, -55.2371063, 55.2140427
1: -11.9441967, 22.6080017, -11.8745222, 22.4637051, -34.4079018, 34.4825249
2: -9.6993847, 25.4314804, -9.6261940, 25.2707977, -34.9701843, 35.0576744
3: -9.5444984, 29.1303082, -9.4760361, 28.9416599, -38.2563400, 38.3845367
4: -16.6033516, 25.5383396, -16.5154419, 25.3492374, -41.8369675, 41.9384689
5: -7.4028144, 29.1602077, -7.3212476, 29.0064621, -35.9819946, 36.0558929
6: -38.3059692, 11.9983959, -38.2220116, 11.9612103, -50.2671814, 50.2204056
7: -11.1146984, 28.7617245, -11.0123682, 28.6480026, -38.5482559, 38.5645409
8: -21.2188911, 30.0439796, -21.1349449, 29.8487186, -50.6362686, 50.7496567
9: -13.8783083, 28.3473034, -13.7073612, 28.2657280, -42.1440353, 42.0546646
10: -22.4472141, 31.9725647, -22.0736523, 31.8604679, -54.3076820, 54.0462189
11: -23.9418316, 14.5907631, -23.6886024, 14.5673428, -38.5091743, 38.2793655
12: -44.5614738, 4.2830524, -44.2374840, 4.1527662, -45.1779480, 44.9785461
13: -37.4907913, 22.3289948, -37.4524574, 22.2384987, -59.4210892, 59.4556274
14: -65.2236633, 2.5560112, -64.8475037, 2.4479618, -67.6716232, 67.4035187
15: -21.7943459, 20.4278011, -21.7557564, 20.3212051, -42.1155510, 42.1835556
16: -23.6021652, 21.6974926, -23.4094410, 21.6651802, -45.2673454, 45.1069336
17: -58.6363564, -1.2900829, -58.3871994, -1.3708696, -56.0547180, 55.8567619
18: -36.0087433, 14.6283054, -35.8360634, 14.5843544, -50.5930977, 50.4643707
19: -26.5829277, 9.4457111, -26.4213905, 9.4049911, -35.9879189, 35.8671036
20: -21.6838379, 15.8408957, -21.5193825, 15.7978592, -37.4816971, 37.3602791
21: -27.4925137, 12.9198990, -27.2646561, 12.8552914, -40.3478050, 40.1845551
22: -32.1731262, 10.6316996, -32.0893784, 10.5813560, -42.7544823, 42.7210770
23: -24.6974163, 14.0095100, -24.5767708, 13.9697037, -38.6671219, 38.5862808
24: -30.7814617, 13.7326517, -30.7277985, 13.7169561, -44.4984169, 44.4604492
25: -28.9761238, 12.9116631, -28.8826866, 12.8575745, -41.8336983, 41.7943497
26: -41.2303772, 16.9992981, -41.0085449, 16.9045677, -58.1349449, 58.0078430
27: -26.1203671, 18.2365627, -26.0444717, 18.1738853, -44.2942505, 44.2810364
28: -25.1329784, 17.2895679, -25.0571518, 17.2565022, -42.3894806, 42.3467178
29: -27.7161636, 10.9022465, -27.6044350, 10.8601875, -38.3968353, 38.3264389
30: -26.9346581, 18.2974510, -26.8312435, 18.2666969, -45.2013550, 45.1286926
31: -35.5822296, 12.0716705, -35.3872070, 12.0281811, -47.6104126, 47.4588776
32: -35.3458748, 10.9750500, -35.2123184, 10.9173946, -45.7994843, 45.7202988
33: -63.6728134, -3.6114049, -63.6089973, -3.7995172, -55.2219009, 55.3933945
34: -57.8152237, -6.2588978, -57.7606049, -6.4049263, -47.4707260, 47.6126022
35: -56.0867882, -4.2232494, -56.0528374, -4.3786774, -44.7871323, 44.9524117
36: -53.5069580, 0.8912048, -53.4749641, 0.8211174, -49.3484116, 49.3895798
37: -78.3204651, -14.2841091, -78.2494507, -14.3313522, -60.7525787, 60.7328491
38: -63.8562012, 0.4250445, -63.7954254, 0.3370485, -59.5933685, 59.6246490
39: -72.1723099, -8.1215353, -72.1064301, -8.2155256, -57.9570541, 57.9870682
40: -51.4258842, -6.1652985, -51.3340683, -6.2337394, -45.1921463, 45.1687698
41: -40.0911179, 12.2773972, -40.0315475, 12.2172813, -52.3083992, 52.3089447
42: -26.2284737, 11.9522057, -26.1506424, 11.9056902, -38.1341629, 38.1028481

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1401
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1637
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
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
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
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1695
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
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
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
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 850

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0653180
time: 56.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1055620
time: 61.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -37.6750107, 17.6151257, -37.7103615, 17.6318359, -55.3068466, 55.3254852
1: -11.9441967, 22.6080017, -11.9657211, 22.5259247, -34.4701233, 34.5737228
2: -9.6993847, 25.4314804, -9.7466259, 25.3705330, -35.0699158, 35.1781082
3: -9.5444984, 29.1303082, -9.6143894, 29.0858650, -38.3996582, 38.5257683
4: -16.6033516, 25.5383396, -16.6643181, 25.5088539, -41.9971848, 42.0878868
5: -7.4028144, 29.1602077, -7.4572959, 29.1546898, -36.1315727, 36.1904945
6: -38.3059692, 11.9983959, -38.3294716, 12.0469923, -50.3529625, 50.3278656
7: -11.1146984, 28.7617245, -11.1485672, 28.7033806, -38.6076736, 38.6978683
8: -21.2188911, 30.0439796, -21.2822361, 29.9732609, -50.7645264, 50.8961868
9: -13.8783083, 28.3473034, -13.7752028, 28.3305779, -42.2088852, 42.1225052
10: -22.4472141, 31.9725647, -22.1959763, 31.9696083, -54.4168243, 54.1685410
11: -23.9418316, 14.5907631, -23.9798851, 14.7207966, -38.6626282, 38.5706482
12: -44.5614738, 4.2830524, -44.5266953, 4.4116354, -45.4361115, 45.2698975
13: -37.4907913, 22.3289948, -37.5690613, 22.3918762, -59.5687103, 59.5939178
14: -65.2236633, 2.5560112, -65.1376801, 2.6683207, -67.8919830, 67.6936951
15: -21.7943459, 20.4278011, -21.9252663, 20.5295944, -42.3239403, 42.3530655
16: -23.6021652, 21.6974926, -23.5628967, 21.7303352, -45.3325005, 45.2603912
17: -58.6363564, -1.2900829, -58.6855392, -1.1677999, -56.2566223, 56.1848717
18: -36.0087433, 14.6283054, -35.9162064, 14.6403675, -50.6491089, 50.5445099
19: -26.5829277, 9.4457111, -26.5564671, 9.4895496, -36.0724792, 36.0021782
20: -21.6838379, 15.8408957, -21.6477699, 15.8856583, -37.5694962, 37.4886665
21: -27.4925137, 12.9198990, -27.4552746, 12.9705515, -40.4630661, 40.3751755
22: -32.1731262, 10.6316996, -32.1707878, 10.6590414, -42.8321686, 42.8024864
23: -24.6974163, 14.0095100, -24.6922855, 14.0428152, -38.7402306, 38.7017975
24: -30.7814617, 13.7326517, -30.7950249, 13.7554350, -44.5368958, 44.5276756
25: -28.9761238, 12.9116631, -28.9712372, 12.9401140, -41.9162369, 41.8829002
26: -41.2303772, 16.9992981, -41.1629562, 17.0237999, -58.2541771, 58.1622543
27: -26.1203671, 18.2365627, -26.1496220, 18.2357693, -44.3561363, 44.3861847
28: -25.1329784, 17.2895679, -25.1558552, 17.3247604, -42.4577408, 42.4454231
29: -27.7161636, 10.9022465, -27.7175636, 10.9551792, -38.4919434, 38.4419403
30: -26.9346581, 18.2974510, -26.9309635, 18.3245258, -45.2591858, 45.2284164
31: -35.5822296, 12.0716705, -35.5318146, 12.1188564, -47.7010880, 47.6034851
32: -35.3458748, 10.9750500, -35.3452721, 11.0239277, -45.9066925, 45.8579865
33: -63.6728134, -3.6114049, -63.7141953, -3.6704903, -55.3794250, 55.4999390
34: -57.8152237, -6.2588978, -57.8608856, -6.2553644, -47.6493378, 47.7135162
35: -56.0867882, -4.2232494, -56.0976105, -4.2985907, -44.8924255, 45.0042496
36: -53.5069580, 0.8912048, -53.5825005, 0.9249563, -49.4544830, 49.4965363
37: -78.3204651, -14.2841091, -78.3756561, -14.2304573, -60.8386536, 60.8486557
38: -63.8562012, 0.4250445, -63.9154739, 0.4641418, -59.7172699, 59.7525406
39: -72.1723099, -8.1215353, -72.2177048, -8.1143332, -58.0569382, 58.1165161
40: -51.4258842, -6.1652985, -51.4203644, -6.1341543, -45.2917290, 45.2550659
41: -40.0911179, 12.2773972, -40.1123810, 12.2805481, -52.3716660, 52.3897781
42: -26.2284737, 11.9522057, -26.2430267, 11.9789267, -38.2074013, 38.1952324

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1401
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
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1637
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
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
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
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1695
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
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1542
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
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
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
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0991722
time: 53.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1394100
time: 42.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -37.5470581, 17.5360661, -37.6545868, 17.6013470, -55.1484070, 55.1906509
1: -11.8615799, 22.3914604, -11.9382629, 22.5753002, -34.4368820, 34.3297234
2: -9.6344433, 25.2080078, -9.6934128, 25.4033127, -35.0377579, 34.9014206
3: -9.4750595, 28.8060226, -9.5375290, 29.0700970, -38.3227158, 38.1125412
4: -16.5111485, 25.2401047, -16.5963001, 25.4900208, -41.8854904, 41.7203217
5: -7.3230343, 28.8960075, -7.3952103, 29.1108456, -36.0124435, 35.8627243
6: -38.1886635, 11.9430285, -38.2896690, 11.9907169, -50.1793823, 50.2326965
7: -11.0070934, 28.5529919, -11.1089058, 28.7192173, -38.5255280, 38.4462013
8: -21.1247005, 29.7533703, -21.2101917, 30.0014267, -50.7123108, 50.5252380
9: -13.6609411, 28.1970787, -13.8667946, 28.3027954, -41.9637375, 42.0638733
10: -22.0462856, 31.8955116, -22.4328156, 31.9594612, -54.0001373, 54.3283272
11: -23.5842628, 14.5463181, -23.8963070, 14.5843525, -38.1686172, 38.4426270
12: -44.1580505, 4.2001410, -44.5255661, 4.2762136, -44.8907928, 45.1920319
13: -37.4265251, 22.1502914, -37.4820862, 22.2869911, -59.3554382, 59.3346100
14: -64.7329102, 2.5013342, -65.1685944, 2.5522118, -67.2851257, 67.6699295
15: -21.7250710, 20.2357216, -21.7877407, 20.3878803, -42.1129532, 42.0234604
16: -23.3612366, 21.6226273, -23.5854111, 21.6735802, -45.0348167, 45.2080383
17: -58.2656174, -1.3970776, -58.5858688, -1.3052101, -55.7152557, 55.9748688
18: -35.7520370, 14.5607700, -35.9692993, 14.6184263, -50.3704643, 50.5300674
19: -26.3105984, 9.3931255, -26.5323658, 9.4402065, -35.7508049, 35.9254913
20: -21.4263115, 15.8012714, -21.6392479, 15.8371458, -37.2634583, 37.4405212
21: -27.1635704, 12.8707256, -27.4469967, 12.9170952, -40.0806656, 40.3177223
22: -31.9569473, 10.5546589, -32.1146774, 10.6246891, -42.5816345, 42.6693344
23: -24.4466610, 13.9464588, -24.6388302, 14.0035877, -38.4502487, 38.5852890
24: -30.5494766, 13.6749153, -30.7032318, 13.7254915, -44.2749672, 44.3781471
25: -28.7248840, 12.8354998, -28.9066525, 12.9056034, -41.6304855, 41.7421532
26: -40.9030304, 16.9170876, -41.1822510, 16.9927101, -57.8957405, 58.0993385
27: -25.9621773, 18.1550064, -26.0772705, 18.2315788, -44.1937561, 44.2322769
28: -24.9470329, 17.2324963, -25.0815125, 17.2838936, -42.2309265, 42.3140106
29: -27.4532928, 10.8441143, -27.6511459, 10.8964405, -38.1686707, 38.3173790
30: -26.6990509, 18.2435036, -26.8765030, 18.2912388, -44.9902878, 45.1200066
31: -35.2434998, 12.0110655, -35.5156517, 12.0645924, -47.3080902, 47.5267181
32: -35.1608696, 10.9233503, -35.3219147, 10.9690285, -45.6553192, 45.7970200
33: -63.5976830, -3.8139691, -63.6547890, -3.6186485, -55.3750076, 55.1869125
34: -57.7118759, -6.4315271, -57.7853508, -6.2663336, -47.5499191, 47.4147949
35: -56.0264015, -4.3986034, -56.0681686, -4.2289200, -44.9242477, 44.7409210
36: -53.4244957, 0.8048449, -53.4828873, 0.8868904, -49.3358002, 49.3096924
37: -78.1385422, -14.3488865, -78.2705688, -14.2893410, -60.6171417, 60.6836243
38: -63.7303162, 0.3015327, -63.8221970, 0.4151111, -59.5477753, 59.5222015
39: -72.0375290, -8.2196798, -72.1393204, -8.1265736, -57.9072189, 57.9431152
40: -51.2880325, -6.2536554, -51.4039078, -6.1727252, -45.1153069, 45.1502533
41: -40.0070839, 12.2078838, -40.0774460, 12.2714920, -52.2785759, 52.2853317
42: -26.1153736, 11.8955994, -26.2128410, 11.9469023, -38.0622749, 38.1084404

Time for backsubstitution: 1.81 seconds

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
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
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
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
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
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
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
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1603
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
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1414
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
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
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 740
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

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0726926
time: 51.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1142017
time: 44.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -37.5470581, 17.5360661, -37.7665176, 17.6711540, -55.2182121, 55.3025818
1: -11.8615799, 22.3914604, -12.0298834, 22.6375885, -34.4991684, 34.4213448
2: -9.6344433, 25.2080078, -9.8145628, 25.5030861, -35.1375275, 35.0225716
3: -9.4750595, 28.8060226, -9.6761112, 29.2144337, -38.4660568, 38.2541885
4: -16.5111485, 25.2401047, -16.7457123, 25.6497536, -42.0457687, 41.8700638
5: -7.3230343, 28.8960075, -7.5316548, 29.2592354, -36.1621857, 35.9975166
6: -38.1886635, 11.9430285, -38.3985596, 12.0764952, -50.2651596, 50.3415871
7: -11.0070934, 28.5529919, -11.2450762, 28.7746429, -38.5849838, 38.5791931
8: -21.1247005, 29.7533703, -21.3583431, 30.1262169, -50.8407898, 50.6722336
9: -13.6609411, 28.1970787, -13.9346733, 28.3676224, -42.0285645, 42.1317520
10: -22.0462856, 31.8955116, -22.5551796, 32.0704269, -54.1122131, 54.4506912
11: -23.5842628, 14.5463181, -24.1889954, 14.7379580, -38.3222198, 38.7353134
12: -44.1580505, 4.2001410, -44.8149414, 4.5350628, -45.1487885, 45.4834366
13: -37.4265251, 22.1502914, -37.5984459, 22.4396057, -59.5022583, 59.4726715
14: -64.7329102, 2.5013342, -65.4584808, 2.7719822, -67.5048904, 67.9598160
15: -21.7250710, 20.2357216, -21.9570866, 20.5989265, -42.3239975, 42.1928101
16: -23.3612366, 21.6226273, -23.7395477, 21.7394104, -45.1006470, 45.3621750
17: -58.2656174, -1.3970776, -58.8844910, -1.1023874, -55.9173203, 56.3030167
18: -35.7520370, 14.5607700, -36.0494461, 14.6746922, -50.4267273, 50.6102142
19: -26.3105984, 9.3931255, -26.6678715, 9.5254421, -35.8360405, 36.0609970
20: -21.4263115, 15.8012714, -21.7678070, 15.9250984, -37.3514099, 37.5690765
21: -27.1635704, 12.8707256, -27.6379719, 13.0328598, -40.1964302, 40.5086975
22: -31.9569473, 10.5546589, -32.1965256, 10.7029600, -42.6599083, 42.7511826
23: -24.4466610, 13.9464588, -24.7549477, 14.0772038, -38.5238647, 38.7014084
24: -30.5494766, 13.6749153, -30.7707977, 13.7640324, -44.3135071, 44.4457130
25: -28.7248840, 12.8354998, -28.9954224, 12.9882450, -41.7131271, 41.8309212
26: -40.9030304, 16.9170876, -41.3367233, 17.1116142, -58.0146446, 58.2538109
27: -25.9621773, 18.1550064, -26.1822586, 18.2934418, -44.2556190, 44.3372650
28: -24.9470329, 17.2324963, -25.1804104, 17.3525448, -42.2995758, 42.4129066
29: -27.4532928, 10.8441143, -27.7650337, 10.9916706, -38.2639618, 38.4335594
30: -26.6990509, 18.2435036, -26.9767952, 18.3486023, -45.0476532, 45.2202988
31: -35.2434998, 12.0110655, -35.6611977, 12.1558781, -47.3993759, 47.6722641
32: -35.1608696, 10.9233503, -35.4551392, 11.0755291, -45.7626114, 45.9349747
33: -63.5976830, -3.8139691, -63.7597694, -3.4898462, -55.5322723, 55.2931900
34: -57.7118759, -6.4315271, -57.8855019, -6.1170816, -47.7282562, 47.5158615
35: -56.0264015, -4.3986034, -56.1129608, -4.1485958, -45.0297318, 44.7926636
36: -53.4244957, 0.8048449, -53.5905838, 0.9906683, -49.4420624, 49.4168396
37: -78.1385422, -14.3488865, -78.3977280, -14.1879692, -60.7015839, 60.8015823
38: -63.7303162, 0.3015327, -63.9412537, 0.5421028, -59.6716461, 59.6497116
39: -72.0375290, -8.2196798, -72.2513123, -8.0267601, -58.0047913, 58.0745087
40: -51.2880325, -6.2536554, -51.4897690, -6.0736217, -45.2144089, 45.2361145
41: -40.0070839, 12.2078838, -40.1590500, 12.3343277, -52.3414116, 52.3669357
42: -26.1153736, 11.8955994, -26.3064384, 12.0199671, -38.1353416, 38.2020378

Time for backsubstitution: 1.79 seconds

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
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
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
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 546
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
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1436
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
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1603
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
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1414
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
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
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
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 740
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

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1064176
time: 50.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1479475
time: 56.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -37.6228485, 17.5709095, -37.6748619, 17.6150055, -55.2378540, 55.2457733
1: -11.9058628, 22.4677734, -11.9441519, 22.6077919, -34.5136566, 34.4119263
2: -9.6686707, 25.2739639, -9.6993380, 25.4312859, -35.0999565, 34.9733009
3: -9.5230742, 28.9455109, -9.5444393, 29.1299133, -38.4309769, 38.2522812
4: -16.5583916, 25.3525848, -16.6032677, 25.5380363, -41.9805374, 41.8333893
5: -7.3674693, 29.0104313, -7.4027500, 29.1599121, -36.1055717, 35.9756432
6: -38.2304077, 11.9675026, -38.3058472, 11.9983492, -50.2287560, 50.2733498
7: -11.0508232, 28.6512794, -11.1146221, 28.7614441, -38.6115646, 38.5380440
8: -21.1813087, 29.8533993, -21.2187996, 30.0436974, -50.8120422, 50.6259232
9: -13.7185116, 28.2987900, -13.8782482, 28.3470058, -42.0655174, 42.1770401
10: -22.0915985, 31.9343319, -22.4470978, 31.9724464, -54.0597839, 54.3814316
11: -23.6973877, 14.5838842, -23.9415245, 14.5907173, -38.2881050, 38.5254097
12: -44.2447472, 4.2374544, -44.5612411, 4.2829781, -44.9574738, 45.2660141
13: -37.4608688, 22.2469788, -37.4906960, 22.3284607, -59.4857178, 59.4517288
14: -64.8692017, 2.5292072, -65.2232513, 2.5559902, -67.4251938, 67.7524567
15: -21.7631378, 20.3320293, -21.7942772, 20.4275703, -42.1907082, 42.1263046
16: -23.4257355, 21.6800880, -23.6020660, 21.6973152, -45.1230507, 45.2821541
17: -58.3942986, -1.3325071, -58.6360245, -1.2902279, -55.8183594, 56.0920372
18: -35.8443260, 14.6088047, -36.0084610, 14.6282024, -50.4725266, 50.6172638
19: -26.4293690, 9.4304218, -26.5825825, 9.4456472, -35.8750153, 36.0130043
20: -21.5320053, 15.8288469, -21.6835461, 15.8408470, -37.3728523, 37.5123940
21: -27.2739124, 12.8948488, -27.4922104, 12.9198771, -40.1937904, 40.3870583
22: -32.0941391, 10.5927591, -32.1727180, 10.6316395, -42.7257767, 42.7654762
23: -24.5839462, 13.9893446, -24.6970234, 14.0094776, -38.5934219, 38.6863670
24: -30.7315598, 13.7201347, -30.7809505, 13.7326126, -44.4641724, 44.5010834
25: -28.8876228, 12.8797359, -28.9756966, 12.9116220, -41.7992439, 41.8554306
26: -41.0181160, 16.9628639, -41.2300797, 16.9992218, -58.0173378, 58.1929436
27: -26.0614414, 18.1789894, -26.1200523, 18.2365284, -44.2979698, 44.2990417
28: -25.0670433, 17.2726326, -25.1326370, 17.2895126, -42.3565559, 42.4052696
29: -27.6062889, 10.8787193, -27.7157288, 10.9022074, -38.3240814, 38.4168015
30: -26.8375950, 18.2800064, -26.9342690, 18.2974014, -45.1349945, 45.2142754
31: -35.3994102, 12.0594521, -35.5817795, 12.0716152, -47.4710236, 47.6412315
32: -35.2196426, 10.9434605, -35.3456116, 10.9750118, -45.7215118, 45.8427734
33: -63.6426163, -3.7851348, -63.6725197, -3.6114550, -55.4171448, 55.2363892
34: -57.7823334, -6.3932657, -57.8150215, -6.2589760, -47.6035004, 47.4846573
35: -56.0703354, -4.3677158, -56.0866547, -4.2232771, -44.9523849, 44.7959862
36: -53.4789047, 0.8283167, -53.5067329, 0.8911953, -49.3753662, 49.3578568
37: -78.2564316, -14.3178968, -78.3201447, -14.2841597, -60.7121277, 60.7662048
38: -63.8085403, 0.3447838, -63.8558426, 0.4249568, -59.6116333, 59.6010361
39: -72.1159058, -8.2048464, -72.1719055, -8.1215878, -57.9785919, 57.9921265
40: -51.3455811, -6.2294860, -51.4256897, -6.1654286, -45.1801529, 45.1962051
41: -40.0421867, 12.2283554, -40.0909996, 12.2773399, -52.3195267, 52.3193550
42: -26.1563568, 11.9129715, -26.2282791, 11.9521275, -38.1084824, 38.1412506

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1129351
time: 50.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1544453
time: 59.68 seconds

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

Time for backsubstitution: 1.77 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1466596
time: 275.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1881893
time: 48.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -37.7103615, 17.6318359, -37.5989189, 17.5620937, -55.2724533, 55.2307549
1: -11.9657211, 22.5259247, -11.8745222, 22.4637051, -34.4294281, 34.4004478
2: -9.7466259, 25.3705330, -9.6261940, 25.2707977, -35.0174255, 34.9967270
3: -9.6143894, 29.0858650, -9.4760361, 28.9416599, -38.3298111, 38.3318939
4: -16.6643181, 25.5088539, -16.5154419, 25.3492374, -41.8980484, 41.9088364
5: -7.4572959, 29.1546898, -7.3212476, 29.0064621, -36.0351257, 36.0500908
6: -38.3294716, 12.0469923, -38.2220116, 11.9612103, -50.2906799, 50.2690048
7: -11.1485672, 28.7033806, -11.0123682, 28.6480026, -38.5766220, 38.5027084
8: -21.2822361, 29.9732609, -21.1349449, 29.8487186, -50.6997223, 50.6814384
9: -13.7752028, 28.3305779, -13.7073612, 28.2657280, -42.0409317, 42.0379410
10: -22.1959763, 31.9696083, -22.0736523, 31.8604679, -54.0564423, 54.0432587
11: -23.9798851, 14.7207966, -23.6886024, 14.5673428, -38.5472260, 38.4094009
12: -44.5266953, 4.4116354, -44.2374840, 4.1527662, -45.1412354, 45.1080551
13: -37.5690613, 22.3918762, -37.4524574, 22.2384987, -59.5161972, 59.5255508
14: -65.1376801, 2.6683207, -64.8475037, 2.4479618, -67.5856400, 67.5158234
15: -21.9252663, 20.5295944, -21.7557564, 20.3212051, -42.2464714, 42.2853508
16: -23.5628967, 21.7303352, -23.4094410, 21.6651802, -45.2280769, 45.1397781
17: -58.6855392, -1.1677999, -58.3871994, -1.3708696, -56.1138763, 55.9876747
18: -35.9162064, 14.6403675, -35.8360634, 14.5843544, -50.5005608, 50.4764328
19: -26.5564671, 9.4895496, -26.4213905, 9.4049911, -35.9614563, 35.9109421
20: -21.6477699, 15.8856583, -21.5193825, 15.7978592, -37.4456291, 37.4050407
21: -27.4552746, 12.9705515, -27.2646561, 12.8552914, -40.3105659, 40.2352066
22: -32.1707878, 10.6590414, -32.0893784, 10.5813560, -42.7521439, 42.7484207
23: -24.6922855, 14.0428152, -24.5767708, 13.9697037, -38.6619873, 38.6195869
24: -30.7950249, 13.7554350, -30.7277985, 13.7169561, -44.5119820, 44.4832344
25: -28.9712372, 12.9401140, -28.8826866, 12.8575745, -41.8288116, 41.8227997
26: -41.1629562, 17.0237999, -41.0085449, 16.9045677, -58.0675240, 58.0323448
27: -26.1496220, 18.2357693, -26.0444717, 18.1738853, -44.3235092, 44.2802429
28: -25.1558552, 17.3247604, -25.0571518, 17.2565022, -42.4123573, 42.3819122
29: -27.7175636, 10.9551792, -27.6044350, 10.8601875, -38.3984222, 38.3780365
30: -26.9309635, 18.3245258, -26.8312435, 18.2666969, -45.1976624, 45.1557693
31: -35.5318146, 12.1188564, -35.3872070, 12.0281811, -47.5599976, 47.5060654
32: -35.3452721, 11.0239277, -35.2123184, 10.9173946, -45.8055725, 45.7751007
33: -63.7141953, -3.6704903, -63.6089973, -3.7995172, -55.2655258, 55.3165207
34: -57.8608856, -6.2553644, -57.7606049, -6.4049263, -47.5217667, 47.5994644
35: -56.0976105, -4.2985907, -56.0528374, -4.3786774, -44.7952499, 44.8487129
36: -53.5825005, 0.9249563, -53.4749641, 0.8211174, -49.4251938, 49.4243317
37: -78.3756561, -14.2304573, -78.2494507, -14.3313522, -60.7811737, 60.7514496
38: -63.9154739, 0.4641418, -63.7954254, 0.3370485, -59.6629639, 59.6589661
39: -72.2177048, -8.1143332, -72.1064301, -8.2155256, -58.0161514, 57.9865723
40: -51.4203644, -6.1341543, -51.3340683, -6.2337394, -45.1866264, 45.1999130
41: -40.1123810, 12.2805481, -40.0315475, 12.2172813, -52.3296623, 52.3120956
42: -26.2430267, 11.9789267, -26.1506424, 11.9056902, -38.1487160, 38.1295700

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1433
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
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
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
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 941
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
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
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
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1637241, upper bound: 20.0677367
time: 48.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1637241, upper bound: 20.1079732
time: 51.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -37.7103615, 17.6318359, -37.7103615, 17.6318359, -55.3421974, 55.3421974
1: -11.9657211, 22.5259247, -11.9657211, 22.5259247, -34.4916458, 34.4916458
2: -9.7466259, 25.3705330, -9.7466259, 25.3705330, -35.1171570, 35.1171570
3: -9.6143894, 29.0858650, -9.6143894, 29.0858650, -38.4498901, 38.4498901
4: -16.6643181, 25.5088539, -16.6643181, 25.5088539, -42.0429688, 42.0429688
5: -7.4572959, 29.1546898, -7.4572959, 29.1546898, -36.1662483, 36.1662521
6: -38.3294716, 12.0469923, -38.3294716, 12.0469923, -50.3764648, 50.3764648
7: -11.1485672, 28.7033806, -11.1485672, 28.7033806, -38.6172867, 38.6172905
8: -21.2822361, 29.9732609, -21.2822361, 29.9732609, -50.8101044, 50.8101196
9: -13.7752028, 28.3305779, -13.7752028, 28.3305779, -42.1057816, 42.1057816
10: -22.1959763, 31.9696083, -22.1959763, 31.9696083, -54.1655846, 54.1655846
11: -23.9798851, 14.7207966, -23.9798851, 14.7207966, -38.7006836, 38.7006836
12: -44.5266953, 4.4116354, -44.5266953, 4.4116354, -45.3427277, 45.3427277
13: -37.5690613, 22.3918762, -37.5690613, 22.3918762, -59.6129608, 59.6129608
14: -65.1376801, 2.6683207, -65.1376801, 2.6683207, -67.8059998, 67.8059998
15: -21.9252663, 20.5295944, -21.9252663, 20.5295944, -42.4548607, 42.4548607
16: -23.5628967, 21.7303352, -23.5628967, 21.7303352, -45.2932320, 45.2932320
17: -58.6855392, -1.1677999, -58.6855392, -1.1677999, -56.2177734, 56.2177734
18: -35.9162064, 14.6403675, -35.9162064, 14.6403675, -50.5565720, 50.5565720
19: -26.5564671, 9.4895496, -26.5564671, 9.4895496, -36.0460167, 36.0460167
20: -21.6477699, 15.8856583, -21.6477699, 15.8856583, -37.5334282, 37.5334282
21: -27.4552746, 12.9705515, -27.4552746, 12.9705515, -40.4258270, 40.4258270
22: -32.1707878, 10.6590414, -32.1707878, 10.6590414, -42.8298302, 42.8298302
23: -24.6922855, 14.0428152, -24.6922855, 14.0428152, -38.7350998, 38.7350998
24: -30.7950249, 13.7554350, -30.7950249, 13.7554350, -44.5504608, 44.5504608
25: -28.9712372, 12.9401140, -28.9712372, 12.9401140, -41.9113503, 41.9113503
26: -41.1629562, 17.0237999, -41.1629562, 17.0237999, -58.1867561, 58.1867561
27: -26.1496220, 18.2357693, -26.1496220, 18.2357693, -44.3853912, 44.3853912
28: -25.1558552, 17.3247604, -25.1558552, 17.3247604, -42.4806137, 42.4806137
29: -27.7175636, 10.9551792, -27.7175636, 10.9551792, -38.4849854, 38.4849854
30: -26.9309635, 18.3245258, -26.9309635, 18.3245258, -45.2554893, 45.2554893
31: -35.5318146, 12.1188564, -35.5318146, 12.1188564, -47.6506729, 47.6506729
32: -35.3452721, 11.0239277, -35.3452721, 11.0239277, -45.9203644, 45.9203720
33: -63.7141953, -3.6704903, -63.7141953, -3.6704903, -55.4429626, 55.4429550
34: -57.8608856, -6.2553644, -57.8608856, -6.2553644, -47.7291794, 47.7291794
35: -56.0976105, -4.2985907, -56.0976105, -4.2985907, -44.9427719, 44.9427719
36: -53.5825005, 0.9249563, -53.5825005, 0.9249563, -49.4797745, 49.4797745
37: -78.3756561, -14.2304573, -78.3756561, -14.2304573, -60.8624115, 60.8624191
38: -63.9154739, 0.4641418, -63.9154739, 0.4641418, -59.7185059, 59.7185059
39: -72.2177048, -8.1143332, -72.2177048, -8.1143332, -58.0857849, 58.0857849
40: -51.4203644, -6.1341543, -51.4203644, -6.1341543, -45.2862091, 45.2862091
41: -40.1123810, 12.2805481, -40.1123810, 12.2805481, -52.3929291, 52.3929291
42: -26.2430267, 11.9789267, -26.2430267, 11.9789267, -38.2219543, 38.2219543

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 755
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1433
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
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 518
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
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 941
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
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
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
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0749994
time: 48.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1079732
time: 55.25 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -37.7869415, 17.6849670, -37.5989189, 17.5620937, -55.3490372, 55.2838860
1: -12.0358181, 22.6702728, -11.8745222, 22.4637051, -34.4995232, 34.5447960
2: -9.8205633, 25.5312386, -9.6261940, 25.2707977, -35.0913620, 35.1574326
3: -9.6830425, 29.2746544, -9.4760361, 28.9416599, -38.3980179, 38.5278549
4: -16.7527771, 25.6980553, -16.5154419, 25.3492374, -41.9867249, 42.0987549
5: -7.5392194, 29.3086128, -7.3212476, 29.0064621, -36.1167831, 36.2056198
6: -38.4148369, 12.0841856, -38.2220116, 11.9612103, -50.3760452, 50.3061981
7: -11.2508888, 28.8171768, -11.0123682, 28.6480026, -38.6812744, 38.6239777
8: -21.3670597, 30.1687698, -21.1349449, 29.8487186, -50.7832565, 50.8781357
9: -13.9462061, 28.4120865, -13.7073612, 28.2657280, -42.2119331, 42.1194458
10: -22.5696926, 32.0835609, -22.0736523, 31.8604679, -54.4301605, 54.1572113
11: -24.2344532, 14.7444267, -23.6886024, 14.5673428, -38.8017960, 38.4330292
12: -44.8508148, 4.5418711, -44.2374840, 4.1527662, -45.4693298, 45.2365952
13: -37.6071968, 22.4815483, -37.4524574, 22.2384987, -59.5591507, 59.6024780
14: -65.5135117, 2.7758503, -64.8475037, 2.4479618, -67.9614716, 67.6233521
15: -21.9636936, 20.6388512, -21.7557564, 20.3212051, -42.2848969, 42.3946075
16: -23.7563953, 21.7632580, -23.4094410, 21.6651802, -45.4215775, 45.1726990
17: -58.9349403, -1.0872889, -58.3871994, -1.3708696, -56.3828430, 56.0588531
18: -36.0888824, 14.6846046, -35.8360634, 14.5843544, -50.6732368, 50.5206680
19: -26.7184391, 9.5309238, -26.4213905, 9.4049911, -36.1234283, 35.9523163
20: -21.8123398, 15.9288893, -21.5193825, 15.7978592, -37.6101990, 37.4482727
21: -27.6834335, 13.0356693, -27.2646561, 12.8552914, -40.5387268, 40.3003235
22: -32.2549133, 10.7099876, -32.0893784, 10.5813560, -42.8362694, 42.7993660
23: -24.8135300, 14.0831165, -24.5767708, 13.9697037, -38.7832336, 38.6598892
24: -30.8490391, 13.7711840, -30.7277985, 13.7169561, -44.5659943, 44.4989815
25: -29.0649223, 12.9943647, -28.8826866, 12.8575745, -41.9224968, 41.8770523
26: -41.3847961, 17.1182690, -41.0085449, 16.9045677, -58.2893639, 58.1268158
27: -26.2252922, 18.2984486, -26.0444717, 18.1738853, -44.3991776, 44.3429184
28: -25.2318401, 17.3582363, -25.0571518, 17.2565022, -42.4883423, 42.4153900
29: -27.8300304, 10.9975071, -27.6044350, 10.8601875, -38.5130234, 38.4217758
30: -27.0348873, 18.3548126, -26.8312435, 18.2666969, -45.3015823, 45.1860580
31: -35.7277298, 12.1629467, -35.3872070, 12.0281811, -47.7559128, 47.5501556
32: -35.4790726, 11.0814457, -35.2123184, 10.9173946, -45.9374084, 45.8275223
33: -63.7777939, -3.4826007, -63.6089973, -3.7995172, -55.3282089, 55.5506668
34: -57.9153976, -6.1095772, -57.7606049, -6.4049263, -47.5716629, 47.7909622
35: -56.1316032, -4.1429443, -56.0528374, -4.3786774, -44.8388443, 45.0579300
36: -53.6146507, 0.9949770, -53.4749641, 0.8211174, -49.4555283, 49.4958420
37: -78.4475555, -14.1827555, -78.2494507, -14.3313522, -60.8704987, 60.8173218
38: -63.9752769, 0.5520248, -63.7954254, 0.3370485, -59.7208099, 59.7485580
39: -72.2843094, -8.0216751, -72.1064301, -8.2155256, -58.0883408, 58.0846710
40: -51.5117798, -6.0661950, -51.3340683, -6.2337394, -45.2780418, 45.2678719
41: -40.1727638, 12.3402367, -40.0315475, 12.2172813, -52.3900452, 52.3717842
42: -26.3220310, 12.0252666, -26.1506424, 11.9056902, -38.2277222, 38.1759109

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 720
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
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
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

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1901981, upper bound: 20.0435439
time: 51.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0837815
time: 55.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -37.7869415, 17.6849670, -37.7103615, 17.6318359, -55.4187775, 55.3953285
1: -12.0358181, 22.6702728, -11.9657211, 22.5259247, -34.5617447, 34.6359940
2: -9.8205633, 25.5312386, -9.7466259, 25.3705330, -35.1910973, 35.2778625
3: -9.6830425, 29.2746544, -9.6143894, 29.0858650, -38.5181122, 38.6458511
4: -16.7527771, 25.6980553, -16.6643181, 25.5088539, -42.1316528, 42.2328949
5: -7.5392194, 29.3086128, -7.4572959, 29.1546898, -36.2479439, 36.3217850
6: -38.4148369, 12.0841856, -38.3294716, 12.0469923, -50.4618301, 50.4136581
7: -11.2508888, 28.8171768, -11.1485672, 28.7033806, -38.7219620, 38.7385597
8: -21.3670597, 30.1687698, -21.2822361, 29.9732609, -50.8936310, 51.0067978
9: -13.9462061, 28.4120865, -13.7752028, 28.3305779, -42.2767830, 42.1872902
10: -22.5696926, 32.0835609, -22.1959763, 31.9696083, -54.5392990, 54.2795372
11: -24.2344532, 14.7444267, -23.9798851, 14.7207966, -38.9552498, 38.7243118
12: -44.8508148, 4.5418711, -44.5266953, 4.4116354, -45.6708069, 45.4712677
13: -37.6071968, 22.4815483, -37.5690613, 22.3918762, -59.6558990, 59.6898956
14: -65.5135117, 2.7758503, -65.1376801, 2.6683207, -68.1818314, 67.9135284
15: -21.9636936, 20.6388512, -21.9252663, 20.5295944, -42.4932861, 42.5641174
16: -23.7563953, 21.7632580, -23.5628967, 21.7303352, -45.4867325, 45.3261566
17: -58.9349403, -1.0872889, -58.6855392, -1.1677999, -56.4867554, 56.2888374
18: -36.0888824, 14.6846046, -35.9162064, 14.6403675, -50.7292480, 50.6008110
19: -26.7184391, 9.5309238, -26.5564671, 9.4895496, -36.2079887, 36.0873909
20: -21.8123398, 15.9288893, -21.6477699, 15.8856583, -37.6979980, 37.5766602
21: -27.6834335, 13.0356693, -27.4552746, 12.9705515, -40.6539841, 40.4909439
22: -32.2549133, 10.7099876, -32.1707878, 10.6590414, -42.9139557, 42.8807755
23: -24.8135300, 14.0831165, -24.6922855, 14.0428152, -38.8563461, 38.7754021
24: -30.8490391, 13.7711840, -30.7950249, 13.7554350, -44.6044731, 44.5662079
25: -29.0649223, 12.9943647, -28.9712372, 12.9401140, -42.0050354, 41.9656029
26: -41.3847961, 17.1182690, -41.1629562, 17.0237999, -58.4085960, 58.2812271
27: -26.2252922, 18.2984486, -26.1496220, 18.2357693, -44.4610596, 44.4480705
28: -25.2318401, 17.3582363, -25.1558552, 17.3247604, -42.5566025, 42.5140915
29: -27.8300304, 10.9975071, -27.7175636, 10.9551792, -38.5995636, 38.5287323
30: -27.0348873, 18.3548126, -26.9309635, 18.3245258, -45.3594131, 45.2857742
31: -35.7277298, 12.1629467, -35.5318146, 12.1188564, -47.8465881, 47.6947632
32: -35.4790726, 11.0814457, -35.3452721, 11.0239277, -46.0522003, 45.9728088
33: -63.7777939, -3.4826007, -63.7141953, -3.6704903, -55.5056458, 55.6770859
34: -57.9153976, -6.1095772, -57.8608856, -6.2553644, -47.7790604, 47.9207153
35: -56.1316032, -4.1429443, -56.0976105, -4.2985907, -44.9863739, 45.1519775
36: -53.6146507, 0.9949770, -53.5825005, 0.9249563, -49.5099564, 49.5512466
37: -78.4475555, -14.1827555, -78.3756561, -14.2304573, -60.9503784, 60.9298859
38: -63.9752769, 0.5520248, -63.9154739, 0.4641418, -59.7763824, 59.8082199
39: -72.2843094, -8.0216751, -72.2177048, -8.1143332, -58.1574249, 58.1848373
40: -51.5117798, -6.0661950, -51.4203644, -6.1341543, -45.3776245, 45.3541679
41: -40.1727638, 12.3402367, -40.1123810, 12.2805481, -52.4533119, 52.4526176
42: -26.3220310, 12.0252666, -26.2430267, 11.9789267, -38.3009567, 38.2682953

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 935
type: A, layer: 1, pos: 1687
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
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1642
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
type: A, layer: 1, pos: 720
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
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1293
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 740
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
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0524070
time: 52.29 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1344526, upper bound: 20.0926433
time: 51.45 seconds

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

Time for backsubstitution: 1.76 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0506689
time: 44.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0922015
time: 59.25 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -37.6584930, 17.6058388, -37.7665176, 17.6711540, -55.3296471, 55.3723564
1: -11.9526653, 22.4536572, -12.0298834, 22.6375885, -34.5902557, 34.4835396
2: -9.7547369, 25.3077259, -9.8145628, 25.5030861, -35.2578239, 35.1222878
3: -9.6134033, 28.9502335, -9.6761112, 29.2144337, -38.5840073, 38.3743057
4: -16.6596565, 25.3996696, -16.7457123, 25.6497536, -42.1795197, 42.0149765
5: -7.4590597, 29.0442562, -7.5316548, 29.2592354, -36.2782974, 36.1286697
6: -38.2960663, 12.0287485, -38.3985596, 12.0764952, -50.3725624, 50.4273071
7: -11.1432791, 28.6083679, -11.2450762, 28.7746429, -38.6994858, 38.6198273
8: -21.2718506, 29.8779354, -21.3583431, 30.1262169, -50.9693298, 50.7826157
9: -13.7286978, 28.2617912, -13.9346733, 28.3676224, -42.0963211, 42.1964645
10: -22.1683121, 32.0042801, -22.5551796, 32.0704269, -54.2387390, 54.5594597
11: -23.8754463, 14.6997147, -24.1889954, 14.7379580, -38.6134033, 38.8887100
12: -44.4472656, 4.4588995, -44.8149414, 4.5350628, -45.3834534, 45.6848145
13: -37.5430527, 22.3044033, -37.5984459, 22.4396057, -59.5896301, 59.5697021
14: -65.0231018, 2.7215672, -65.4584808, 2.7719822, -67.7950821, 68.1800461
15: -21.8946953, 20.4439011, -21.9570866, 20.5989265, -42.4936218, 42.4009857
16: -23.5145607, 21.6881008, -23.7395477, 21.7394104, -45.2539711, 45.4276505
17: -58.5639610, -1.1940899, -58.8844910, -1.1023874, -56.1473770, 56.4069633
18: -35.8323593, 14.6167269, -36.0494461, 14.6746922, -50.5070496, 50.6661720
19: -26.4456406, 9.4776363, -26.6678715, 9.5254421, -35.9710846, 36.1455078
20: -21.5547485, 15.8889322, -21.7678070, 15.9250984, -37.4798470, 37.6567383
21: -27.3542156, 12.9858875, -27.6379719, 13.0328598, -40.3870773, 40.6238594
22: -32.0385399, 10.6324348, -32.1965256, 10.7029600, -42.7415009, 42.8289604
23: -24.5621834, 14.0194492, -24.7549477, 14.0772038, -38.6393890, 38.7743988
24: -30.6168098, 13.7133942, -30.7707977, 13.7640324, -44.3808441, 44.4841919
25: -28.8133545, 12.9178114, -28.9954224, 12.9882450, -41.8015976, 41.9132347
26: -41.0575180, 17.0362759, -41.3367233, 17.1116142, -58.1691322, 58.3730011
27: -26.0670910, 18.2167778, -26.1822586, 18.2934418, -44.3605347, 44.3990364
28: -25.0457306, 17.3005810, -25.1804104, 17.3525448, -42.3982773, 42.4809914
29: -27.5665169, 10.9389849, -27.7650337, 10.9916706, -38.3710327, 38.5199661
30: -26.7987862, 18.3014126, -26.9767952, 18.3486023, -45.1473885, 45.2782059
31: -35.3881721, 12.1017494, -35.6611977, 12.1558781, -47.5440521, 47.7629471
32: -35.2938309, 11.0297604, -35.4551392, 11.0755291, -45.9078903, 46.0497360
33: -63.7027512, -3.6850243, -63.7597694, -3.4898462, -55.6585541, 55.4704361
34: -57.8122482, -6.2820435, -57.8855019, -6.1170816, -47.8581314, 47.7232208
35: -56.0712242, -4.3186312, -56.1129608, -4.1485958, -45.1237793, 44.9400482
36: -53.5321198, 0.9088001, -53.5905838, 0.9906683, -49.4975662, 49.4714355
37: -78.2647476, -14.2481279, -78.3977280, -14.1879692, -60.8142395, 60.8814240
38: -63.8504906, 0.4286551, -63.9412537, 0.5421028, -59.7313690, 59.7052612
39: -72.1487579, -8.1184826, -72.2513123, -8.0267601, -58.1044922, 58.1431503
40: -51.3742981, -6.1540074, -51.4897690, -6.0736217, -45.3006744, 45.3357620
41: -40.0879173, 12.2710686, -40.1590500, 12.3343277, -52.4222450, 52.4301186
42: -26.2076416, 11.9688530, -26.3064384, 12.0199671, -38.2276077, 38.2752914

Time for backsubstitution: 1.86 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 733

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0579539
time: 43.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0995021
time: 50.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 95.93 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0899973
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1302383
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1234881
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1637246
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0653180
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1055620
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0991722
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1394100
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0726926
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1142017
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1064176
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1479475
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1129351
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1544453
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1466596
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.1881893
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1637241, upper bound: 20.0677367
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1637241, upper bound: 20.1079732
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0749994
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.1079732
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1901981, upper bound: 20.0435439
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0837815
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1079733, upper bound: 20.0524070
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1344526, upper bound: 20.0926433
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0506689
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1881894, upper bound: 20.0922015
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0579539
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 95.93
Output dim: 5, lower bound: -20.1324437, upper bound: 20.0995021
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 95.93
Output dim: 5, lower bound: -20.1901985, upper bound: 20.1344525
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 95.93
Output dim: 5, lower bound: -20.1344530, upper bound: 20.1417534

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 60.38 + 3614.30 = 3674.67 seconds
