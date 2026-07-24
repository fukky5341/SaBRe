## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 109.86210131
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490)
1: (-84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591)
2: (-73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443)
3: (-82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425)
4: (-85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136)
5: (-84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891)
6: (-104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552)
7: (-102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182)
8: (-98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266)
9: (-82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299)
10: (-117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769)
11: (-122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353)
12: (-113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700)
13: (-115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660)
14: (-172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130)
15: (-97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083)
16: (-131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306)
17: (-178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383)
18: (-114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708)
19: (-88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800)
20: (-78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332)
21: (-108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175)
22: (-112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821)
23: (-89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540)
24: (-107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015)
25: (-91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116)
26: (-127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797)
27: (-111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618)
28: (-86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463)
29: (-120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066)
30: (-107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661)
31: (-113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204)
32: (-101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892)
33: (-142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755)
34: (-121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955)
35: (-118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390)
36: (-111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718)
37: (-162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274)
38: (-145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403)
39: (-166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481)
40: (-141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322)
41: (-103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708)
42: (-79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822)

## BASE Result
execution time: IAR + LP analysis = 2.82 + 141.52 = 144.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -117.8238352, upper bound: 117.8238352


# Binary Search by BASE starts (time budget: 17855.66 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=170.32391357421875
rel_dist={4: [-113.45889199656516, 113.45889199845624]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=170.32391357421875
rel_dist={4: [-109.87288027651347, 109.87288028306807]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=170.32391357421875
rel_dist={4: [-106.78651155613684, 106.78651156283166]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=170.32391357421875
rel_dist={4: [-108.4037440570051, 108.40374405376852]}

## Binary Search Result
Binary search time: 2828.01 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 15027.65 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1753

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4197573, upper bound: 114.3510602
time: 572.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4197573, upper bound: 114.4197572
time: 170.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 743.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 743.33
Output dim: 4, lower bound: -114.4197573, upper bound: 114.3510602
IS_A2, status: Status.UNKNOWN, split count: 1, time: 743.33
Output dim: 4, lower bound: -114.4197573, upper bound: 114.4197572

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -138.5800323, 94.3328018, -138.8785858, 94.5090485, -233.0890503, 233.2113647
1: -84.0452118, 81.9322052, -84.2615814, 82.1875000, -166.2327118, 166.1937866
2: -73.4815826, 72.5476379, -73.6882248, 72.7993927, -146.2809448, 146.2358398
3: -81.7929459, 88.3473587, -82.0145569, 88.6302032, -170.4231567, 170.3619080
4: -84.8026276, 84.7828979, -85.0822372, 85.1047211, -169.9073334, 169.8651428
5: -84.7145691, 89.7864761, -84.9097672, 90.0006561, -174.7152252, 174.6962280
6: -104.0428009, 78.8762894, -104.3909454, 79.1105347, -183.1533356, 183.2672424
7: -101.8312836, 92.5378265, -102.0656815, 92.7728119, -194.6040802, 194.6035156
8: -97.8317947, 102.0174103, -98.1725006, 102.4713821, -200.3031616, 200.1899109
9: -82.5124054, 85.0542679, -82.6393433, 85.1861038, -167.6985016, 167.6936035
10: -116.8588791, 111.5304642, -117.0361481, 111.7579803, -228.6168060, 228.5666046
11: -122.3995972, 93.2638397, -122.7482452, 93.4516754, -215.8512726, 216.0120850
12: -112.7327042, 103.5859146, -112.9759140, 103.8463058, -216.5790100, 216.5618286
13: -115.0631256, 125.8430557, -115.2837448, 126.2240295, -241.2871552, 241.1267853
14: -171.9501953, 93.9311066, -172.2278137, 94.1523743, -266.1025391, 266.1589050
15: -96.9255219, 83.5481491, -97.1362610, 83.7723541, -180.6978760, 180.6843719
16: -131.5195923, 99.7394028, -131.8077087, 99.9265442, -231.4461365, 231.5471039
17: -178.1523132, 130.0074921, -178.3969269, 130.1969604, -308.3492737, 308.4044189
18: -114.2605362, 96.4115753, -114.6407928, 96.7770081, -211.0375366, 211.0523682
19: -88.2060852, 56.2233849, -88.4339752, 56.3700562, -144.5761414, 144.6573639
20: -78.3180695, 65.9524689, -78.4616318, 66.1002655, -144.4183044, 144.4140930
21: -108.4516296, 69.7487183, -108.6909943, 69.8880157, -178.3396149, 178.4397125
22: -112.3793259, 73.5788803, -112.5930939, 73.6909790, -186.0703125, 186.1719666
23: -89.3072815, 67.2667084, -89.5472031, 67.4617004, -156.7689819, 156.8139038
24: -107.5138702, 62.7410088, -107.7802811, 62.8355675, -170.3494415, 170.5212708
25: -90.8351593, 74.4861069, -91.0302277, 74.6011353, -165.4362946, 165.5162964
26: -127.2234497, 109.6227875, -127.5223846, 109.8760071, -237.0994263, 237.1451569
27: -111.0378952, 73.9102402, -111.2664948, 74.0251312, -185.0630188, 185.1767273
28: -86.1485062, 76.0328903, -86.3622818, 76.2240143, -162.3724976, 162.3951569
29: -119.8755264, 73.6089325, -120.1192627, 73.7198715, -193.5953979, 193.7281952
30: -107.6216736, 83.2056885, -107.8651199, 83.4272461, -191.0489044, 191.0708008
31: -113.3913727, 68.9537659, -113.7434998, 69.1381531, -182.5295258, 182.6972656
32: -100.9048462, 78.5883026, -101.1509552, 78.7720261, -179.6768799, 179.7392578
33: -141.8261719, 112.3337708, -142.0153809, 112.5131302, -254.3392639, 254.3491516
34: -121.3448868, 88.9165649, -121.6043625, 89.1421204, -210.4869995, 210.5209351
35: -118.4033508, 88.3695984, -118.6188583, 88.5240784, -206.9274292, 206.9884338
36: -111.1933212, 89.6344376, -111.3774109, 89.7733231, -200.9666443, 201.0118408
37: -162.3024597, 98.1968231, -162.7702637, 98.5065918, -260.8090515, 260.9671021
38: -145.4420776, 114.2952805, -145.6450500, 114.4562607, -259.8983154, 259.9403076
39: -165.7766418, 110.8459473, -165.9932861, 110.9417114, -276.7183533, 276.8392334
40: -140.9156799, 92.1817017, -141.3275452, 92.5013123, -233.4169769, 233.5092468
41: -103.1066437, 76.3266754, -103.4474640, 76.5269165, -179.6335602, 179.7741394
42: -79.6109467, 73.8389130, -79.8196335, 74.0321503, -153.6430969, 153.6585388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4050419, upper bound: 114.2944461
time: 154.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4053761, upper bound: 114.3366289
time: 385.18 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -138.9136963, 94.5485229, -138.9372559, 94.5661011, -233.4797974, 233.4857635
1: -84.2710114, 82.2552795, -84.2783890, 82.2749023, -166.5459137, 166.5336609
2: -73.6965103, 72.8657379, -73.7028427, 72.8859253, -146.5824280, 146.5685730
3: -82.0218506, 88.7029419, -82.0279694, 88.7247772, -170.7466278, 170.7309113
4: -85.0892181, 85.1925964, -85.0971832, 85.2165985, -170.3058167, 170.2897797
5: -84.9159164, 90.0528564, -84.9237518, 90.0711212, -174.9870300, 174.9766083
6: -104.4795837, 79.1233521, -104.5083847, 79.1352997, -183.6148834, 183.6317139
7: -102.0740662, 92.8300858, -102.0839386, 92.8529968, -194.9270630, 194.9140167
8: -98.1804047, 102.5947571, -98.1878815, 102.6273422, -200.8077393, 200.7826385
9: -82.6520386, 85.2095337, -82.6610031, 85.2221222, -167.8741608, 167.8705444
10: -117.0635605, 111.7780838, -117.0753021, 111.7888489, -228.8523865, 228.8533936
11: -122.8321457, 93.4648590, -122.8598251, 93.4782944, -216.3104095, 216.3246765
12: -113.0360870, 103.8601151, -113.0579147, 103.8710251, -216.9070740, 216.9180298
13: -115.3009872, 126.3173752, -115.3136139, 126.3441544, -241.6451416, 241.6309509
14: -172.2617035, 94.2005234, -172.2794342, 94.2185135, -266.4801941, 266.4799194
15: -97.1568909, 83.8023682, -97.1728058, 83.8345337, -180.9914246, 180.9751740
16: -131.8388062, 99.9420700, -131.8813171, 99.9529419, -231.7917480, 231.8233643
17: -178.4376831, 130.2258148, -178.4579315, 130.2394104, -308.6770935, 308.6837158
18: -114.7386169, 96.7890930, -114.7645569, 96.7972031, -211.5357971, 211.5536499
19: -88.4870605, 56.3765488, -88.5041809, 56.3818817, -144.8689423, 144.8807373
20: -78.4876404, 66.1168823, -78.5014191, 66.1255188, -144.6131592, 144.6182861
21: -108.7421265, 69.8975220, -108.7599487, 69.9045105, -178.6466064, 178.6574707
22: -112.6252823, 73.7019882, -112.6454697, 73.7112274, -186.3365021, 186.3474426
23: -89.6050339, 67.4740906, -89.6225586, 67.4821243, -157.0871582, 157.0966492
24: -107.8387299, 62.8428612, -107.8573990, 62.8497810, -170.6885071, 170.7002563
25: -91.0613480, 74.6133728, -91.0825806, 74.6226807, -165.6840210, 165.6959534
26: -127.5917358, 109.8875504, -127.6166992, 109.8999710, -237.4916687, 237.5042267
27: -111.3071518, 74.0376892, -111.3251495, 74.0446320, -185.3517761, 185.3628235
28: -86.4127808, 76.2394562, -86.4294739, 76.2466431, -162.6594238, 162.6689148
29: -120.1568069, 73.7280731, -120.1767731, 73.7346802, -193.8914795, 193.9048462
30: -107.9137802, 83.4463654, -107.9310608, 83.4575424, -191.3713226, 191.3774261
31: -113.8268890, 69.1465073, -113.8527679, 69.1537933, -182.9806824, 182.9992676
32: -101.2048111, 78.7892685, -101.2244186, 78.7985687, -180.0033875, 180.0136719
33: -142.0460815, 112.5240021, -142.0673523, 112.5338745, -254.5799408, 254.5913544
34: -121.6681976, 89.1509094, -121.6860199, 89.1593628, -210.8275452, 210.8369293
35: -118.6689911, 88.5320892, -118.6864548, 88.5389175, -207.2079163, 207.2185364
36: -111.4184189, 89.7789917, -111.4337769, 89.7834549, -201.2018280, 201.2127686
37: -162.8889923, 98.5133057, -162.9231262, 98.5196991, -261.4086914, 261.4364319
38: -145.6770935, 114.4703064, -145.6979828, 114.4787369, -260.1558228, 260.1682739
39: -166.0265808, 110.9531860, -166.0439148, 110.9658356, -276.9924316, 276.9971008
40: -141.4282227, 92.5099487, -141.4586487, 92.5221558, -233.9503479, 233.9685822
41: -103.5363464, 76.5381927, -103.5630188, 76.5490570, -180.0853882, 180.1011963
42: -79.8706284, 74.0463562, -79.8895569, 74.0581284, -153.9287567, 153.9359131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4050419, upper bound: 114.3631995
time: 304.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4053761, upper bound: 114.4053759
time: 185.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 492.20 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 492.20
Output dim: 4, lower bound: -114.4050419, upper bound: 114.2944461
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 492.20
Output dim: 4, lower bound: -114.4053761, upper bound: 114.3366289
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 492.20
Output dim: 4, lower bound: -114.4050419, upper bound: 114.3631995
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 492.20
Output dim: 4, lower bound: -114.4053761, upper bound: 114.4053759

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -138.5454102, 94.3247147, -138.6080017, 94.4455566, -232.9909668, 232.9327087
1: -84.0202789, 81.9261780, -84.0672684, 82.1402435, -166.1605225, 165.9934387
2: -73.4431000, 72.5414886, -73.3871155, 72.7513199, -146.1944275, 145.9286041
3: -81.7492218, 88.3374176, -81.6726913, 88.5521851, -170.3014069, 170.0101013
4: -84.7611542, 84.7747650, -84.7578430, 85.0409698, -169.8021240, 169.5325928
5: -84.6724091, 89.7772369, -84.5794144, 89.9279785, -174.6003876, 174.3566437
6: -104.0268402, 78.8583221, -104.2657318, 78.9701691, -182.9970093, 183.1240540
7: -101.7942963, 92.5306396, -101.7764282, 92.7164993, -194.5108032, 194.3070679
8: -97.7884369, 102.0086594, -97.8331909, 102.4026260, -200.1910553, 199.8418579
9: -82.5013504, 85.0239716, -82.5530396, 84.9498749, -167.4512177, 167.5769958
10: -116.8414459, 111.4532242, -116.8992386, 111.1529770, -227.9944153, 228.3524475
11: -122.3854218, 93.2061844, -122.6373901, 93.0000992, -215.3854980, 215.8435669
12: -112.7228241, 103.5058594, -112.8984528, 103.2173004, -215.9401245, 216.4043121
13: -115.0480881, 125.8208618, -115.1662064, 126.0485077, -241.0965881, 240.9870605
14: -171.9277344, 93.8788300, -172.0520935, 93.7413330, -265.6690674, 265.9309082
15: -96.8951111, 83.5333099, -96.9043961, 83.6554413, -180.5505524, 180.4376831
16: -131.4987488, 99.6996765, -131.6436768, 99.6178131, -231.1165619, 231.3433533
17: -178.1387329, 129.9420166, -178.2911682, 129.6824646, -307.8211975, 308.2331543
18: -114.2437439, 96.3775787, -114.5088654, 96.5105133, -210.7542419, 210.8864441
19: -88.1935577, 56.2032089, -88.3357239, 56.2125778, -144.4061279, 144.5389404
20: -78.3058624, 65.9339905, -78.3662033, 65.9558105, -144.2616577, 144.3002014
21: -108.4382324, 69.7151184, -108.5860214, 69.6249084, -178.0631256, 178.3011475
22: -112.3643646, 73.5462341, -112.4771805, 73.4356308, -185.7999878, 186.0234070
23: -89.2961426, 67.2488785, -89.4595947, 67.3249283, -156.6210632, 156.7084656
24: -107.4983673, 62.7348442, -107.6592484, 62.7875977, -170.2859497, 170.3940735
25: -90.8244934, 74.4666748, -90.9466019, 74.4493561, -165.2738342, 165.4132690
26: -127.2085571, 109.5638123, -127.4059906, 109.4137650, -236.6223145, 236.9698029
27: -111.0125961, 73.9037018, -111.0687714, 73.9742508, -184.9868469, 184.9724731
28: -86.1354370, 76.0235748, -86.2600327, 76.1528168, -162.2882538, 162.2835999
29: -119.8635712, 73.5694656, -120.0259247, 73.4104538, -193.2740173, 193.5953979
30: -107.6097336, 83.1778641, -107.7710495, 83.2097778, -190.8195038, 190.9489136
31: -113.3734512, 68.9334183, -113.6029663, 68.9790802, -182.3525085, 182.5363770
32: -100.8931122, 78.5641861, -101.0591965, 78.5834503, -179.4765625, 179.6233826
33: -141.7882996, 112.3218002, -141.7176208, 112.4196854, -254.2079773, 254.0394287
34: -121.3137360, 88.9045563, -121.3598404, 89.0479355, -210.3616638, 210.2643890
35: -118.3679428, 88.3605957, -118.3409576, 88.4536057, -206.8215485, 206.7015533
36: -111.1708221, 89.6240463, -111.2013245, 89.6928864, -200.8637085, 200.8253784
37: -162.2807312, 98.1802597, -162.6003418, 98.3778229, -260.6585693, 260.7805786
38: -145.4083862, 114.2843781, -145.3825531, 114.3705673, -259.7789612, 259.6669312
39: -165.7494507, 110.8368683, -165.7821350, 110.8711090, -276.6205444, 276.6190186
40: -140.8893127, 92.1742096, -141.1226044, 92.4428177, -233.3321228, 233.2968140
41: -103.0905685, 76.3122559, -103.3214111, 76.4169922, -179.5075531, 179.6336670
42: -79.5995789, 73.8015747, -79.7304077, 73.7412491, -153.3408203, 153.5319824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4026165, upper bound: 114.2469327
time: 120.39 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4026165, upper bound: 114.2923591
time: 343.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -138.5610809, 94.3284607, -138.9206543, 94.6683044, -233.2293854, 233.2491150
1: -84.0313263, 81.9289703, -84.2739105, 82.2643585, -166.2956848, 166.2028809
2: -73.4647598, 72.5435333, -73.6907730, 73.0116882, -146.4764404, 146.2343140
3: -81.7730713, 88.3404922, -82.0076904, 88.8550110, -170.6280670, 170.3481750
4: -84.7840118, 84.7768555, -85.0916061, 85.2946014, -170.0785980, 169.8684692
5: -84.6993256, 89.7803040, -84.9247055, 90.2409668, -174.9402924, 174.7050171
6: -104.0326996, 78.8487320, -104.4640350, 79.1003571, -183.1330566, 183.3127747
7: -101.8139954, 92.5327454, -102.1095581, 92.8667603, -194.6807404, 194.6423035
8: -97.8121414, 102.0120163, -98.1740112, 102.6854858, -200.4976196, 200.1860199
9: -82.5052338, 85.0409546, -82.7209320, 85.2220917, -167.7272949, 167.7618866
10: -116.8487930, 111.4949570, -117.4033279, 111.7267303, -228.5755310, 228.8982849
11: -122.3893051, 93.2394333, -123.0516663, 93.4236679, -215.8129730, 216.2911072
12: -112.7252350, 103.5547562, -113.4862976, 103.8256912, -216.5509338, 217.0410461
13: -115.0377426, 125.8314896, -115.2689667, 126.3084183, -241.3461456, 241.1004333
14: -171.9365234, 93.9119186, -172.5216370, 94.1406860, -266.0772095, 266.4335327
15: -96.8907242, 83.5386581, -97.1390762, 83.8608551, -180.7515869, 180.6777344
16: -131.5055542, 99.7122803, -131.9258728, 99.9419556, -231.4475098, 231.6381531
17: -178.1432648, 129.9803467, -178.6957550, 130.1886597, -308.3319092, 308.6760864
18: -114.2507019, 96.3960724, -114.8192749, 96.7875290, -211.0382080, 211.2153473
19: -88.1989670, 56.2130089, -88.5662842, 56.3704224, -144.5693970, 144.7792969
20: -78.3114624, 65.9432449, -78.6112366, 66.0993652, -144.4108124, 144.5544586
21: -108.4428864, 69.7348709, -108.9391479, 69.8867340, -178.3295898, 178.6740112
22: -112.3636551, 73.5631561, -112.6548920, 73.7136841, -186.0773315, 186.2180481
23: -89.3007965, 67.2536545, -89.6420288, 67.4631271, -156.7639160, 156.8956909
24: -107.5017929, 62.7341766, -107.8321075, 62.8427505, -170.3445435, 170.5662842
25: -90.8260803, 74.4763107, -91.0840759, 74.6250610, -165.4511108, 165.5603943
26: -127.2128448, 109.5986176, -127.8755264, 109.8795395, -237.0923615, 237.4741516
27: -111.0231171, 73.9021530, -111.3059082, 74.0350189, -185.0581207, 185.2080688
28: -86.1399078, 76.0237350, -86.4170532, 76.2447739, -162.3846741, 162.4407806
29: -119.8657379, 73.5921173, -120.1896286, 73.7130127, -193.5787354, 193.7817383
30: -107.6131134, 83.1825943, -107.9413452, 83.4213867, -191.0345001, 191.1239319
31: -113.3822403, 68.9424057, -113.9018936, 69.1416016, -182.5238342, 182.8442993
32: -100.8966675, 78.5762024, -101.2519913, 78.7808304, -179.6774902, 179.8281708
33: -141.8101807, 112.3263474, -142.0281372, 112.6945953, -254.5047760, 254.3544922
34: -121.3315964, 88.9088135, -121.6174545, 89.2672195, -210.5988159, 210.5262604
35: -118.3867264, 88.3642960, -118.6089096, 88.7309113, -207.1176453, 206.9732056
36: -111.1800079, 89.6287918, -111.3941345, 89.8413010, -201.0212860, 201.0229187
37: -162.2870789, 98.1807022, -162.8252869, 98.5314102, -260.8184509, 261.0059814
38: -145.4242401, 114.2865982, -145.6865540, 114.5172195, -259.9414673, 259.9731445
39: -165.7585449, 110.8403931, -166.0311279, 111.0362930, -276.7948303, 276.8714905
40: -140.9014740, 92.1643143, -141.3858643, 92.5149841, -233.4164429, 233.5501709
41: -103.0974426, 76.3080597, -103.4883652, 76.5416565, -179.6390991, 179.7964172
42: -79.6036453, 73.8139191, -79.9926758, 74.0285645, -153.6322021, 153.8065796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4029484, upper bound: 114.2892853
time: 286.09 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4029484, upper bound: 114.3345411
time: 148.48 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -138.8790283, 94.5404510, -138.6667480, 94.5026321, -233.3816528, 233.2071991
1: -84.2460785, 82.2492752, -84.0840759, 82.2276840, -166.4737549, 166.3333435
2: -73.6580048, 72.8596344, -73.4017868, 72.8378525, -146.4958496, 146.2614136
3: -81.9781418, 88.6930008, -81.6861115, 88.6467438, -170.6248779, 170.3791199
4: -85.0477142, 85.1844635, -84.7727966, 85.1528473, -170.2005463, 169.9572601
5: -84.8737793, 90.0436096, -84.5933990, 89.9984360, -174.8722076, 174.6370087
6: -104.4636230, 79.1054001, -104.3831329, 78.9949341, -183.4585571, 183.4885254
7: -102.0370560, 92.8228912, -101.7946701, 92.7966766, -194.8337097, 194.6175537
8: -98.1370468, 102.5859909, -97.8485565, 102.5585861, -200.6956177, 200.4345398
9: -82.6409607, 85.1792755, -82.5746536, 84.9859238, -167.6268921, 167.7539368
10: -117.0460892, 111.7008057, -116.9382553, 111.1838531, -228.2299194, 228.6390533
11: -122.8179779, 93.4072113, -122.7489624, 93.0267181, -215.8446960, 216.1561737
12: -113.0261765, 103.7799988, -112.9804230, 103.2420502, -216.2682190, 216.7604218
13: -115.2858887, 126.2952118, -115.1959915, 126.1686172, -241.4544983, 241.4911804
14: -172.2391968, 94.1482697, -172.1036682, 93.8075333, -266.0467224, 266.2519531
15: -97.1264496, 83.7875214, -96.9409561, 83.7175751, -180.8440247, 180.7284698
16: -131.8179626, 99.9023056, -131.7173309, 99.6441956, -231.4621582, 231.6196289
17: -178.4241486, 130.1603699, -178.3521729, 129.7249146, -308.1490479, 308.5125122
18: -114.7218323, 96.7550964, -114.6326065, 96.5307770, -211.2525940, 211.3876953
19: -88.4745331, 56.3563576, -88.4059372, 56.2244263, -144.6989594, 144.7622986
20: -78.4754486, 66.0983887, -78.4060287, 65.9810562, -144.4564819, 144.5043945
21: -108.7287445, 69.8639069, -108.6549377, 69.6413879, -178.3701172, 178.5188446
22: -112.6103210, 73.6693268, -112.5295258, 73.4558563, -186.0661774, 186.1988525
23: -89.5938873, 67.4562225, -89.5349121, 67.3453751, -156.9392395, 156.9911194
24: -107.8231964, 62.8366776, -107.7364044, 62.8017731, -170.6249695, 170.5730896
25: -91.0506744, 74.5939560, -90.9989471, 74.4708862, -165.5215607, 165.5928955
26: -127.5768509, 109.8286209, -127.5002899, 109.4377365, -237.0145569, 237.3288727
27: -111.2818680, 74.0311432, -111.1274796, 73.9937439, -185.2756042, 185.1586304
28: -86.3997345, 76.2301483, -86.3272324, 76.1754379, -162.5751648, 162.5573730
29: -120.1448517, 73.6885681, -120.0834198, 73.4252625, -193.5700989, 193.7719879
30: -107.9018250, 83.4185333, -107.8370361, 83.2400513, -191.1418610, 191.2555695
31: -113.8089905, 69.1261368, -113.7122879, 68.9947052, -182.8036957, 182.8384247
32: -101.1930771, 78.7651291, -101.1326828, 78.6099701, -179.8030243, 179.8978119
33: -142.0082397, 112.5119705, -141.7696381, 112.4403381, -254.4485779, 254.2816162
34: -121.6370163, 89.1388245, -121.4415283, 89.0651474, -210.7021637, 210.5803528
35: -118.6335602, 88.5231018, -118.4085312, 88.4683304, -207.1018982, 206.9316254
36: -111.3959122, 89.7685928, -111.2577591, 89.7030029, -201.0989075, 201.0263519
37: -162.8672791, 98.4967041, -162.7531891, 98.3909073, -261.2581787, 261.2498779
38: -145.6434326, 114.4593811, -145.4355774, 114.3930130, -260.0364380, 259.8949280
39: -165.9993591, 110.9441833, -165.8327942, 110.8952179, -276.8945923, 276.7769775
40: -141.4018707, 92.5024185, -141.2538147, 92.4635925, -233.8654633, 233.7562256
41: -103.5202560, 76.5237579, -103.4369812, 76.4390869, -179.9593506, 179.9607239
42: -79.8592377, 74.0090027, -79.8003540, 73.7672272, -153.6264496, 153.8093567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4026165, upper bound: 114.3195118
time: 216.03 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4026165, upper bound: 114.3607483
time: 639.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -138.8946838, 94.5442200, -138.9793396, 94.7254333, -233.6201019, 233.5235596
1: -84.2570877, 82.2520599, -84.2906647, 82.3517761, -166.6088562, 166.5427246
2: -73.6796875, 72.8616714, -73.7054138, 73.0982285, -146.7779236, 146.5670776
3: -82.0019531, 88.6960678, -82.0211029, 88.9496002, -170.9515381, 170.7171631
4: -85.0705872, 85.1865540, -85.1065674, 85.4065094, -170.4770966, 170.2931061
5: -84.9006958, 90.0466919, -84.9387131, 90.3114090, -175.2120972, 174.9854126
6: -104.4694824, 79.0957794, -104.5814514, 79.1251068, -183.5945892, 183.6772308
7: -102.0567856, 92.8250046, -102.1278000, 92.9469528, -195.0037384, 194.9528046
8: -98.1607208, 102.5893173, -98.1893692, 102.8414917, -201.0021973, 200.7786713
9: -82.6448441, 85.1962280, -82.7425995, 85.2581024, -167.9029541, 167.9388275
10: -117.0534668, 111.7425766, -117.4424896, 111.7576065, -228.8110657, 229.1850433
11: -122.8218536, 93.4404755, -123.1632767, 93.4502716, -216.2721252, 216.6037292
12: -113.0286179, 103.8289337, -113.5682602, 103.8504028, -216.8790131, 217.3971863
13: -115.2756195, 126.3058243, -115.2987823, 126.4285278, -241.7041321, 241.6046143
14: -172.2479706, 94.1813049, -172.5732727, 94.2068634, -266.4548340, 266.7545776
15: -97.1220856, 83.7928619, -97.1755905, 83.9229889, -181.0450592, 180.9684448
16: -131.8247681, 99.9149170, -131.9995117, 99.9683151, -231.7930908, 231.9144287
17: -178.4286652, 130.1987000, -178.7568054, 130.2310028, -308.6596680, 308.9555054
18: -114.7287903, 96.7735748, -114.9430466, 96.8077393, -211.5365295, 211.7166138
19: -88.4799347, 56.3661690, -88.6365356, 56.3822594, -144.8621826, 145.0027008
20: -78.4810333, 66.1076355, -78.6511230, 66.1246185, -144.6056366, 144.7587585
21: -108.7333984, 69.8836975, -109.0081177, 69.9032364, -178.6366272, 178.8918152
22: -112.6096344, 73.6862640, -112.7072220, 73.7339020, -186.3435364, 186.3934784
23: -89.5985260, 67.4610443, -89.7173538, 67.4835358, -157.0820618, 157.1784058
24: -107.8266525, 62.8360405, -107.9092255, 62.8569374, -170.6835480, 170.7452698
25: -91.0522690, 74.6035919, -91.1364365, 74.6465912, -165.6988525, 165.7400055
26: -127.5811615, 109.8633881, -127.9698563, 109.9034424, -237.4845886, 237.8332520
27: -111.2923889, 74.0296326, -111.3646011, 74.0545349, -185.3469238, 185.3942108
28: -86.4041901, 76.2303009, -86.4842453, 76.2674026, -162.6716003, 162.7145386
29: -120.1470184, 73.7112427, -120.2471466, 73.7277679, -193.8747711, 193.9583740
30: -107.9052200, 83.4232559, -108.0073242, 83.4516678, -191.3568878, 191.4305725
31: -113.8177795, 69.1351318, -114.0112076, 69.1572266, -182.9750061, 183.1463318
32: -101.1966400, 78.7771454, -101.3255310, 78.8073425, -180.0039825, 180.1026764
33: -142.0301208, 112.5165329, -142.0801086, 112.7153244, -254.7454529, 254.5966492
34: -121.6548996, 89.1431198, -121.6991196, 89.2844925, -210.9393921, 210.8422394
35: -118.6523514, 88.5267792, -118.6764679, 88.7456818, -207.3980103, 207.2032471
36: -111.4050751, 89.7733231, -111.4505386, 89.8514328, -201.2565002, 201.2238617
37: -162.8735962, 98.4971390, -162.9781189, 98.5444870, -261.4180603, 261.4752502
38: -145.6592712, 114.4616013, -145.7395020, 114.5396881, -260.1989441, 260.2011108
39: -166.0084534, 110.9477081, -166.0817261, 111.0604248, -277.0688782, 277.0294189
40: -141.4140015, 92.4925842, -141.5170135, 92.5358047, -233.9497986, 234.0095978
41: -103.5271301, 76.5195541, -103.6039276, 76.5637665, -180.0908966, 180.1234741
42: -79.8632812, 74.0213318, -80.0626450, 74.0545502, -153.9178314, 154.0839691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4029484, upper bound: 114.3615214
time: 205.03 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4029484, upper bound: 114.4029481
time: 171.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 379.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4026165, upper bound: 114.2469327
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4026165, upper bound: 114.2923591
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4029484, upper bound: 114.2892853
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4029484, upper bound: 114.3345411
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4026165, upper bound: 114.3195118
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4026165, upper bound: 114.3607483
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4029484, upper bound: 114.3615214
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 379.00
Output dim: 4, lower bound: -114.4029484, upper bound: 114.4029481
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=170.32391357421875
rel_dist={4: [-114.42300729727798, 114.42300729596855]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1753

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1925203, upper bound: 111.1416280
time: 179.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1925216, upper bound: 111.1925215
time: 111.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 291.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 291.64
Output dim: 4, lower bound: -111.1925203, upper bound: 111.1416280
IS_A2, status: Status.UNKNOWN, split count: 1, time: 291.64
Output dim: 4, lower bound: -111.1925216, upper bound: 111.1925215

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -138.5800323, 94.3328018, -138.8271179, 94.4609070, -233.0409393, 233.1599121
1: -84.0452118, 81.9322052, -84.2467194, 82.1147156, -166.1599121, 166.1789246
2: -73.4815826, 72.5476379, -73.6753159, 72.7270050, -146.2085876, 146.2229462
3: -81.7929459, 88.3473587, -82.0026779, 88.5511398, -170.3440552, 170.3500214
4: -84.8026276, 84.7828979, -85.0686188, 85.0118866, -169.8145142, 169.8515015
5: -84.7145691, 89.7864761, -84.8969955, 89.9412689, -174.6558380, 174.6834717
6: -104.0428009, 78.8762894, -104.2934418, 79.0883484, -183.1311340, 183.1697235
7: -101.8312836, 92.5378265, -102.0492554, 92.7048721, -194.5361328, 194.5870819
8: -97.8317947, 102.0174103, -98.1588669, 102.3413849, -200.1731873, 200.1762695
9: -82.5124054, 85.0542679, -82.6203308, 85.1553650, -167.6677704, 167.6745911
10: -116.8588791, 111.5304642, -117.0028687, 111.7315521, -228.5904236, 228.5333252
11: -122.3995972, 93.2638397, -122.6570587, 93.4277115, -215.8273010, 215.9208679
12: -112.7327042, 103.5859146, -112.9070206, 103.8244019, -216.5570984, 216.4929199
13: -115.0631256, 125.8430557, -115.2575531, 126.1243362, -241.1874695, 241.1006165
14: -171.9501953, 93.9311066, -172.1833649, 94.0973511, -266.0475464, 266.1144409
15: -96.9255219, 83.5481491, -97.1040726, 83.7173233, -180.6428375, 180.6521912
16: -131.5195923, 99.7394028, -131.7390137, 99.9035950, -231.4231873, 231.4784088
17: -178.1523132, 130.0074921, -178.3448334, 130.1611176, -308.3134155, 308.3522949
18: -114.2605362, 96.4115753, -114.5377655, 96.7593307, -211.0198517, 210.9493408
19: -88.2060852, 56.2233849, -88.3751678, 56.3596344, -144.5657043, 144.5985565
20: -78.3180695, 65.9524689, -78.4281387, 66.0785522, -144.3966217, 144.3806000
21: -108.4516296, 69.7487183, -108.6333313, 69.8736725, -178.3252869, 178.3820496
22: -112.3793259, 73.5788803, -112.5476837, 73.6734772, -186.0527954, 186.1265564
23: -89.3072815, 67.2667084, -89.4843903, 67.4438095, -156.7510681, 156.7510986
24: -107.5138702, 62.7410088, -107.7157745, 62.8229675, -170.3368378, 170.4567566
25: -90.8351593, 74.4861069, -90.9844894, 74.5825119, -165.4176636, 165.4705811
26: -127.2234497, 109.6227875, -127.4431610, 109.8542938, -237.0777435, 237.0659332
27: -111.0378952, 73.9102402, -111.2167511, 74.0082550, -185.0461426, 185.1269836
28: -86.1485062, 76.0328903, -86.3058701, 76.2046967, -162.3531952, 162.3387451
29: -119.8755264, 73.6089325, -120.0699921, 73.7070389, -193.5825653, 193.6789246
30: -107.6216736, 83.2056885, -107.8097076, 83.4010086, -191.0226746, 191.0153961
31: -113.3913727, 68.9537659, -113.6518707, 69.1244202, -182.5157928, 182.6056366
32: -100.9048462, 78.5883026, -101.0896301, 78.7490082, -179.6538544, 179.6779327
33: -141.8261719, 112.3337708, -141.9699097, 112.4947968, -254.3209381, 254.3036652
34: -121.3448868, 88.9165649, -121.5361938, 89.1266327, -210.4715271, 210.4527588
35: -118.4033508, 88.3695984, -118.5619049, 88.5110092, -206.9143372, 206.9315033
36: -111.1933212, 89.6344376, -111.3301392, 89.7643585, -200.9576721, 200.9645691
37: -162.3024597, 98.1968231, -162.6433411, 98.4948883, -260.7973022, 260.8401489
38: -145.4420776, 114.2952805, -145.5993958, 114.4367065, -259.8787842, 259.8946838
39: -165.7766418, 110.8459473, -165.9496002, 110.9199142, -276.6965637, 276.7955322
40: -140.9156799, 92.1817017, -141.2187347, 92.4821320, -233.3978119, 233.4004364
41: -103.1066437, 76.3266754, -103.3517914, 76.5069580, -179.6135864, 179.6784668
42: -79.6109467, 73.8389130, -79.7614975, 74.0089722, -153.6199188, 153.6004028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1813979, upper bound: 111.1049362
time: 182.10 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1830227, upper bound: 111.1320953
time: 128.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -138.9136963, 94.5485229, -138.9293060, 94.5601807, -233.4738770, 233.4778290
1: -84.2710114, 82.2552795, -84.2759018, 82.2682877, -166.5393066, 166.5311737
2: -73.6965103, 72.8657379, -73.7007141, 72.8791275, -146.5756378, 146.5664520
3: -82.0218506, 88.7029419, -82.0258865, 88.7174301, -170.7392883, 170.7288208
4: -85.0892181, 85.1925964, -85.0945282, 85.2084885, -170.2976990, 170.2871246
5: -84.9159164, 90.0528564, -84.9210815, 90.0650101, -174.9809265, 174.9739380
6: -104.4795837, 79.1233521, -104.4986267, 79.1313019, -183.6108704, 183.6219788
7: -102.0740662, 92.8300858, -102.0806122, 92.8453064, -194.9193573, 194.9106903
8: -98.1804047, 102.5947571, -98.1853714, 102.6163712, -200.7967682, 200.7801208
9: -82.6520386, 85.2095337, -82.6579895, 85.2178497, -167.8698730, 167.8675232
10: -117.0635605, 111.7780838, -117.0713348, 111.7852249, -228.8487701, 228.8494110
11: -122.8321457, 93.4648590, -122.8505249, 93.4737854, -216.3059387, 216.3153839
12: -113.0360870, 103.8601151, -113.0505981, 103.8673172, -216.9033813, 216.9107056
13: -115.3009872, 126.3173752, -115.3093567, 126.3351593, -241.6361389, 241.6267242
14: -172.2617035, 94.2005234, -172.2734070, 94.2120667, -266.4737549, 266.4739075
15: -97.1568909, 83.8023682, -97.1674957, 83.8237381, -180.9806213, 180.9698639
16: -131.8388062, 99.9420700, -131.8664856, 99.9492950, -231.7880707, 231.8085175
17: -178.4376831, 130.2258148, -178.4511414, 130.2348633, -308.6725464, 308.6769409
18: -114.7386169, 96.7890930, -114.7558365, 96.7944794, -211.5330963, 211.5449219
19: -88.4870605, 56.3765488, -88.4983749, 56.3801079, -144.8671722, 144.8749237
20: -78.4876404, 66.1168823, -78.4967422, 66.1226349, -144.6102753, 144.6136169
21: -108.7421265, 69.8975220, -108.7538986, 69.9021606, -178.6442871, 178.6514282
22: -112.6252823, 73.7019882, -112.6386795, 73.7081146, -186.3333740, 186.3406525
23: -89.6050339, 67.4740906, -89.6166000, 67.4794464, -157.0844727, 157.0906830
24: -107.8387299, 62.8428612, -107.8511353, 62.8474617, -170.6861877, 170.6939850
25: -91.0613480, 74.6133728, -91.0754166, 74.6195679, -165.6809082, 165.6887817
26: -127.5917358, 109.8875504, -127.6083221, 109.8957672, -237.4874725, 237.4958496
27: -111.3071518, 74.0376892, -111.3190842, 74.0422974, -185.3494568, 185.3567810
28: -86.4127808, 76.2394562, -86.4237976, 76.2442322, -162.6570129, 162.6632385
29: -120.1568069, 73.7280731, -120.1700592, 73.7324677, -193.8892517, 193.8981018
30: -107.9137802, 83.4463654, -107.9252548, 83.4538116, -191.3675842, 191.3716125
31: -113.8268890, 69.1465073, -113.8440018, 69.1513672, -182.9782562, 182.9905090
32: -101.2048111, 78.7892685, -101.2178192, 78.7954636, -180.0002747, 180.0070801
33: -142.0460815, 112.5240021, -142.0601807, 112.5305405, -254.5766296, 254.5841827
34: -121.6681976, 89.1509094, -121.6796646, 89.1565323, -210.8247070, 210.8305511
35: -118.6689911, 88.5320892, -118.6805038, 88.5366058, -207.2055969, 207.2125854
36: -111.4184189, 89.7789917, -111.4286346, 89.7819519, -201.2003479, 201.2076263
37: -162.8889923, 98.5133057, -162.9116516, 98.5175400, -261.4065247, 261.4249573
38: -145.6770935, 114.4703064, -145.6906433, 114.4758987, -260.1529846, 260.1609192
39: -166.0265808, 110.9531860, -166.0380859, 110.9615479, -276.9881287, 276.9912720
40: -141.4282227, 92.5099487, -141.4483948, 92.5180664, -233.9462891, 233.9583435
41: -103.5363464, 76.5381927, -103.5540619, 76.5453949, -180.0817413, 180.0922394
42: -79.8706284, 74.0463562, -79.8831635, 74.0541687, -153.9248047, 153.9295044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1813979, upper bound: 111.1559253
time: 214.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1830227, upper bound: 111.1830225
time: 257.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 474.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 474.46
Output dim: 4, lower bound: -111.1813979, upper bound: 111.1049362
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 474.46
Output dim: 4, lower bound: -111.1830227, upper bound: 111.1320953
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 474.46
Output dim: 4, lower bound: -111.1813979, upper bound: 111.1559253
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 474.46
Output dim: 4, lower bound: -111.1830227, upper bound: 111.1830225

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -138.4644012, 94.3057404, -138.5565338, 94.3973846, -232.8617554, 232.8622742
1: -83.9620056, 81.9120636, -84.0524139, 82.0674820, -166.0294800, 165.9644775
2: -73.3530426, 72.5270844, -73.3742218, 72.6789246, -146.0319672, 145.9013062
3: -81.6469574, 88.3141098, -81.6608276, 88.4730988, -170.1200409, 169.9749298
4: -84.6640472, 84.7556763, -84.7442474, 84.9481201, -169.6121521, 169.4999084
5: -84.5736694, 89.7555313, -84.5666428, 89.8685760, -174.4422455, 174.3221741
6: -103.9893799, 78.8163300, -104.1681671, 78.9480057, -182.9373779, 182.9844971
7: -101.7077560, 92.5137329, -101.7599716, 92.6485291, -194.3562775, 194.2736969
8: -97.6869278, 101.9881134, -97.8195190, 102.2726364, -199.9595642, 199.8076172
9: -82.4754791, 84.9531860, -82.5340729, 84.9190979, -167.3945770, 167.4872589
10: -116.8005981, 111.2722778, -116.8660660, 111.1265259, -227.9271240, 228.1383362
11: -122.3522568, 93.0712738, -122.5462036, 92.9761047, -215.3283539, 215.6174774
12: -112.6996155, 103.3181076, -112.8295593, 103.1953888, -215.8950043, 216.1476746
13: -115.0129089, 125.7686996, -115.1400375, 125.9488220, -240.9617310, 240.9087219
14: -171.8751831, 93.7563324, -172.0076294, 93.6862640, -265.5614014, 265.7639771
15: -96.8255463, 83.4984665, -96.8721695, 83.6004410, -180.4259949, 180.3706360
16: -131.4497681, 99.6073761, -131.5749207, 99.5948639, -231.0446320, 231.1822968
17: -178.1069641, 129.7884827, -178.2391205, 129.6466675, -307.7536011, 308.0275879
18: -114.2042999, 96.2979126, -114.4058151, 96.4928818, -210.6971741, 210.7037048
19: -88.1642075, 56.1559753, -88.2769318, 56.2021408, -144.3663483, 144.4329071
20: -78.2773209, 65.8907471, -78.3327026, 65.9340668, -144.2113953, 144.2234497
21: -108.4068375, 69.6364441, -108.5283203, 69.6105499, -178.0173950, 178.1647644
22: -112.3296814, 73.4699097, -112.4316864, 73.4181213, -185.7478027, 185.9015808
23: -89.2699814, 67.2074203, -89.3967667, 67.3070526, -156.5770264, 156.6041870
24: -107.4620209, 62.7204208, -107.5947189, 62.7750130, -170.2370300, 170.3151245
25: -90.7994919, 74.4212494, -90.9008484, 74.4307175, -165.2302094, 165.3220978
26: -127.1736984, 109.4257660, -127.3267288, 109.3920441, -236.5657349, 236.7525024
27: -110.9533997, 73.8884354, -111.0190125, 73.9573898, -184.9107971, 184.9074402
28: -86.1048737, 76.0022202, -86.2036285, 76.1334839, -162.2383575, 162.2058411
29: -119.8355560, 73.4770203, -119.9766846, 73.3976288, -193.2331696, 193.4537048
30: -107.5816422, 83.1127777, -107.7156372, 83.1835098, -190.7651520, 190.8284149
31: -113.3314209, 68.8857880, -113.5112991, 68.9653549, -182.2967529, 182.3970795
32: -100.8656082, 78.5077133, -100.9978943, 78.5604477, -179.4260559, 179.5056152
33: -141.6994934, 112.2938004, -141.6721497, 112.4013977, -254.1008911, 253.9659424
34: -121.2406769, 88.8763428, -121.2915802, 89.0324631, -210.2731018, 210.1679230
35: -118.2849350, 88.3396149, -118.2840195, 88.4405594, -206.7254791, 206.6236267
36: -111.1181107, 89.5999985, -111.1540070, 89.6839142, -200.8020325, 200.7539978
37: -162.2298737, 98.1415939, -162.4733582, 98.3661194, -260.5959473, 260.6149597
38: -145.3296509, 114.2587738, -145.3369141, 114.3510208, -259.6806641, 259.5957031
39: -165.6864014, 110.8157272, -165.7384338, 110.8493118, -276.5357056, 276.5541382
40: -140.8277283, 92.1565704, -141.0137177, 92.4236526, -233.2513733, 233.1702881
41: -103.0528717, 76.2785797, -103.2257309, 76.3970566, -179.4499207, 179.5043030
42: -79.5728912, 73.7141647, -79.6722488, 73.7180786, -153.2909546, 153.3864136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1789336, upper bound: 111.0633368
time: 279.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1793945, upper bound: 111.1027932
time: 186.54 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -138.5491638, 94.3257751, -138.8692017, 94.6200790, -233.1692352, 233.1949768
1: -84.0226593, 81.9268799, -84.2590332, 82.1916046, -166.2142639, 166.1859131
2: -73.4541779, 72.5408783, -73.6778717, 72.9392853, -146.3934631, 146.2187500
3: -81.7604675, 88.3359833, -81.9958420, 88.7759018, -170.5363464, 170.3318176
4: -84.7723007, 84.7729645, -85.0780258, 85.2017517, -169.9740601, 169.8509521
5: -84.6898499, 89.7762985, -84.9119568, 90.1815338, -174.8713837, 174.6882629
6: -104.0262451, 78.8304291, -104.3664932, 79.0781555, -183.1043701, 183.1969147
7: -101.8032074, 92.5294037, -102.0931396, 92.7988052, -194.6020050, 194.6225433
8: -97.7998199, 102.0084610, -98.1603241, 102.5554733, -200.3552856, 200.1687927
9: -82.5006790, 85.0325470, -82.7019119, 85.1913452, -167.6920166, 167.7344666
10: -116.8423462, 111.4720001, -117.3700790, 111.7002945, -228.5426331, 228.8420715
11: -122.3829041, 93.2237549, -122.9604492, 93.3997421, -215.7826233, 216.1842041
12: -112.7204056, 103.5347443, -113.4173203, 103.8037949, -216.5242004, 216.9520569
13: -115.0211105, 125.8240967, -115.2427673, 126.2087479, -241.2298584, 241.0668640
14: -171.9277344, 93.8993988, -172.4771881, 94.0856781, -266.0134277, 266.3765869
15: -96.8700485, 83.5326385, -97.1069183, 83.8058624, -180.6759033, 180.6395569
16: -131.4964905, 99.6945419, -131.8570862, 99.9190521, -231.4155426, 231.5516357
17: -178.1374969, 129.9629211, -178.6436768, 130.1528320, -308.2903442, 308.6065674
18: -114.2442780, 96.3880920, -114.7162476, 96.7698975, -211.0141754, 211.1043396
19: -88.1943665, 56.2064819, -88.5074921, 56.3599892, -144.5543518, 144.7139587
20: -78.3072357, 65.9372864, -78.5776901, 66.0776520, -144.3848877, 144.5149841
21: -108.4372711, 69.7260361, -108.8814545, 69.8723755, -178.3096313, 178.6074829
22: -112.3532791, 73.5532379, -112.6094360, 73.6961975, -186.0494690, 186.1626587
23: -89.2965851, 67.2452850, -89.5791931, 67.4452515, -156.7418365, 156.8244629
24: -107.4939270, 62.7297020, -107.7675400, 62.8301468, -170.3240662, 170.4972382
25: -90.8200989, 74.4701233, -91.0383682, 74.6064301, -165.4265289, 165.5084839
26: -127.2059631, 109.5831223, -127.7962570, 109.8578262, -237.0637817, 237.3793640
27: -111.0135803, 73.8968964, -111.2561264, 74.0181351, -185.0317078, 185.1530151
28: -86.1342621, 76.0179138, -86.3606415, 76.2254486, -162.3597107, 162.3785400
29: -119.8594055, 73.5814590, -120.1403503, 73.7001648, -193.5595703, 193.7218018
30: -107.6077118, 83.1673965, -107.8859329, 83.3951797, -191.0028687, 191.0533142
31: -113.3763733, 68.9351807, -113.8102264, 69.1278839, -182.5042267, 182.7453918
32: -100.8914261, 78.5684433, -101.1906052, 78.7577896, -179.6491852, 179.7590332
33: -141.7997742, 112.3217773, -141.9826813, 112.6762695, -254.4760437, 254.3044434
34: -121.3229599, 88.9039307, -121.5492706, 89.2517700, -210.5747070, 210.4532013
35: -118.3758621, 88.3609314, -118.5519714, 88.7178497, -207.0937195, 206.9128876
36: -111.1712570, 89.6252365, -111.3468704, 89.8323517, -201.0035706, 200.9721069
37: -162.2771912, 98.1705017, -162.6983643, 98.5197220, -260.7969055, 260.8688660
38: -145.4127808, 114.2809143, -145.6409912, 114.4976501, -259.9104309, 259.9218445
39: -165.7467346, 110.8369141, -165.9874420, 111.0145264, -276.7612610, 276.8243103
40: -140.8923035, 92.1529007, -141.2770538, 92.4958344, -233.3881073, 233.4299622
41: -103.0915375, 76.2959290, -103.3927155, 76.5216904, -179.6132202, 179.6886444
42: -79.5989456, 73.7976074, -79.9345016, 74.0054016, -153.6043396, 153.7321167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1789336, upper bound: 111.0905693
time: 310.57 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1793945, upper bound: 111.1299519
time: 216.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -138.7980194, 94.5214920, -138.6587524, 94.4967194, -233.2947388, 233.1802368
1: -84.1878052, 82.2351837, -84.0815964, 82.2210541, -166.4088593, 166.3167725
2: -73.5679321, 72.8452377, -73.3996124, 72.8310547, -146.3989868, 146.2448425
3: -81.8758545, 88.6696930, -81.6840363, 88.6393890, -170.5152435, 170.3537292
4: -84.9506073, 85.1653824, -84.7701340, 85.1447372, -170.0953369, 169.9355164
5: -84.7750320, 90.0219574, -84.5907440, 89.9922943, -174.7673340, 174.6127014
6: -104.4261856, 79.0633850, -104.3733826, 78.9909363, -183.4171143, 183.4367676
7: -101.9505081, 92.8060303, -101.7913513, 92.7889404, -194.7394409, 194.5973816
8: -98.0355377, 102.5654755, -97.8460693, 102.5476074, -200.5831451, 200.4115295
9: -82.6150742, 85.1084747, -82.5716629, 84.9816132, -167.5966797, 167.6801300
10: -117.0051575, 111.5198441, -116.9343033, 111.1801910, -228.1853333, 228.4541473
11: -122.7848587, 93.2722473, -122.7396698, 93.0222168, -215.8070679, 216.0119019
12: -113.0030136, 103.5923309, -112.9731216, 103.2383118, -216.2413330, 216.5654602
13: -115.2506866, 126.2430573, -115.1917648, 126.1596451, -241.4103088, 241.4348145
14: -172.1865997, 94.0257568, -172.0976105, 93.8011169, -265.9877014, 266.1233521
15: -97.0568695, 83.7526550, -96.9356537, 83.7067871, -180.7636414, 180.6882935
16: -131.7690735, 99.8099289, -131.7025146, 99.6405487, -231.4096069, 231.5124207
17: -178.3923340, 130.0068054, -178.3453217, 129.7203522, -308.1126709, 308.3521118
18: -114.6824112, 96.6753845, -114.6239014, 96.5280151, -211.2104187, 211.2992859
19: -88.4451981, 56.3091507, -88.4001083, 56.2226295, -144.6678314, 144.7092590
20: -78.4469147, 66.0551529, -78.4013672, 65.9781570, -144.4250793, 144.4565125
21: -108.6973648, 69.7852249, -108.6489029, 69.6390533, -178.3364258, 178.4341125
22: -112.5756531, 73.5930023, -112.5227509, 73.4527664, -186.0284119, 186.1157532
23: -89.5677338, 67.4147644, -89.5289764, 67.3426819, -156.9104156, 156.9437256
24: -107.7868729, 62.8222580, -107.7301483, 62.7994499, -170.5863190, 170.5523987
25: -91.0256805, 74.5484924, -90.9917679, 74.4677734, -165.4934540, 165.5402527
26: -127.5420227, 109.6905136, -127.4918900, 109.4334869, -236.9755096, 237.1824036
27: -111.2227020, 74.0158844, -111.1214294, 73.9914093, -185.2141113, 185.1373138
28: -86.3691864, 76.2087784, -86.3215637, 76.1730347, -162.5422058, 162.5303192
29: -120.1168442, 73.5961304, -120.0767059, 73.4230652, -193.5399170, 193.6728210
30: -107.8737259, 83.3534241, -107.8312149, 83.2363205, -191.1100464, 191.1846161
31: -113.7669907, 69.0784683, -113.7035065, 68.9922714, -182.7592621, 182.7819519
32: -101.1656036, 78.7086716, -101.1260834, 78.6068802, -179.7724762, 179.8347473
33: -141.9194336, 112.4839478, -141.7624817, 112.4370346, -254.3564758, 254.2464142
34: -121.5640030, 89.1105957, -121.4351120, 89.0622864, -210.6262817, 210.5457153
35: -118.5505829, 88.5020447, -118.4025803, 88.4660721, -207.0166626, 206.9046326
36: -111.3431931, 89.7445221, -111.2525864, 89.7014999, -201.0446930, 200.9971008
37: -162.8164978, 98.4580231, -162.7417297, 98.3887329, -261.2052307, 261.1997681
38: -145.5647736, 114.4337006, -145.4282532, 114.3901825, -259.9549561, 259.8619385
39: -165.9363098, 110.9230270, -165.8269653, 110.8909607, -276.8272705, 276.7499695
40: -141.3403931, 92.4848328, -141.2436066, 92.4595642, -233.7999420, 233.7283936
41: -103.4825745, 76.4900818, -103.4280090, 76.4354553, -179.9180298, 179.9180908
42: -79.8325577, 73.9215622, -79.7939606, 73.7632751, -153.5958252, 153.7155151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1790165, upper bound: 111.1154765
time: 179.93 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1793964, upper bound: 111.1539044
time: 276.84 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -138.8827820, 94.5415115, -138.9713440, 94.7194977, -233.6022644, 233.5128479
1: -84.2484589, 82.2499695, -84.2882233, 82.3451843, -166.5936432, 166.5381775
2: -73.6690826, 72.8590164, -73.7032776, 73.0914307, -146.7605133, 146.5622864
3: -81.9893646, 88.6915741, -82.0190125, 88.9422531, -170.9316101, 170.7105865
4: -85.0588837, 85.1826401, -85.1038971, 85.3983841, -170.4572754, 170.2865295
5: -84.8912201, 90.0426941, -84.9360199, 90.3052826, -175.1964722, 174.9786987
6: -104.4630432, 79.0775146, -104.5717010, 79.1211243, -183.5841675, 183.6492004
7: -102.0459671, 92.8216553, -102.1244659, 92.9392471, -194.9851990, 194.9461212
8: -98.1484222, 102.5858002, -98.1868591, 102.8305206, -200.9788818, 200.7726440
9: -82.6402893, 85.1878204, -82.7395782, 85.2538223, -167.8941040, 167.9273987
10: -117.0470352, 111.7196045, -117.4385223, 111.7539368, -228.8009644, 229.1581116
11: -122.8154755, 93.4247971, -123.1539841, 93.4457703, -216.2612305, 216.5787811
12: -113.0237885, 103.8089066, -113.5609589, 103.8466873, -216.8704681, 217.3698730
13: -115.2589569, 126.2984314, -115.2945709, 126.4195251, -241.6784821, 241.5930023
14: -172.2392120, 94.1687698, -172.5672302, 94.2004623, -266.4396667, 266.7359924
15: -97.1014023, 83.7868576, -97.1703033, 83.9122162, -181.0136108, 180.9571533
16: -131.8157349, 99.8971939, -131.9846802, 99.9646835, -231.7804260, 231.8818665
17: -178.4228821, 130.1811676, -178.7499390, 130.2264709, -308.6493530, 308.9310913
18: -114.7223587, 96.7655792, -114.9343414, 96.8049927, -211.5273438, 211.6998901
19: -88.4753265, 56.3596344, -88.6306992, 56.3804626, -144.8557892, 144.9903259
20: -78.4768219, 66.1016998, -78.6464615, 66.1217575, -144.5985718, 144.7481537
21: -108.7277832, 69.8748550, -109.0020752, 69.9009094, -178.6286926, 178.8769226
22: -112.5992279, 73.6763306, -112.7004547, 73.7307968, -186.3300171, 186.3767853
23: -89.5943298, 67.4526443, -89.7113953, 67.4808502, -157.0751648, 157.1640320
24: -107.8187408, 62.8315468, -107.9029388, 62.8546066, -170.6733398, 170.7344818
25: -91.0462952, 74.5974121, -91.1292419, 74.6434784, -165.6897736, 165.7266541
26: -127.5742416, 109.8478928, -127.9614716, 109.8992310, -237.4734802, 237.8093414
27: -111.2828598, 74.0243530, -111.3585358, 74.0521851, -185.3350372, 185.3828888
28: -86.3985519, 76.2245026, -86.4785614, 76.2649841, -162.6635132, 162.7030640
29: -120.1406708, 73.7005768, -120.2404175, 73.7255707, -193.8662415, 193.9409790
30: -107.8997955, 83.4080658, -108.0014954, 83.4479294, -191.3477173, 191.4095612
31: -113.8118744, 69.1278992, -114.0024490, 69.1547928, -182.9666748, 183.1303406
32: -101.1913910, 78.7694168, -101.3189163, 78.8042450, -179.9956360, 180.0883331
33: -142.0196991, 112.5119629, -142.0729370, 112.7120361, -254.7317352, 254.5848999
34: -121.6462784, 89.1382217, -121.6927109, 89.2816467, -210.9279175, 210.8309326
35: -118.6415176, 88.5234222, -118.6705246, 88.7434387, -207.3849487, 207.1939240
36: -111.3963165, 89.7697906, -111.4453888, 89.8499451, -201.2462463, 201.2151794
37: -162.8637085, 98.4869156, -162.9666443, 98.5423203, -261.4060059, 261.4535522
38: -145.6478119, 114.4559402, -145.7321777, 114.5368576, -260.1846619, 260.1881104
39: -165.9966278, 110.9442291, -166.0759277, 111.0561905, -277.0527954, 277.0201416
40: -141.4048157, 92.4811707, -141.5068054, 92.5317459, -233.9365540, 233.9879303
41: -103.5212097, 76.5074310, -103.5949478, 76.5601196, -180.0813293, 180.1023407
42: -79.8586044, 74.0050659, -80.0562210, 74.0505981, -153.9091797, 154.0612793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1789336, upper bound: 111.1427769
time: 143.66 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1810039, upper bound: 111.1810036
time: 179.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 325.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1789336, upper bound: 111.0633368
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1793945, upper bound: 111.1027932
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1789336, upper bound: 111.0905693
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1793945, upper bound: 111.1299519
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1790165, upper bound: 111.1154765
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1793964, upper bound: 111.1539044
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1789336, upper bound: 111.1427769
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 325.79
Output dim: 4, lower bound: -111.1810039, upper bound: 111.1810036

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -138.0573425, 94.0012817, -138.4084778, 94.2496109, -232.3069458, 232.4097595
1: -83.7518692, 81.4497757, -84.0110168, 81.8421021, -165.5939636, 165.4607849
2: -73.1540985, 72.1146851, -73.3392029, 72.4796982, -145.6337891, 145.4538879
3: -81.4241409, 87.7520447, -81.6285324, 88.2023392, -169.6264801, 169.3805847
4: -84.4322510, 84.2428131, -84.7083893, 84.7004242, -169.1326752, 168.9512024
5: -84.3709869, 89.3888855, -84.5300446, 89.6926270, -174.0636139, 173.9189148
6: -103.4193573, 78.5678177, -103.8965378, 78.8926697, -182.3120270, 182.4643555
7: -101.4930573, 92.0595016, -101.7227402, 92.4262238, -193.9192810, 193.7822266
8: -97.4335632, 101.2207565, -97.7859497, 101.9014435, -199.3350067, 199.0066833
9: -82.3233337, 84.7741394, -82.4799805, 84.8428040, -167.1661377, 167.2541199
10: -116.4984207, 111.0559616, -116.7452621, 111.0700150, -227.5684204, 227.8012085
11: -121.9470673, 92.9066467, -122.3698730, 92.9104767, -214.8575439, 215.2765045
12: -112.2412567, 103.0831833, -112.6081924, 103.1452103, -215.3864746, 215.6913757
13: -114.7619095, 125.2348557, -115.0726852, 125.6961975, -240.4580994, 240.3075409
14: -171.5418396, 93.4763489, -171.8897705, 93.5621185, -265.1039429, 265.3661194
15: -96.5367889, 83.0797501, -96.7958527, 83.4035263, -179.9403076, 179.8755951
16: -130.9609375, 99.4134140, -131.3540802, 99.5450287, -230.5059357, 230.7674866
17: -177.7593689, 129.4193420, -178.1112061, 129.4801178, -307.2395020, 307.5305481
18: -113.6398163, 95.9989929, -114.1429901, 96.4471588, -210.0869598, 210.1419830
19: -87.8122559, 56.0136490, -88.1173248, 56.1720200, -143.9842834, 144.1309814
20: -78.1003723, 65.7251663, -78.2543488, 65.8786163, -143.9789734, 143.9794922
21: -108.1079102, 69.5041046, -108.3997345, 69.5675507, -177.6754456, 177.9038086
22: -112.0508728, 73.3212738, -112.3280563, 73.3558197, -185.4066925, 185.6493225
23: -88.9869080, 67.0679474, -89.2680588, 67.2623749, -156.2492676, 156.3359985
24: -107.0859833, 62.6270866, -107.4275436, 62.7415237, -169.8275146, 170.0546265
25: -90.5200882, 74.2944641, -90.7772369, 74.3850479, -164.9051361, 165.0717010
26: -126.9023666, 109.2087250, -127.2026367, 109.3236542, -236.2259827, 236.4113617
27: -110.7529678, 73.7416382, -110.9394302, 73.9008179, -184.6537781, 184.6810608
28: -85.8728867, 75.8430786, -86.0962982, 76.0773468, -161.9502258, 161.9393768
29: -119.5364609, 73.3313293, -119.8618698, 73.3320618, -192.8685303, 193.1931763
30: -107.3516312, 82.9118958, -107.6166077, 83.1102829, -190.4618988, 190.5285034
31: -112.6614151, 68.6751709, -113.1934052, 68.9255981, -181.5870056, 181.8685760
32: -100.4966888, 78.3191223, -100.8284836, 78.5090790, -179.0057678, 179.1475830
33: -141.3448944, 112.1087418, -141.5073395, 112.3627396, -253.7076416, 253.6160889
34: -120.8596191, 88.6745453, -121.1104202, 88.9861145, -209.8457336, 209.7849579
35: -117.9641190, 88.1917877, -118.1322556, 88.4040833, -206.3681946, 206.3240051
36: -110.9189606, 89.4956970, -111.0606308, 89.6537399, -200.5726624, 200.5563354
37: -161.5191650, 97.9051666, -162.1390381, 98.3390656, -259.8582153, 260.0441895
38: -144.9934540, 114.0474854, -145.1819763, 114.2948380, -259.2882996, 259.2294617
39: -165.3119812, 110.6821136, -165.5715637, 110.8111420, -276.1231079, 276.2536316
40: -140.1284180, 91.8585510, -140.6857300, 92.3791656, -232.5075836, 232.5442810
41: -102.6031342, 76.1075592, -103.0110016, 76.3537674, -178.9569092, 179.1185608
42: -79.2534332, 73.5230179, -79.5200348, 73.6618271, -152.9152527, 153.0430603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=573, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0364570
time: 211.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0616573
time: 203.20 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -138.4476776, 94.2915039, -138.5498352, 94.3917542, -232.8394318, 232.8413391
1: -83.9550476, 81.8931198, -84.0496902, 82.0599365, -166.0149841, 165.9428101
2: -73.3473740, 72.5105972, -73.3719940, 72.6723938, -146.0197754, 145.8825989
3: -81.6425018, 88.2911606, -81.6590576, 88.4639740, -170.1064758, 169.9502258
4: -84.6582489, 84.7360458, -84.7419739, 84.9402695, -169.5985107, 169.4780273
5: -84.5690231, 89.7392120, -84.5647888, 89.8620911, -174.4311218, 174.3039856
6: -103.9680939, 78.8062439, -104.1596680, 78.9439850, -182.9120789, 182.9659119
7: -101.7013779, 92.4928284, -101.7574692, 92.6401825, -194.3415527, 194.2503052
8: -97.6817169, 101.9568863, -97.8174591, 102.2602005, -199.9419098, 199.7743530
9: -82.4675446, 84.9405899, -82.5309296, 84.9140930, -167.3816376, 167.4715118
10: -116.7814178, 111.2650681, -116.8582535, 111.1236420, -227.9050598, 228.1233063
11: -122.3251724, 93.0616837, -122.5354843, 92.9722443, -215.2974243, 215.5971680
12: -112.6779327, 103.3118439, -112.8209763, 103.1928940, -215.8708191, 216.1328125
13: -115.0006485, 125.7577133, -115.1351852, 125.9444046, -240.9450531, 240.8928986
14: -171.8614807, 93.7437057, -172.0021057, 93.6812286, -265.5426941, 265.7458191
15: -96.8110657, 83.4749146, -96.8664703, 83.5911560, -180.4022217, 180.3413849
16: -131.4348755, 99.5980759, -131.5689697, 99.5911789, -231.0260620, 231.1670532
17: -178.0901794, 129.7564697, -178.2324524, 129.6327820, -307.7229614, 307.9888916
18: -114.1822433, 96.2906952, -114.3970413, 96.4900208, -210.6722412, 210.6877441
19: -88.1516266, 56.1517944, -88.2715759, 56.2004929, -144.3521118, 144.4233704
20: -78.2667236, 65.8833160, -78.3284836, 65.9311447, -144.1978760, 144.2117920
21: -108.3914490, 69.6300201, -108.5220795, 69.6079712, -177.9994202, 178.1520844
22: -112.3132324, 73.4413300, -112.4251633, 73.4066620, -185.7198944, 185.8664856
23: -89.2541428, 67.2017212, -89.3901825, 67.3047943, -156.5589294, 156.5918884
24: -107.4440918, 62.7154617, -107.5875626, 62.7730103, -170.2171021, 170.3030090
25: -90.7792053, 74.4141083, -90.8927994, 74.4278870, -165.2070923, 165.3069000
26: -127.1569672, 109.4165344, -127.3200378, 109.3883133, -236.5452881, 236.7365417
27: -110.9441071, 73.8814240, -111.0152817, 73.9545670, -184.8986511, 184.8966980
28: -86.0921860, 75.9957657, -86.1985550, 76.1309509, -162.2231445, 162.1943207
29: -119.8179626, 73.4485321, -119.9696960, 73.3858185, -193.2037811, 193.4182281
30: -107.5706177, 83.1035995, -107.7112045, 83.1798782, -190.7504730, 190.8147888
31: -113.3107529, 68.8778000, -113.5030289, 68.9621887, -182.2729340, 182.3808289
32: -100.8507690, 78.5005493, -100.9919586, 78.5576172, -179.4083862, 179.4924927
33: -141.6837311, 112.2865295, -141.6658325, 112.3984756, -254.0822144, 253.9523621
34: -121.2245178, 88.8670731, -121.2851562, 89.0288391, -210.2533569, 210.1522064
35: -118.2713013, 88.3341370, -118.2785645, 88.4383850, -206.7096863, 206.6127014
36: -111.1064529, 89.5950928, -111.1493988, 89.6819763, -200.7884216, 200.7444916
37: -162.1999359, 98.1378098, -162.4613953, 98.3646240, -260.5645752, 260.5992126
38: -145.3136749, 114.2504959, -145.3300476, 114.3477478, -259.6614380, 259.5805359
39: -165.6649780, 110.8098297, -165.7300110, 110.8469162, -276.5119019, 276.5398560
40: -140.8006897, 92.1480865, -141.0029907, 92.4203491, -233.2210236, 233.1510773
41: -103.0341949, 76.2720947, -103.2182922, 76.3944473, -179.4286194, 179.4903870
42: -79.5612640, 73.7053833, -79.6673813, 73.7145844, -153.2758484, 153.3727722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=573, inp2_unstable=574, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0364570
time: 147.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0616573
time: 5744.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5894.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5894.71
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0364570
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5894.71
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0616573
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5894.71
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0364570
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5894.71
Output dim: 4, lower bound: -111.1771925, upper bound: 111.0616573
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1789336, upper bound: 111.0905693
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1793945, upper bound: 111.1299519
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1790165, upper bound: 111.1154765
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1793964, upper bound: 111.1539044
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1789336, upper bound: 111.1427769
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5894.71
Output dim: 4, lower bound: -111.1810039, upper bound: 111.1810036
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=170.32391357421875
rel_dist={4: [-111.19460966977735, 111.19460966338143]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1051
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1048
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1047
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1087
type: A, layer: 1, pos: 1246
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1151
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1247
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1050
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1049
type: A, layer: 1, pos: 1070
type: A, layer: 1, pos: 1215
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1276
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1036
type: A, layer: 1, pos: 1134
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1243
type: A, layer: 1, pos: 1071
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1135
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1103
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1261
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1271
type: A, layer: 1, pos: 1086
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1054
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1270
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1231
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1263
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1262
type: A, layer: 1, pos: 1167
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1055
type: A, layer: 1, pos: 1229
type: A, layer: 1, pos: 1255
type: A, layer: 1, pos: 1275
type: A, layer: 1, pos: 1242
type: A, layer: 1, pos: 1063
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1277
type: A, layer: 1, pos: 1067
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1119
type: A, layer: 1, pos: 1198
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1230
type: A, layer: 1, pos: 1068
type: A, layer: 1, pos: 1199
type: A, layer: 1, pos: 1052
type: A, layer: 1, pos: 1260
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1259
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1069
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1085
type: A, layer: 1, pos: 1118

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1753

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8710830, upper bound: 109.8301588
time: 130.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8712227, upper bound: 109.8712226
time: 164.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 294.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 294.61
Output dim: 4, lower bound: -109.8710830, upper bound: 109.8301588
IS_A2, status: Status.UNKNOWN, split count: 1, time: 294.61
Output dim: 4, lower bound: -109.8712227, upper bound: 109.8712226

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -138.5800323, 94.3328018, -138.8056488, 94.4410553, -233.0210876, 233.1384430
1: -84.0452118, 81.9322052, -84.2405090, 82.0842514, -166.1294556, 166.1727142
2: -73.4815826, 72.5476379, -73.6699219, 72.6969147, -146.1784973, 146.2175598
3: -81.7929459, 88.3473587, -81.9978180, 88.5176544, -170.3106079, 170.3451843
4: -84.8026276, 84.7828979, -85.0629349, 84.9731293, -169.7757568, 169.8458252
5: -84.7145691, 89.7864761, -84.8916321, 89.9163666, -174.6309357, 174.6781006
6: -104.0428009, 78.8762894, -104.2522354, 79.0791855, -183.1219788, 183.1285095
7: -101.8312836, 92.5378265, -102.0423737, 92.6782684, -194.5095520, 194.5802002
8: -97.8317947, 102.0174103, -98.1531296, 102.2863922, -200.1181946, 200.1705322
9: -82.5124054, 85.0542679, -82.6123352, 85.1427612, -167.6551666, 167.6665955
10: -116.8588791, 111.5304642, -116.9888763, 111.7206573, -228.5795288, 228.5193481
11: -122.3995972, 93.2638397, -122.6187744, 93.4180756, -215.8176727, 215.8826141
12: -112.7327042, 103.5859146, -112.8784103, 103.8151245, -216.5478210, 216.4643250
13: -115.0631256, 125.8430557, -115.2469788, 126.0822449, -241.1453705, 241.0900269
14: -171.9501953, 93.9311066, -172.1652374, 94.0744781, -266.0246582, 266.0963440
15: -96.9255219, 83.5481491, -97.0910110, 83.6943512, -180.6198425, 180.6391449
16: -131.5195923, 99.7394028, -131.7099152, 99.8944397, -231.4140320, 231.4493103
17: -178.1523132, 130.0074921, -178.3236389, 130.1460876, -308.2984009, 308.3311157
18: -114.2605362, 96.4115753, -114.4939423, 96.7522812, -211.0128174, 210.9055176
19: -88.2060852, 56.2233849, -88.3503723, 56.3553619, -144.5614471, 144.5737610
20: -78.3180695, 65.9524689, -78.4141846, 66.0694962, -144.3875427, 144.3666382
21: -108.4516296, 69.7487183, -108.6091232, 69.8677521, -178.3193665, 178.3578491
22: -112.3793259, 73.5788803, -112.5294342, 73.6661987, -186.0455322, 186.1083069
23: -89.3072815, 67.2667084, -89.4578705, 67.4363937, -156.7436829, 156.7245789
24: -107.5138702, 62.7410088, -107.6893387, 62.8176842, -170.3315582, 170.4303284
25: -90.8351593, 74.4861069, -90.9660187, 74.5747681, -165.4099274, 165.4521179
26: -127.2234497, 109.6227875, -127.4098053, 109.8451843, -237.0686340, 237.0325775
27: -111.0378952, 73.9102402, -111.1959000, 74.0014038, -185.0393066, 185.1061401
28: -86.1485062, 76.0328903, -86.2820511, 76.1965485, -162.3450623, 162.3149261
29: -119.8755264, 73.6089325, -120.0500107, 73.7016754, -193.5772095, 193.6589203
30: -107.6216736, 83.2056885, -107.7867966, 83.3899155, -191.0115814, 190.9924927
31: -113.3913727, 68.9537659, -113.6132889, 69.1188202, -182.5101929, 182.5670471
32: -100.9048462, 78.5883026, -101.0638657, 78.7393646, -179.6441956, 179.6521606
33: -141.8261719, 112.3337708, -141.9515533, 112.4870834, -254.3132629, 254.2853241
34: -121.3448868, 88.9165649, -121.5082779, 89.1200867, -210.4649658, 210.4248352
35: -118.4033508, 88.3695984, -118.5391006, 88.5054779, -206.9088287, 206.9086914
36: -111.1933212, 89.6344376, -111.3108215, 89.7605896, -200.9539185, 200.9452515
37: -162.3024597, 98.1968231, -162.5899963, 98.4901123, -260.7925415, 260.7868042
38: -145.4420776, 114.2952805, -145.5806122, 114.4284973, -259.8705444, 259.8758545
39: -165.7766418, 110.8459473, -165.9316864, 110.9106979, -276.6873474, 276.7776489
40: -140.9156799, 92.1817017, -141.1726685, 92.4745102, -233.3901978, 233.3543701
41: -103.1066437, 76.3266754, -103.3113861, 76.4987946, -179.6054382, 179.6380615
42: -79.6109467, 73.8389130, -79.7369995, 73.9993134, -153.6102600, 153.5759125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=736, inp2_unstable=736, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8619282, upper bound: 109.8012469
time: 135.43 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8636511, upper bound: 109.8227446
time: 438.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -138.9136963, 94.5485229, -138.9260559, 94.5577240, -233.4714203, 233.4745789
1: -84.2710114, 82.2552795, -84.2748871, 82.2655945, -166.5366058, 166.5301666
2: -73.6965103, 72.8657379, -73.6998138, 72.8763428, -146.5728455, 146.5655518
3: -82.0218506, 88.7029419, -82.0250397, 88.7144318, -170.7362823, 170.7279816
4: -85.0892181, 85.1925964, -85.0934296, 85.2051697, -170.2943878, 170.2860260
5: -84.9159164, 90.0528564, -84.9199829, 90.0625000, -174.9784241, 174.9728394
6: -104.4795837, 79.1233521, -104.4946442, 79.1296844, -183.6092682, 183.6179962
7: -102.0740662, 92.8300858, -102.0792160, 92.8421326, -194.9161987, 194.9093018
8: -98.1804047, 102.5947571, -98.1843567, 102.6118851, -200.7922668, 200.7791138
9: -82.6520386, 85.2095337, -82.6567459, 85.2161102, -167.8681488, 167.8662720
10: -117.0635605, 111.7780838, -117.0696945, 111.7837219, -228.8472748, 228.8477783
11: -122.8321457, 93.4648590, -122.8466949, 93.4719391, -216.3040466, 216.3115540
12: -113.0360870, 103.8601151, -113.0476074, 103.8657990, -216.9018402, 216.9077148
13: -115.3009872, 126.3173752, -115.3076324, 126.3314819, -241.6324768, 241.6250000
14: -172.2617035, 94.2005234, -172.2709198, 94.2094345, -266.4711304, 266.4714355
15: -97.1568909, 83.8023682, -97.1653137, 83.8193207, -180.9762115, 180.9676819
16: -131.8388062, 99.9420700, -131.8604279, 99.9477997, -231.7865906, 231.8024902
17: -178.4376831, 130.2258148, -178.4483337, 130.2329712, -308.6706543, 308.6741333
18: -114.7386169, 96.7890930, -114.7522278, 96.7933731, -211.5319519, 211.5413208
19: -88.4870605, 56.3765488, -88.4959793, 56.3793602, -144.8664246, 144.8725281
20: -78.4876404, 66.1168823, -78.4948349, 66.1214294, -144.6090698, 144.6117096
21: -108.7421265, 69.8975220, -108.7514267, 69.9012070, -178.6433258, 178.6489563
22: -112.6252823, 73.7019882, -112.6358948, 73.7068253, -186.3320923, 186.3378906
23: -89.6050339, 67.4740906, -89.6141663, 67.4783325, -157.0833740, 157.0882568
24: -107.8387299, 62.8428612, -107.8485565, 62.8464813, -170.6852112, 170.6914062
25: -91.0613480, 74.6133728, -91.0724945, 74.6182709, -165.6796265, 165.6858673
26: -127.5917358, 109.8875504, -127.6048660, 109.8940582, -237.4857788, 237.4923859
27: -111.3071518, 74.0376892, -111.3166046, 74.0413361, -185.3484802, 185.3542786
28: -86.4127808, 76.2394562, -86.4214783, 76.2432480, -162.6560364, 162.6609192
29: -120.1568069, 73.7280731, -120.1672821, 73.7315521, -193.8883362, 193.8953552
30: -107.9137802, 83.4463654, -107.9228516, 83.4522705, -191.3660583, 191.3692169
31: -113.8268890, 69.1465073, -113.8404083, 69.1503677, -182.9772644, 182.9869080
32: -101.2048111, 78.7892685, -101.2151108, 78.7941666, -179.9989777, 180.0043640
33: -142.0460815, 112.5240021, -142.0572815, 112.5291748, -254.5752563, 254.5812836
34: -121.6681976, 89.1509094, -121.6770020, 89.1553421, -210.8235474, 210.8278961
35: -118.6689911, 88.5320892, -118.6780396, 88.5356750, -207.2046661, 207.2101135
36: -111.4184189, 89.7789917, -111.4265137, 89.7813721, -201.1997375, 201.2055054
37: -162.8889923, 98.5133057, -162.9069214, 98.5166245, -261.4056091, 261.4202271
38: -145.6770935, 114.4703064, -145.6876831, 114.4747238, -260.1518250, 260.1579285
39: -166.0265808, 110.9531860, -166.0356750, 110.9598083, -276.9863586, 276.9888611
40: -141.4282227, 92.5099487, -141.4442444, 92.5163879, -233.9446106, 233.9541931
41: -103.5363464, 76.5381927, -103.5503540, 76.5438995, -180.0802307, 180.0885315
42: -79.8706284, 74.0463562, -79.8805389, 74.0525436, -153.9231720, 153.9268799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=574, inp2_unstable=575, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1051
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1048
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1047
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1087
type: B, layer: 1, pos: 1246
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1151
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1247
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1050
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1049
type: B, layer: 1, pos: 1070
type: B, layer: 1, pos: 1215
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1276
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1036
type: B, layer: 1, pos: 1134
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1243
type: B, layer: 1, pos: 1071
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1135
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1103
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1261
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1271
type: B, layer: 1, pos: 1086
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1054
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1270
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1231
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1263
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1262
type: B, layer: 1, pos: 1167
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1055
type: B, layer: 1, pos: 1229
type: B, layer: 1, pos: 1255
type: B, layer: 1, pos: 1275
type: B, layer: 1, pos: 1242
type: B, layer: 1, pos: 1063
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1277
type: B, layer: 1, pos: 1067
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1119
type: B, layer: 1, pos: 1198
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1230
type: B, layer: 1, pos: 1068
type: B, layer: 1, pos: 1199
type: B, layer: 1, pos: 1052
type: B, layer: 1, pos: 1260
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1259
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1069
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1085
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8620700, upper bound: 109.8422450
time: 402.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8620700, upper bound: 109.8422450
time: 581.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 986.27 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 986.27
Output dim: 4, lower bound: -109.8619282, upper bound: 109.8012469
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 986.27
Output dim: 4, lower bound: -109.8636511, upper bound: 109.8227446
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 986.27
Output dim: 4, lower bound: -109.8620700, upper bound: 109.8422450
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 986.27
Output dim: 4, lower bound: -109.8620700, upper bound: 109.8422450
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=170.32391357421875
rel_dist={4: [-109.87288027651347, 109.87288028306807]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 15762.73 seconds
