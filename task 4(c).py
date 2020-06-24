import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed to ensure reproducible runs
RSEED = 50

df = pd.read_csv('data.csv',skiprows=3)
#print(df)
X = df[list(df.columns[0:2]) + list(df.columns[3:])]
#print(X)
Y = df.iloc[:,[2]]
#print(Y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=RSEED)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a list of feature names
feat_labels = ['SL000318', 'SL002662', 'SL003168', 'SL000403', 'SL000591', 'SL000053', 'SL004332', 'SL000592', 'SL000584', 'SL003322', 'SL000319', 'SL000276', 'SL003951', 'SL003060', 'SL000324', 'SL000345', 'SL004016', 'SL004333', 'SL004342', 'SL004066', 'SL004182', 'SL001716', 'SL000133', 'SL000573', 'SL004010', 'SL003043', 'SL004365', 'SL004643', 'SL000055', 'SL004144', 'SL004142', 'SL004143', 'SL003193', 'SL000563', 'SL003280', 'SL002539', 'SL000300', 'SL003328', 'SL000466', 'SL000045', 'SL000498', 'SL000038', 'SL000527', 'SL001796', 'SL000546', 'SL006114', 'SL000002', 'SL004126', 'SL003735', 'SL001996', 'SL003738', 'SL004180', 'SL004692', 'SL001777', 'SL004136', 'SL004759', 'SL004140', 'SL004141', 'SL000009', 'SL002519', 'SL003803', 'SL001890', 'SL003872', 'SL004751', 'SL004697', 'SL004698', 'SL017612', 'SL004588', 'SL004147', 'SL004148', 'SL004149', 'SL003307', 'SL004646', 'SL000509', 'SL004579', 'SL004153', 'SL004637', 'SL004760', 'SL004183', 'SL000551', 'SL000557', 'SL003648', 'SL003178', 'SL002506', 'SL004230', 'SL001992', 'SL004639', 'SL004672', 'SL004081', 'SL000337', 'SL000383', 'SL005155', 'SL002644', 'SL000441', 'SL000450', 'SL000456', 'SL004676', 'SL005172', 'SL001947', 'SL002528', 'SL000530', 'SL001721', 'SL004685', 'SL003041', 'SL000049', 'SL004364', 'SL003197', 'SL000590', 'SL004326', 'SL005153', 'SL003334', 'SL005159', 'SL004591', 'SL005178', 'SL000497','SL005199', 'SL000528', 'SL005235', 'SL005206', 'SL004683', 'SL005236', 'SL005217', 'SL005218', 'SL005220', 'SL005226', 'SL005229', 'SL001938', 'SL000019', 'SL004589', 'SL004329', 'SL000309', 'SL000312', 'SL003220', 'SL004752', 'SL004336', 'SL004337', 'SL004338', 'SL003849', 'SL004345', 'SL000440', 'SL000445', 'SL003176', 'SL000462', 'SL001717', 'SL000474', 'SL004352', 'SL004347', 'SL000496', 'SL004356', 'SL004355', 'SL000515', 'SL000524', 'SL000525', 'SL003191', 'SL000581', 'SL002077', 'SL000424', 'SL000020', 'SL004415', 'SL004128', 'SL004625', 'SL001995', 'SL006924', 'SL001902', 'SL002081', 'SL004726', 'SL004704', 'SL004644', 'SL000427', 'SL004645', 'SL006915', 'SL002763', 'SL003800', 'SL003915', 'SL004301', 'SL000695', 'SL000134', 'SL003332', 'SL004649', 'SL004650', 'SL003177', 'SL001897', 'SL003199', 'SL000605', 'SL004652', 'SL004768', 'SL008008', 'SL000320', 'SL006911', 'SL000409', 'SL006914', 'SL004725', 'SL004723', 'SL004718', 'SL003793', 'SL004724', 'SL000554', 'SL004009', 'SL006922', 'SL006923', 'SL004305', 'SL004306', 'SL006372', 'SL000248', 'SL000323', 'SL002783', 'SL003303', 'SL004330', 'SL003329', 'SL000480', 'SL002621', 'SL003302', 'SL000131', 'SL002541', 'SL000006', 'SL004668', 'SL004669', 'SL004670', 'SL000396', 'SL000398', 'SL005156', 'SL000007', 'SL003327', 'SL005168', 'SL004067', 'SL005171', 'SL000047', 'SL000506', 'SL000526', 'SL003542', 'SL003192', 'SL000048', 'SL002561', 'SL004362', 'SL001720', 'SL004686', 'SL004851', 'SL000136', 'SL004078', 'SL000668', 'SL004855', 'SL002655', 'SL004856', 'SL004872', 'SL003169', 'SL005160', 'SL005165', 'SL003173', 'SL017610', 'SL005256', 'SL005170', 'SL001802', 'SL001997', 'SL004850', 'SL004861', 'SL004875', 'SL005193', 'SL005194', 'SL005196', 'SL004516', 'SL005205', 'SL004862', 'SL004939', 'SL005219', 'SL005059', 'SL004687', 'SL000384', 'SL000250', 'SL000004', 'SL000338', 'SL003301', 'SL005157','SL005158', 'SL000420', 'SL000428', 'SL005164', 'SL004271', 'SL004354', 'SL001795', 'SL003326', 'SL000519', 'SL008416', 'SL000164', 'SL000532', 'SL003323', 'SL002704', 'SL004260', 'SL000603', 'SL000017', 'SL000633', 'SL004343', 'SL000437', 'SL003308', 'SL005204', 'SL004689', 'SL004327', 'SL000325', 'SL000343', 'SL004339', 'SL003744', 'SL005234', 'SL000468', 'SL000478', 'SL001718', 'SL002508', 'SL003309', 'SL003324', 'SL002640', 'SL005152', 'SL005227', 'SL005228', 'SL005233', 'SL004771', 'SL004750', 'SL006918', 'SL003733', 'SL004486', 'SL002524', 'SL003305', 'SL001800', 'SL005215', 'SL004626', 'SL010368', 'SL007261', 'SL004469', 'SL008956', 'SL008611', 'SL008421', 'SL006610', 'SL010369', 'SL007280', 'SL009213', 'SL000344', 'SL004673', 'SL010371', 'SL000358', 'SL002525', 'SL007284', 'SL010372', 'SL000658', 'SL012783', 'SL002695', 'SL008759', 'SL000678', 'SL008023', 'SL007122', 'SL008639', 'SL003916', 'SL003918', 'SL003863', 'SL010393', 'SL007100', 'SL008904', 'SL006992', 'SL005437', 'SL010374', 'SL010375', 'SL000640', 'SL006397', 'SL005797', 'SL004457', 'SL000158', 'SL010378', 'SL003770', 'SL010379', 'SL004118', 'SL010390', 'SL010391', 'SL004661', 'SL006803', 'SL006544', 'SL007804', 'SL000310', 'SL010449', 'SL008773', 'SL002706', 'SL010450', 'SL006108', 'SL010451', 'SL008623', 'SL010454', 'SL010455', 'SL007049', 'SL004438', 'SL010457', 'SL017613', 'SL008609', 'SL010461', 'SL006542', 'SL004858', 'SL010463', 'SL004466', 'SL004860', 'SL005087', 'SL007385', 'SL010464', 'SL004610', 'SL007674', 'SL010465', 'SL004805', 'SL007673', 'SL004515', 'SL010468', 'SL010469', 'SL001998', 'SL007206', 'SL007207', 'SL006892', 'SL005574', 'SL000272', 'SL010520', 'SL002792', 'SL003994', 'SL010491', 'SL010492', 'SL010288', 'SL004868', 'SL010495', 'SL010496', 'SL010522', 'SL003726', 'SL010610', 'SL009400', 'SL004781', 'SL006910', 'SL010612', 'SL006550', 'SL006777', 'SL004654', 'SL010499', 'SL010613', 'SL000064', 'SL007228', 'SL010500', 'SL010508', 'SL010510', 'SL000550', 'SL010512', 'SL007250', 'SL017614', 'SL005258', 'SL000565', 'SL010616', 'SL007560', 'SL007747', 'SL005261', 'SL010617', 'SL010619', 'SL010529', 'SL000104', 'SL003703', 'SL006374', 'SL004660', 'SL010521', 'SL010490', 'SL010493', 'SL004869', 'SL004635', 'SL010494', 'SL005250', 'SL007324', 'SL004137', 'SL010498', 'SL010349', 'SL010348', 'SL004636', 'SL009324', 'SL004298', 'SL017188', 'SL010523', 'SL004151', 'SL004152', 'SL000039', 'SL004125', 'SL004876', 'SL001797', 'SL006916', 'SL006917', 'SL005084', 'SL004155', 'SL009089', 'SL010513', 'SL010515', 'SL001945', 'SL010517', 'SL000582', 'SL005224', 'SL002705', 'SL005225', 'SL006480', 'SL008122', 'SL011073', 'SL000271', 'SL000283', 'SL000299', 'SL003167', 'SL000342', 'SL004331', 'SL000408', 'SL004335', 'SL003172', 'SL002075', 'SL001713', 'SL004350', 'SL003179', 'SL004536', 'SL000507', 'SL000508', 'SL003187', 'SL003190', 'SL004008', 'SL004712', 'SL004015', 'SL003196', 'SL000089', 'SL000589', 'SL000615', 'SL003862', 'SL004367', 'SL009216', 'SL004258', 'SL000249', 'SL000251', 'SL005392', 'SL009341', 'SL007056', 'SL000311', 'SL004865', 'SL010489','SL003711', 'SL007121', 'SL006029', 'SL008936', 'SL007640', 'SL006694', 'SL011049', 'SL004120', 'SL009412', 'SL008916', 'SL004060', 'SL004845', 'SL010462', 'SL008504', 'SL006512', 'SL007806', 'SL008909', 'SL007059', 'SL011069', 'SL010376', 'SL010501', 'SL006805', 'SL004804', 'SL008193', 'SL005491', 'SL010530', 'SL007327', 'SL005263', 'SL000566', 'SL008588', 'SL010381', 'SL010518', 'SL018625', 'SL003201', 'SL007531', 'SL004864', 'SL004720', 'SL007471', 'SL006830', 'SL003679', 'SL003919', 'SL004580', 'SL010466', 'SL000467', 'SL000254', 'SL000252', 'SL000617', 'SL000268', 'SL000382', 'SL000458', 'SL000076', 'SL002517', 'SL000306', 'SL003970', 'SL011509', 'SL000570', 'SL011510', 'SL011448', 'SL003993', 'SL000346', 'SL002684', 'SL000542', 'SL000481', 'SL000645', 'SL003974', 'SL000027', 'SL004304', 'SL003200', 'SL004642', 'SL007642', 'SL000638', 'SL005508', 'SL004867', 'SL000377', 'SL007888', 'SL008382', 'SL007403', 'SL010458', 'SL004844', 'SL007651', 'SL003990', 'SL004063', 'SL006912', 'SL002086', 'SL006913', 'SL007423', 'SL007620', 'SL007358', 'SL010503', 'SL010504', 'SL004765', 'SL010505', 'SL010502', 'SL010509', 'SL011071', 'SL003761', 'SL010514', 'SL010467', 'SL004492', 'SL010528', 'SL011629', 'SL004823', 'SL011530', 'SL011532', 'SL003785', 'SL000449', 'SL008102', 'SL011631', 'SL004919', 'SL008063', 'SL010973', 'SL005630', 'SL011528', 'SL006919', 'SL011529', 'SL011630', 'SL011533', 'SL005679', 'SL000588', 'SL008157', 'SL011549', 'SL008835', 'SL004938', 'SL011232', 'SL003753', 'SL005685', 'SL005687', 'SL009431', 'SL010927', 'SL009628', 'SL006132', 'SL000493', 'SL009629', 'SL010328', 'SL001905', 'SL005629', 'SL004820', 'SL011709', 'SL007266', 'SL008808', 'SL005372', 'SL008143', 'SL004119', 'SL003171', 'SL000451', 'SL003680', 'SL004511', 'SL000322', 'SL003104', 'SL000414', 'SL004340', 'SL000426', 'SL000674', 'SL004068', 'SL000138', 'SL000461', 'SL004353', 'SL004351', 'SL004346', 'SL001943', 'SL000483', 'SL003183', 'SL003186', 'SL000517', 'SL004359', 'SL004360', 'SL002755', 'SL000537', 'SL000540', 'SL000541', 'SL000545', 'SL000560', 'SL003198', 'SL000088', 'SL000586', 'SL000613', 'SL000415', 'SL000124', 'SL000601', 'SL002093', 'SL000587', 'SL017611', 'SL008378', 'SL000247', 'SL003658', 'SL008039', 'SL011769', 'SL004920', 'SL004757', 'SL009207', 'SL012188', 'SL004795', 'SL005588', 'SL007022', 'SL004899', 'SL011211', 'SL003304', 'SL004146', 'SL011770', 'SL012168', 'SL002650', 'SL001999', 'SL004154', 'SL004921', 'SL006268', 'SL003687', 'SL011768', 'SL008331', 'SL005493', 'SL011772', 'SL006705', 'SL000539', 'SL005358', 'SL004901', 'SL008865', 'SL005034', 'SL011535', 'SL011202', 'SL006088', 'SL005115', 'SL000057', 'SL006698', 'SL003655', 'SL004812', 'SL011809', 'SL004940', 'SL013489', 'SL008516', 'SL013490', 'SL000062', 'SL013488', 'SL000572', 'SL000051', 'SL002922', 'SL007752', 'SL013570', 'SL000305', 'SL004837', 'SL007003', 'SL005161', 'SL004334', 'SL014308', 'SL004348', 'SL004349', 'SL005202', 'SL001888', 'SL004484', 'SL009988', 'SL003674', 'SL013988', 'SL014130', 'SL006713', 'SL009045', 'SL014093', 'SL014028', 'SL014088', 'SL007024', 'SL009791', 'SL014096', 'SL014108', 'SL006523', 'SL014069', 'SL006998', 'SL013754', 'SL001753', 'SL005488', 'SL014029', 'SL007502', 'SL008945', 'SL004908', 'SL004070', 'SL010519', 'SL000308', 'SL000314', 'SL000316', 'SL000321', 'SL001691', 'SL005187', 'SL005188', 'SL000470', 'SL005184', 'SL017189', 'SL000522', 'SL003764', 'SL000535', 'SL004363', 'SL012740', 'SL012822', 'SL008644', 'SL009768', 'SL014092', 'SL001896', 'SL014048', 'SL004708', 'SL006197', 'SL008822', 'SL014071', 'SL005575', 'SL014008', 'SL014094', 'SL014129', 'SL014009', 'SL013548', 'SL014111', 'SL013969', 'SL004299', 'SL012457', 'SL012108', 'SL006406', 'SL013989', 'SL014468', 'SL014070', 'SL014091', 'SL011499', 'SL011508', 'SL011498', 'SL000087', 'SL008085', 'SL001774', 'SL001726', 'SL000597', 'SL010830', 'SL004253', 'SL005361', 'SL000479', 'SL007869', 'SL003524', 'SL006119', 'SL004742', 'SL008590', 'SL012754', 'SL016129', 'SL005572', 'SL000347', 'SL007871', 'SL016148', 'SL007153', 'SL008072', 'SL004739', 'SL016128', 'SL001737', 'SL002823', 'SL004156', 'SL002654', 'SL001729', 'SL000070', 'SL000125', 'SL004080', 'SL004133', 'SL004160', 'SL003310', 'SL000003', 'SL003362', 'SL000357', 'SL000360', 'SL007756', 'SL000021', 'SL000516', 'SL009951', 'SL003461', 'SL000433', 'SL000313', 'SL003657', 'SL003710', 'SL004814', 'SL000622', 'SL000022', 'SL004482', 'SL005167', 'SL004064', 'SL003643', 'SL000280', 'SL003300', 'SL001766', 'SL000836', 'SL000460', 'SL003182', 'SL004714', 'SL000510', 'SL003189', 'SL005201', 'SL000521', 'SL000523', 'SL005102', 'SL005789', 'SL000024', 'SL000139', 'SL004605', 'SL004925', 'SL004208', 'SL004209', 'SL012538', 'SL011708', 'SL007729', 'SL005675', 'SL008177', 'SL008099', 'SL004866', 'SL000339', 'SL008380', 'SL003728', 'SL013240', 'SL011628', 'SL008178', 'SL008709', 'SL004458', 'SL003522', 'SL006378', 'SL005352', 'SL012248', 'SL010373', 'SL002036', 'SL003341', 'SL006460', 'SL012469', 'SL007173', 'SL000670', 'SL002756', 'SL003930', 'SL006448', 'SL011616', 'SL007025', 'SL006522', 'SL001973', 'SL002646', 'SL007453', 'SL007281', 'SL006993', 'SL006920', 'SL001815', 'SL005846', 'SL003685', 'SL004296', 'SL004915', 'SL006091', 'SL003440', 'SL008933', 'SL004932', 'SL005694', 'SL002803', 'SL003653', 'SL004914', 'SL008176', 'SL009792', 'SL002522', 'SL008059', 'SL005764', 'SL006528', 'SL006629', 'SL008190', 'SL009868', 'SL004737', 'SL010388', 'SL000142', 'SL004782', 'SL007464', 'SL004852', 'SL004853', 'SL011100', 'SL014270', 'SL014208', 'SL004556', 'SL003739', 'SL004857', 'SL004791', 'SL014294', 'SL007179', 'SL008414', 'SL004859', 'SL007429', 'SL004849', 'SL005174', 'SL005181', 'SL005183', 'SL005185', 'SL005189', 'SL005190', 'SL005191', 'SL007328', 'SL007774', 'SL009202', 'SL012698', 'SL014269', 'SL014289', 'SL014209', 'SL005195', 'SL005197', 'SL005200', 'SL014288', 'SL005207', 'SL005208', 'SL007356', 'SL005703', 'SL005209', 'SL005210', 'SL009054', 'SL008728', 'SL014268', 'SL005212', 'SL005213', 'SL007680', 'SL014148', 'SL008309', 'SL005214', 'SL005169', 'SL014292', 'SL014228', 'SL005221', 'SL005222', 'SL004863', 'SL005223', 'SL002078', 'SL007547', 'SL004366', 'SL005230', 'SL005231', 'SL011406', 'SL016548', 'SL002542', 'SL006476', 'SL016554', 'SL007373', 'SL005308', 'SL015728', 'SL016550', 'SL016551', 'SL006921', 'SL006189', 'SL014735', 'SL016557', 'SL012707', 'SL007145', 'SL013928', 'SL007237', 'SL004716', 'SL016549', 'SL005730', 'SL005793', 'SL016566', 'SL010928', 'SL016555', 'SL016553', 'SL011404', 'SL011405', 'SL016567', 'SL007181','SL010488', 'SL014470', 'SL003520', 'SL005725', 'SL003331', 'SL014469', 'SL016563', 'SL000406', 'SL004400', 'SL000401', 'SL000277', 'SL000558', 'SL007195', 'SL003647', 'SL004131', 'SL004477', 'SL003717', 'SL008703', 'SL006675', 'SL011808', 'SL000130', 'SL006970', 'SL004891', 'SL004145', 'SL000655', 'SL004648', 'SL003080', 'SL014113', 'SL008574', 'SL002702', 'SL010470', 'SL007336', 'SL004671', 'SL016130', 'SL003184', 'SL004871', 'SL007295', 'SL004134', 'SL003690', 'SL003166', 'SL010456', 'SL011068', 'SL000553', 'SL000556', 'SL016928', 'SL010516', 'SL010250', 'SL010384', 'SL010471', 'SL009790']

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the classifier
clf.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf, threshold=0.005)

# Train the selector
sfm.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])


# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

plt.bar(range(len(clf_important.feature_importances_)), clf_important.feature_importances_)
plt.show()

# View The Accuracy Of Our Full Feature (4 Features) Model
print("Accuracy with all features:")
print(accuracy_score(y_test, y_pred))


# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
print("Accuracy with selected features:")
print(accuracy_score(y_test, y_important_pred))
