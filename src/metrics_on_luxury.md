ML-KNN (TF-IDF):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.7159450897571278
    Accuracy (ML): 0.9820710418117685
    Precision (macro): 0.7002182951149984
    Precision (micro): 0.7832512315270936
    Recall (macro): 0.6941980270072069
    Recall (micro): 0.7806873977086743
    Hamming loss: 0.017928958188231595

ML-KNN (Word2Vec):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.7159450897571278
    Accuracy (ML): 0.9823181828394258
    Precision (macro): 0.6936269029897656
    Precision (micro): 0.7873626373626373
    Recall (macro): 0.6713727223854106
    Recall (micro): 0.7817785051827605
    Hamming loss: 0.017681817160574265

ML-KNN (BERT Embeddings):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.7011615628299894
    Accuracy (ML): 0.9805881956458247
    Precision (macro): 0.6825968725869357
    Precision (micro): 0.764898851831602
    Recall (macro): 0.660657311400953
    Recall (micro): 0.7632296781232951
    Hamming loss: 0.01941180435417556

Classifier Chain:
    param_grid=[
        {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.7],
        },
        {
            'classifier': [DecisionTreeClassifier()],
            'classifier__criterion': ['log_loss'],
        },
    ]

    best = {
        'classifier': DecisionTreeClassifier(),
        'classifier__criterion': 'log_loss'
    }

Classifier Chain (TF-IDF):
    Accuracy (subset): 0.7243928194297783
    Accuracy (ML): 0.9879125570109415
    Precision (macro): 0.7757753598458432
    Precision (micro): 0.8547945205479452
    Recall (macro): 0.7802783956375158
    Recall (micro): 0.851063829787234
    Hamming loss: 0.012087442989058393

Classifier Chain (Word2Vec):
    Accuracy (subset): 0.3738120380147835
    Accuracy (ML): 0.9638724752297286
    Precision (macro): 0.43833249213533154
    Precision (micro): 0.5605815831987075
    Recall (macro): 0.4547841145816238
    Recall (micro): 0.5679214402618658
    Hamming loss: 0.03612752477027118

Classifier Chain (BERT Embeddings):
    Accuracy (subset): 0.3706441393875396
    Accuracy (ML): 0.962726639556045
    Precision (macro): 0.43058084288554965
    Precision (micro): 0.5475409836065573
    Recall (macro): 0.4189528190184782
    Recall (micro): 0.546644844517185
    Hamming loss: 0.037273360443955156

CatBoost:
    grid = {
        'iterations': [30, 60],
        'loss_function': ['MultiLogloss', 'MultiCrossEntropy'],
        'allow_const_label': [True],
        'random_state': [13],
    }

    best = {
        'allow_const_label': True,
        'iterations': 60,
        'loss_function': 'MultiLogloss',
        'random_state': 13
    }

CatBoost (TF-IDF):
    Accuracy (subset): 0.6853220696937699
    Accuracy (ML): 0.9896200768383924
    Precision (macro): 0.8574804018874744
    Precision (micro): 0.951878707976269
    Recall (macro): 0.6500885171653668
    Recall (micro): 0.7877795962902346
    Hamming loss: 0.010379923161607765

CatBoost (Word2Vec):
    Accuracy (subset): 0.4582893347412883
    Accuracy (ML): 0.9812622166303446
    Precision (macro): 0.7912662254082772
    Precision (micro): 0.9258312020460358
    Recall (macro): 0.41730798041323797
    Recall (micro): 0.5924713584288053
    Hamming loss: 0.018737783369655574

CatBoost (BERT Embeddings):
    Accuracy (subset): 0.44139387539598735
    Accuracy (ML): 0.9801613156889617
    Precision (macro): 0.7733768035093742
    Precision (micro): 0.9225978647686833
    Recall (macro): 0.390887010393713
    Recall (micro): 0.5657392253136934
    Hamming loss: 0.01983868431103822

CatBoost (Default):
    Accuracy (subset): 0.6916578669482577
    Accuracy (ML): 0.9902042283583098
    Precision (macro): 0.8236345336965119
    Precision (micro): 0.9474695707879565
    Recall (macro): 0.6405215242999983
    Recall (micro): 0.806873977086743
    Hamming loss: 0.009795771641690444

RakelO:
    param_grid=[
        {
            'base_classifier': [MultinomialNB()],
            'base_classifier__alpha': [0.7],
        },
        {
            'base_classifier': [DecisionTreeClassifier()],
            'base_classifier__criterion': ['log_loss'],
        },
    ]

    best = {
        'base_classifier': DecisionTreeClassifier(criterion='log_loss'),
        'base_classifier_require_dense': [True, True],
        'labelset_size': big_basket_y_train.to_numpy().shape[1],
        'model_count': 6,
    }

RakelO (TF-IDF):
    Accuracy (subset): 0.7444561774023232
    Accuracy (ML): 0.9848794625806017
    Precision (macro): 0.7053314392027467
    Precision (micro): 0.8247480403135499
    Recall (macro): 0.6781776732461658
    Recall (micro): 0.8036006546644845
    Hamming loss: 0.015120537419398323

RakelO (Word2Vec):
    Accuracy (subset): 0.4424498416050686
    Accuracy (ML): 0.9686130894875195
    Precision (macro): 0.5260292243564947
    Precision (micro): 0.6453333333333333
    Recall (macro): 0.38276432356001566
    Recall (micro): 0.5280960174577196
    Hamming loss: 0.031386910512480624

RakelO (BERT Embeddings):
    Accuracy (subset): 0.40865892291446676
    Accuracy (ML): 0.9663438855063021
    Precision (macro): 0.4693622430488629
    Precision (micro): 0.6124916051040967
    Recall (macro): 0.36869767547484106
    Recall (micro): 0.49754500818330605
    Hamming loss: 0.0336561144936979

BERT:
    Correct predictions (47 classes):
        [871., 890., 843., 927., 934., 928., 931., 933., 921., 943., 927.,
            932., 920., 885., 929., 931., 930., 905., 932., 922., 944., 937.,
            942., 845., 947., 925., 916., 924., 925., 941., 802., 927., 898.,
            888., 911., 934., 902., 923., 903., 857., 908., 940., 917., 905.,
            936., 945., 935.]

    Accuracy (subset): 0.0
    Accuracy (ML): 0.9663438855063023
    Precision (macro): 0.9989675913457222
    Precision (micro): 0.9571026722925456
    Recall (macro): 0.8800760092168656
    Recall (micro): 0.19203335342575847
    Hamming loss: 0.033658016473054886

DeBERTa:
    Correct predictions (47 classes):
        [932., 930., 910., 942., 942., 933., 940., 945., 919., 943., 945.,
            932., 939., 945., 946., 940., 935., 923., 945., 942., 944., 940.,
            942., 920., 947., 944., 944., 944., 925., 941., 928., 939., 925.,
            929., 936., 944., 944., 942., 930., 917., 946., 944., 930., 936.,
            938., 945., 946.]

    Accuracy (subset): 0.7170010805130005
    Accuracy (ML): 0.9894178705430363
    Precision (macro): 0.9962928845851402
    Precision (micro): 0.8970432946145723
    Recall (macro): 0.9931249859578963
    Recall (micro): 0.8299894403379092
    Hamming loss: 0.01058212947100401