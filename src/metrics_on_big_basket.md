ML-KNN:
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)


ML-KNN (TF-IDF):
    Accuracy (subset): 0.7816222348269994
    Accuracy (ML): 0.9935370216850649
    Precision (macro): 0.7376699073085848
    Precision (micro): 0.8514760914760915
    Recall (macro): 0.7288412123297471
    Recall (micro): 0.8526815456362425
    Hamming loss: 0.006462978314935207

ML-KNN (Word2Vec):
    Accuracy (subset): 0.742484401588202
    Accuracy (ML): 0.9923589598150008
    Precision (macro): 0.6923695251228656
    Precision (micro): 0.8252896557472702
    Recall (macro): 0.6897772201537334
    Recall (micro): 0.8245336442371752
    Hamming loss: 0.007641040184999346

ML-KNN (BERT Embeddings):
    Accuracy (subset): 0.7647948572508981
    Accuracy (ML): 0.9929716247073025
    Precision (macro): 0.7194739236326118
    Precision (micro): 0.83925
    Recall (macro): 0.7110272512403599
    Recall (micro): 0.838690872751499
    Hamming loss: 0.007028375292697471

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
    Accuracy (subset): 0.6337681981470977
    Accuracy (ML): 0.989693776633652
    Precision (macro): 0.6556083658642693
    Precision (micro): 0.7616178291374329
    Recall (macro): 0.6395108905566944
    Recall (micro): 0.768404397068621
    Hamming loss: 0.010306223366348153

Classifier Chain (Word2Vec):
    Accuracy (subset): 0.4758933635847986
    Accuracy (ML): 0.9835234957895196
    Precision (macro): 0.45482268410855237
    Precision (micro): 0.6179793285794407
    Recall (macro): 0.4749428744371764
    Recall (micro): 0.6423217854763491
    Hamming loss: 0.016476504210480386

Classifier Chain (BERT Embeddings):
    Accuracy (subset): 0.44583096993760635
    Accuracy (ML): 0.9819945605538346
    Precision (macro): 0.43733991249492
    Precision (micro): 0.5819825436408977
    Recall (macro): 0.4535442029948647
    Recall (micro): 0.6219187208527648
    Hamming loss: 0.01800543944616548

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
    Accuracy (subset): 0.4286254490451881
    Accuracy (ML): 0.9894737990313713
    Precision (macro): 0.7531470229838161
    Precision (micro): 0.8845868381989115
    Recall (macro): 0.3685328410748345
    Recall (micro): 0.5955196535642905
    Hamming loss: 0.010526200968628649

CatBoost (Word2Vec):
    Accuracy (subset): 0.47003214218188694
    Accuracy (ML): 0.9897701324956005
    Precision (macro): 0.7489442813514712
    Precision (micro): 0.8395232521017346
    Recall (macro): 0.4820822846172004
    Recall (micro): 0.6569786808794137
    Hamming loss: 0.010229867504399552

CatBoost (BERT embeddings):
    Accuracy (subset): 0.4505577613915674
    Accuracy (ML): 0.9893283592943265
    Precision (macro): 0.7358690200420686
    Precision (micro): 0.8424458826154876
    Recall (macro): 0.4516851021226931
    Recall (micro): 0.6287475016655563
    Hamming loss: 0.010671640705673604

CatBoost (Default):
    Accuracy (subset): 0.5882019285309132
    Accuracy (ML): 0.9925789374172811
    Precision (macro): 0.7035416421137527
    Precision (micro): 0.868514041286963
    Recall (macro): 0.5288159435947005
    Recall (micro): 0.7778147901399067
    Hamming loss: 0.007421062582718851

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
    Accuracy (subset): 0.598789941387786
    Accuracy (ML): 0.9871794871794871
    Precision (macro): 0.5813395382934666
    Precision (micro): 0.7145454545454546
    Recall (macro): 0.5143403141253783
    Recall (micro): 0.6872918054630246
    Hamming loss: 0.01282051282051282

RakelO (Word2Vec):
    Accuracy (subset): 0.497447532614861
    Accuracy (ML): 0.985908707477057
    Precision (macro): 0.5567118935981195
    Precision (micro): 0.7037427012539486
    Recall (macro): 0.4384078727457376
    Recall (micro): 0.6122584943371085
    Hamming loss: 0.014091292522943118

RakelO (BERT Embeddings):
    Accuracy (subset): 0.4639818491208168
    Accuracy (ML): 0.9848851753276029
    Precision (macro): 0.5464503809467235
    Precision (micro): 0.6848478783026422
    Recall (macro): 0.39600513375125074
    Recall (micro): 0.5698700866089274
    Hamming loss: 0.015114824672396993