ML-KNN:
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

ML-KNN (TF-IDF):
    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8918025561921551
    Accuracy (ML): 0.9990699777695541
    Precision (macro): 0.8593072155313445
    Precision (micro): 0.918602861978766
    Recall (macro): 0.8627480407472846
    Recall (micro): 0.9193101324299353
    Hamming loss: 0.0009300222304459072

ML-KNN (Word2Vec):
    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8887174966945791
    Accuracy (ML): 0.9989586930582187
    Precision (macro): 0.8474461188225638
    Precision (micro): 0.9084691054411312
    Recall (macro): 0.8564324252657364
    Recall (micro): 0.9101478287650139
    Hamming loss: 0.001041306941781315


ML-KNN (BERT Embeddings):
    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8913618334067871
    Accuracy (ML): 0.9990390653497386
    Precision (macro): 0.855330116138894
    Precision (micro): 0.9153349723417332
    Recall (macro): 0.8596435650225405
    Recall (micro): 0.9173082845703726
    Hamming loss: 0.0009609346502612982

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
    Accuracy (subset): 0.8913618334067871
    Accuracy (ML): 0.9991194376412587
    Precision (macro): 0.8751858191848155
    Precision (micro): 0.9248068006182381
    Recall (macro): 0.8696999252314739
    Recall (micro): 0.9213889744379428
    Hamming loss: 0.0008805623587412816

Classifier Chain (Word2Vec):
    Accuracy (subset): 0.8884971353018951
    Accuracy (ML): 0.9987069776397218
    Precision (macro): 0.8191524466191092
    Precision (micro): 0.8734224201930215
    Recall (macro): 0.8538772066767027
    Recall (micro): 0.9058361564521097
    Hamming loss: 0.0012930223602780704

Classifier Chain (BERT Embeddings):
    Accuracy (subset): 0.8889378580872631
    Accuracy (ML): 0.9987529046634477
    Precision (macro): 0.8144362520560097
    Precision (micro): 0.8802753666566896
    Recall (macro): 0.8544029577460736
    Recall (micro): 0.9057591623036649
    Hamming loss: 0.0012470953365523467

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
    Accuracy (subset): 0.0370207139709123
    Accuracy (ML): 0.9950098522298012
    Precision (macro): 0.5800204278958203
    Precision (micro): 0.9093113482056256
    Recall (macro): 0.16148175760401037
    Recall (micro): 0.14436402833384662
    Hamming loss: 0.004990147770198837

CatBoost (Word2Vec):
    Accuracy (subset): 0.6077567210224769
    Accuracy (ML): 0.9986305798021782
    Precision (macro): 0.9233314530371447
    Precision (micro): 0.9887296094908552
    Recall (macro): 0.8295964478358093
    Recall (micro): 0.7700184785956268
    Hamming loss: 0.0013694201978218226

CatBoost (BERT Embeddings):
    Accuracy (subset): 0.5592772146319964
    Accuracy (ML): 0.9984362731633385
    Precision (macro): 0.9235904318110776
    Precision (micro): 0.9879144716454912
    Recall (macro): 0.8208206192710897
    Recall (micro): 0.7363720357252849
    Hamming loss: 0.0015637268366614234

CatBoost (Default):
    Accuracy (subset): 0.13089466725429705
    Accuracy (ML): 0.9962004219986911
    Precision (macro): 0.3272404456520776
    Precision (micro): 0.8659432387312187
    Recall (macro): 0.1393918603086439
    Recall (micro): 0.3993686479827533
    Hamming loss: 0.00379957800130892

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
    Accuracy (subset): 0.8891582194799471
    Accuracy (ML): 0.9990646784975856
    Precision (macro): 0.8805824161317456
    Precision (micro): 0.9366864856178692
    Recall (macro): 0.8513876420426756
    Recall (micro): 0.8975977825685247
    Hamming loss: 0.00093532150241426

RakelO (Word2Vec):
    Accuracy (subset): 0.8884971353018951
    Accuracy (ML): 0.9993667369997818
    Precision (macro): 0.9228556768932341
    Precision (micro): 0.993170565135735
    Recall (macro): 0.8503068599451714
    Recall (micro): 0.8957499230058515
    Hamming loss: 0.0006332630002181534