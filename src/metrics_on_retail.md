ML-KNN (TF-IDF):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8318488529014845
    Accuracy (ML): 0.9659801678108313
    Precision (macro): 0.8306974479881335
    Precision (micro): 0.8296555750145943
    Recall (macro): 0.8307168608224904
    Recall (micro): 0.8315001170138077
    Hamming loss: 0.034019832189168576


ML-KNN (Word2Vec):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8226720647773279
    Accuracy (ML): 0.9641260341489174
    Precision (macro): 0.8265624390209665
    Precision (micro): 0.8219354838709677
    Recall (macro): 0.8175517040967598
    Recall (micro): 0.8199157500585069
    Hamming loss: 0.03587396585108256

ML-KNN (BERT embeddings):
    param_grid={
        'k': range(1,3),
        's': [0.5, 0.7, 1.0]
    }

    best = MLkNN(k=1, s=0.5)

    Accuracy (subset): 0.8283400809716599
    Accuracy (ML): 0.9656867922314146
    Precision (macro): 0.8315168637183016
    Precision (micro): 0.8300845467355566
    Recall (macro): 0.8249676767342544
    Recall (micro): 0.8271706061315235
    Hamming loss: 0.03431320776858534

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
    Accuracy (subset): 0.6723346828609986
    Accuracy (ML): 0.9371472158657513
    Precision (macro): 0.6950372602354618
    Precision (micro): 0.684393063583815
    Recall (macro): 0.6911678123015544
    Recall (micro): 0.6927217411654575
    Hamming loss: 0.06285278413424866

Classifier Chain (Word2Vec):
    Accuracy (subset): 0.5479082321187584
    Accuracy (ML): 0.9110954644135422
    Precision (macro): 0.5705298040709857
    Precision (micro): 0.5521169138190415
    Recall (macro): 0.6026049699109812
    Recall (micro): 0.6012169435993447
    Hamming loss: 0.08890453558645778

Classifier Chain (BERT Embeddings):
    Accuracy (subset): 0.5292847503373819
    Accuracy (ML): 0.9072581118347707
    Precision (macro): 0.5456093994288039
    Precision (micro): 0.5348811977866985
    Recall (macro): 0.577693782310587
    Recall (micro): 0.5768780716124503
    Hamming loss: 0.09274188816522913

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
    Accuracy (subset): 0.4987854251012146
    Accuracy (ML): 0.9464178841753212
    Precision (macro): 0.9242072294053362
    Precision (micro): 0.9301772589710333
    Recall (macro): 0.5011711622506411
    Recall (micro): 0.503510414228879
    Hamming loss: 0.053582115824678755

CatBoost (Word2Vec):
    Accuracy (subset): 0.6232118758434548
    Accuracy (ML): 0.9535645132899139
    Precision (macro): 0.8479527485888295
    Precision (micro): 0.8585716518205969
    Recall (macro): 0.6427591645472507
    Recall (micro): 0.6428738591153756
    Hamming loss: 0.04643548671008625

CatBoost (BERT Embeddings):
    Accuracy (subset): 0.5924426450742241
    Accuracy (ML): 0.9496684855952592
    Precision (macro): 0.8226649856368919
    Precision (micro): 0.8382329572540919
    Recall (macro): 0.6157665414600738
    Recall (micro): 0.6172478352445588
    Hamming loss: 0.05033151440474095

CatBoost (Default):
    Accuracy (subset): 0.805668016194332
    Accuracy (ML): 0.9728686264155372
    Precision (macro): 0.8966999029633844
    Precision (micro): 0.9008487654320988
    Recall (macro): 0.8185683222319481
    Recall (micro): 0.8196817224432483
    Hamming loss: 0.02713137358446283


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
    Accuracy (subset): 0.6240215924426451
    Accuracy (ML): 0.9263744645895674
    Precision (macro): 0.6361174174732991
    Precision (micro): 0.6360479041916167
    Recall (macro): 0.6211240809655628
    Recall (micro): 0.6214603323192137
    Hamming loss: 0.07362553541043243

RakelO (Word2Vec):
    Accuracy (subset): 0.5600539811066126
    Accuracy (ML): 0.9195329460775686
    Precision (macro): 0.606971085338987
    Precision (micro): 0.6077305778798316
    Recall (macro): 0.5576893496454014
    Recall (micro): 0.5574537795459864
    Hamming loss: 0.0804670539224315

RakelO (BERT Embeddings):
    Accuracy (subset): 0.5246963562753036
    Accuracy (ML): 0.9131960335621664
    Precision (macro): 0.5719989188254623
    Precision (micro): 0.5734748689090676
    Recall (macro): 0.5237189310070938
    Recall (micro): 0.5246899134097823
    Hamming loss: 0.08680396643783371

BERT:
    Correct predictions (23 clases):
        [3667., 3606., 3667., 3579., 3609., 3634., 3597., 3657., 3657.,
        3555., 3517., 3411., 3579., 3599., 3544., 3597., 3556., 3407.,
        3605., 3578., 3544., 3520., 3633.]

    Accuracy (subset): 0.7715749740600586
    Accuracy (ML): 0.9660036378571847
    Precision (macro): 0.9571549176867876
    Precision (micro): 0.8636331336169524
    Recall (macro): 0.9246595688132201
    Recall (micro): 0.7830361326316018
    Hamming loss: 0.03410956263542175

DeBERTa:
    Correct predictions (23 classes):
        [3680., 3590., 3680., 3585., 3590., 3616., 3599., 3665., 3665.,
            3572., 3522., 3435., 3588., 3606., 3536., 3601., 3570., 3437.,
            3589., 3586., 3537., 3521., 3615.]

    Accuracy (subset): 0.8153846263885498
    Accuracy (ML): 0.9667898844100218
    Precision (macro): 0.9847209998239747
    Precision (micro): 0.8437112010796222
    Recall (macro): 0.9820688845860471
    Recall (micro): 0.8200629779577148
    Hamming loss: 0.033210113644599915