

Teste usando todos os 19 talhões e a técnica de validação GroupKFold, realizada no dia 05/04/2024. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html link de uma explicação mais detalhada de como funciona o GroupKFold

Primeiro foi importado as bibliotecas que foram utilizadas para a manipulação dos dados.

Depois, combined_df = pd.concat([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19], ignore_index=True)
combina múltiplos DataFrames em um único DataFrame, usando a função pd.concat() do pandas. 

Posteriormente, X = combined_df.drop(columns=['fid', 'lat', 'long', 'talhao', 'wkt_geom','Altitude1', 'produtivid' ]): Esta linha cria um DataFrame X contendo todas as colunas do DataFrame combined_df, exceto aquelas que foram removidas. As colunas removidas parecem ser identificadores (fid), coordenadas geográficas (lat e long), informações sobre talhões (talhao), geometria (wkt_geom), altitude (Altitude1), e a coluna que parece ser a variável alvo (produtivid).
y = combined_df['produtivid']: Esta linha cria uma série y contendo apenas a coluna 'produtivid' do DataFrame combined_df, que parece ser a variável alvo que você deseja prever.
scaler = StandardScaler(): Você está criando uma instância do StandardScaler, que é usado para padronizar as features dimensionando-as para que tenham média zero e variância unitária.
X_scaled = scaler.fit_transform(X): Esta linha ajusta o scaler aos dados (fit) e, em seguida, transforma (transform) as features do DataFrame X de acordo com os parâmetros ajustados.

No trecho seguinte, foi importado as biblitecas a serem utilizadas na etapa de validação cruzada usando o GroupKfold.
Foi configurada o GroupKFold, from sklearn.model_selection import GroupKFold: Esta linha importa a classe GroupKFold da biblioteca scikit-learn, que é usada para realizar validação cruzada com dados agrupados, from sklearn.ensemble import RandomForestRegressor: Esta linha importa a classe RandomForestRegressor , from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score: Esta linha importa algumas métricas de avaliação para problemas de regressão, import numpy as np: Esta linha importa a biblioteca NumPy, que é usada para trabalhar com arrays e operações matemáticas.

Para finalizar, o último código realiza validação cruzada com agrupamento (GroupKFold) e avalia o desempenho do modelo RandomForestRegressor em cada fold.
groups = combined_df['talhao']: Esta linha define os grupos para a validação cruzada. Cada grupo corresponde a um talhão, como especificado pela coluna 'talhao' do DataFrame combined_df.
group_kfold = GroupKFold(n_splits=4): Esta linha cria uma instância de GroupKFold com 4 folds (n_splits=4) para a validação cruzada. Isso dividirá os dados em 4 partes, mantendo os talhões agrupados intactos.
fold_mse, fold_mae, fold_r2, fold_rmse = [], [], [], []: Aqui são criadas listas vazias para armazenar as métricas de avaliação (MSE, MAE, R² e RMSE) de cada fold.
O loop for fold, (train_index, test_index) in enumerate(group_kfold.split(X_scaled, groups=groups)): itera sobre cada fold gerado pela validação cruzada.a. X_train, X_test = X_scaled[train_index], X_scaled[test_index]: Divide os dados de entrada (X_scaled) em conjuntos de treinamento e teste de acordo com os índices fornecidos pelos folds.b. y_train, y_test = y.iloc[train_index], y.iloc[test_index]: Divide os rótulos (y) correspondentes aos dados de entrada em conjuntos de treinamento e teste de acordo com os índices fornecidos pelos folds.c. rf_model = RandomForestRegressor(n_estimators=1000, random_state=42, min_samples_leaf=30, n_jobs=-1): Cria uma instância do RandomForestRegressor com os parâmetros especificados.d. rf_model.fit(X_train, y_train): Treina o modelo RandomForestRegressor nos dados de treinamento.e. y_pred = rf_model.predict(X_test): Realiza previsões nos dados de teste usando o modelo treinado.f. Calcula as métricas de avaliação (MSE, MAE, R² e RMSE) comparando as previsões (y_pred) com os valores reais (y_test).g. Armazena as métricas de avaliação em listas correspondentes a cada fold.
Após o loop, o código imprime as métricas de avaliação de cada fold na tela e calcula a média das métricas sobre todos os folds.
