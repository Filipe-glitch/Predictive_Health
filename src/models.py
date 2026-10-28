
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class ModelBenchmark:
    """
    Gerencia, treina e compara múltiplos algoritmos
    de Machine Learning para classificação.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = {}

    def _initialize_models(self):
        """
        Cria pipelines com imputação e, quando necessário,
        padronização antes do treinamento de cada modelo.
        """

        modelos_com_escala = {
            "Logistic Regression": LogisticRegression(
                max_iter=5000,
                random_state=self.random_state
            ),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
                n_neighbors=5
            ),
            "Support Vector Machine (SVM)": SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state
            ),
            "Neural Network (MLP)": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state
            )
        }

        modelos_sem_escala = {
            "Baseline (Dummy)": DummyClassifier(
                strategy="stratified",
                random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=4,
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state
            )
        }

        modelos = {}

        for nome, modelo in modelos_com_escala.items():
            modelos[nome] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("modelo", modelo)
            ])

        for nome, modelo in modelos_sem_escala.items():
            modelos[nome] = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("modelo", modelo)
            ])

        return modelos

    def train_and_evaluate(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        cv_folds=5
    ):
        """
        Treina os modelos, avalia o desempenho no conjunto
        de teste e executa validação cruzada estratificada.
        """

        self.results = {}

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        for nome, modelo in self.models.items():

            # Treinamento usando apenas os dados de treino
            modelo.fit(X_train, y_train)

            # Previsões no conjunto de teste
            y_pred = modelo.predict(X_test)

            # Probabilidades para calcular ROC-AUC
            if hasattr(modelo, "predict_proba"):
                y_proba = modelo.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = None
                auc = np.nan

            # Validação cruzada: cada divisão ajusta
            # seu próprio pré-processamento
            cv_scores = cross_val_score(
                modelo,
                X_train,
                y_train,
                cv=skf,
                scoring="accuracy"
            )

            # Métricas de avaliação
            acc = accuracy_score(y_test, y_pred)

            prec = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            rec = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            self.results[nome] = {
                "model_object": modelo,
                "Accuracy": acc,
                "Precision": prec,
                "Recall (Sensibilidade)": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "CV Mean Accuracy": cv_scores.mean(),
                "CV Std Accuracy": cv_scores.std(),
                "y_pred": y_pred,
                "y_proba": y_proba
            }

        resultados = pd.DataFrame(self.results).T

        return resultados.drop(
            columns=["model_object", "y_pred", "y_proba"]
        )
