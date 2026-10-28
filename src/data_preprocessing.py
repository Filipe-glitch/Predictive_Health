
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


class DiabetesDataPipeline:
    """
    Carrega os dados, identifica zeros suspeitos e separa
    os conjuntos de treino e teste.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.cleaned_data = None

    def load_data(self):
        self.raw_data = pd.read_csv(self.filepath)
        return self.raw_data.copy()

    def clean_data(self):
        """
        Substitui zeros suspeitos por NaN.
        Não utiliza a variável-alvo Outcome no tratamento.
        """

        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        invalid_zero_cols = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        for col in invalid_zero_cols:
            df[col] = df[col].replace(0, np.nan)

        self.cleaned_data = df
        return self.cleaned_data.copy()

    def detect_outliers_sd(self, column="BMI", n_std=2):
        """
        Identifica valores fora da faixa de média +/- n desvios-padrão.
        A identificação não significa necessariamente que sejam erros.
        """

        if self.cleaned_data is None:
            self.clean_data()

        df = self.cleaned_data
        series = df[column].dropna()

        mean_val = series.mean()
        std_val = series.std()

        upper_limit = mean_val + n_std * std_val
        lower_limit = mean_val - n_std * std_val

        outliers = df[
            (df[column] > upper_limit) |
            (df[column] < lower_limit)
        ]

        return outliers, mean_val, std_val

    def prepare_train_test(self, test_size=0.2, random_state=42):
        """
        Separa treino e teste sem imputar ou escalar previamente.
        O pré-processamento será ajustado dentro de cada modelo.
        """

        if self.cleaned_data is None:
            self.clean_data()

        X = self.cleaned_data.drop(columns=["Outcome"])
        y = self.cleaned_data["Outcome"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X.columns.tolist()
        )
