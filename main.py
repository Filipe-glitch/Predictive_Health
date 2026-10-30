from pathlib import Path
from src.data_preprocessing import DiabetesDataPipeline
from src.models import ModelBenchmark


def main():
    # Localiza o arquivo de dados a partir da raiz do projeto
    raiz_projeto = Path(__file__).resolve().parent
    caminho_dados = raiz_projeto / "data" / "diabetes.csv"

    # Carrega, limpa e separa os dados
    pipeline = DiabetesDataPipeline(str(caminho_dados))
    pipeline.load_data()
    pipeline.clean_data()

    X_train, X_test, y_train, y_test, colunas = (
        pipeline.prepare_train_test()
    )

    print("Dados preparados com sucesso!")
    print(f"Treino: {X_train.shape}")
    print(f"Teste: {X_test.shape}")

    # Treina e compara os modelos
    benchmark = ModelBenchmark()
    resultados = benchmark.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Exibe os resultados
    print("\nResultados dos modelos:")
    print(resultados.round(4).to_string())
    
    pasta_outputs = raiz_projeto / "outputs"
    pasta_outputs.mkdir(exist_ok=True)

    # Organiza os resultados para consulta e uso no portfólio
    tabela_final = resultados[
        [
            "Accuracy",
            "Precision",
            "Recall (Sensibilidade)",
            "F1-Score",
            "ROC-AUC",
            "CV Mean Accuracy",
            "CV Std Accuracy"
        ]
    ].sort_values(
        by="F1-Score",
        ascending=False
    )

    caminho_resultados = pasta_outputs / "resultados_modelos.csv"
    tabela_final.to_csv(caminho_resultados, index_label="Modelo")

    print("\nResultados salvos em: outputs/resultados_modelos.csv")

    # Identifica o modelo com maior F1-score
    melhor_modelo = resultados["F1-Score"].idxmax()
    melhor_f1 = resultados.loc[melhor_modelo, "F1-Score"]

    print(f"\nMelhor F1-score no teste: {melhor_modelo}")
    print(f"F1-score: {melhor_f1:.4f}")


if __name__ == "__main__":
    main()
