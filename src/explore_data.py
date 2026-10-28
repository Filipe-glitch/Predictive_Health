import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print("Dimensões do conjunto:", df.shape)

print("\nPrimeiras linhas:")
print(df.head())

print("\nTipos de dados:")
print(df.dtypes)

print("\nValores ausentes por coluna:")
print(df.isna().sum())

print("\nQuantidade de zeros por coluna:")
print((df == 0).sum())

print("\nDistribuição de Outcome:")
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(3))


print("\nZeros por coluna e classe de Outcome:")

columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in columns_to_check:
    zero_by_class = df.groupby("Outcome")[col].apply(
        lambda values: (values == 0).sum()
    )

    total_by_class = df.groupby("Outcome").size()

    print(f"\n{col}")
    for outcome in [0, 1]:
        count = zero_by_class.loc[outcome]
        total = total_by_class.loc[outcome]
        print(
            f"  Outcome {outcome}: {count}/{total} "
            f"({count / total:.1%})"
        )


columns_to_check = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in columns_to_check:
    print(f"\n--- {col} ---")

    zero_mask = df[col] == 0

    for outcome in [0, 1]:
        group = df[df["Outcome"] == outcome]

        zero_count = (group[col] == 0).sum()
        positive_values = group.loc[group[col] > 0, col]

        print(f"Outcome {outcome}:")
        print(f"  Valores zero: {zero_count}")
        print(
            "  Mediana dos valores positivos:",
            round(positive_values.median(), 2)
            if not positive_values.empty else "Sem valores"
        )
