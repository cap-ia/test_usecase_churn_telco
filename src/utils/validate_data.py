from typing import List, Tuple

import great_expectations as gx
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation with Great Expectations...")

    # Copie pour ne pas modifier les données du pipeline.
    data = df.copy()

    required_columns = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    # Arrêter avant les contrôles de valeurs si le schéma est incomplet.
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        return False, [
            f"Missing required column: {column}"
            for column in missing_columns
        ]

    if data.empty:
        return False, ["Dataset is empty"]

    if not data.columns.is_unique:
        return False, ["Duplicate column names"]

    # Vérifier la conversion numérique sans masquer les textes invalides.
    conversion_errors = []

    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        original = data[column].replace(r"^\s*$", pd.NA, regex=True)
        numeric = pd.to_numeric(original, errors="coerce")

        invalid = original.notna() & numeric.isna()

        if invalid.any():
            conversion_errors.append(
                f"{column}: {int(invalid.sum())} non-numeric values"
            )

        data[column] = numeric

    if conversion_errors:
        return False, conversion_errors

    # Règle proposée pour les données brutes :
    # TotalCharges peut manquer uniquement si tenure == 0.
    invalid_missing_total = (
        data["TotalCharges"].isna()
        & ~data["tenure"].eq(0)
    )

    if invalid_missing_total.any():
        return False, [
            "TotalCharges is missing for "
            f"{int(invalid_missing_total.sum())} customers "
            "whose tenure is not 0"
        ]

    # Configuration GX en mémoire.
    context = gx.get_context(mode="ephemeral")

    source = context.data_sources.add_pandas(name="telco_source")
    asset = source.add_dataframe_asset(name="telco_data")
    batch_definition = asset.add_batch_definition_whole_dataframe(
        "telco_batch"
    )

    suite = gx.ExpectationSuite(name="telco_validation")

    # Présence des colonnes.
    for column in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=column)
        )

    # Toutes ces colonnes sont obligatoires, sauf l'exception
    # TotalCharges contrôlée ci-dessus.
    for column in required_columns:
        if column != "TotalCharges":
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(
                    column=column
                )
            )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column="customerID"
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotMatchRegex(
            column="customerID",
            regex=r"^\s*$",
        )
    )

    allowed_values = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["Yes", "No"],
    }

    for column, values in allowed_values.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=values,
            )
        )

    # Bornes reprises de ton code : à documenter et à justifier.
    numeric_ranges = {
        "tenure": (0, 120),
        "MonthlyCharges": (0, 200),
        "TotalCharges": (0, None),
    }

    for column, (minimum, maximum) in numeric_ranges.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=minimum,
                max_value=maximum,
            )
        )

    suite = context.suites.add(suite)

    validation = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="telco_validation_run",
            data=batch_definition,
            suite=suite,
        )
    )

    print("Running complete validation suite...")
    results = validation.run(
        batch_parameters={"dataframe": data}
    )

    failed_expectations = []

    for result in results.results:
        if not result.success:
            config = result.expectation_config
            column = config.kwargs.get("column", "dataset")
            failed_expectations.append(
                f"{column}: {config.type}"
            )

    total = len(results.results)
    passed = sum(bool(result.success) for result in results.results)

    print(f"Data validation: {passed}/{total} checks passed")

    if failed_expectations:
        print(f"Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations