
import pytest
import pandas as pd
from survey_results_analyzer import df, clean_df

@pytest.mark.sanity
def test_dataframe_loaded():
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

@pytest.mark.sanity
def test_clean_df_no_missing():
    assert clean_df.isnull().sum().sum() == 0

@pytest.mark.regression
def test_country_column_exists():
    assert 'Country' in df.columns
    assert 'Country' in clean_df.columns

@pytest.mark.regression
def test_top_countries_count():
    top_countries = clean_df['Country'].value_counts().head(10)
    assert len(top_countries) == 10
    assert isinstance(top_countries, pd.Series)

@pytest.mark.regression
def test_language_column_exists():
    assert 'LanguageHaveWorkedWith' in clean_df.columns

@pytest.mark.regression
def test_most_common_languages():
    lang_series = clean_df['LanguageHaveWorkedWith'].str.split(';').explode()
    common_languages = lang_series.value_counts().head(10)
    assert len(common_languages) == 10
    assert isinstance(common_languages, pd.Series)

@pytest.mark.regression
def test_basic_statistics():
    stats = clean_df.describe(include='all')
    assert not stats.empty

@pytest.mark.regression
def test_numeric_columns_for_correlation():
    numeric_cols = clean_df.select_dtypes(include='number').columns
    assert len(numeric_cols) > 0

@pytest.mark.longrun
def test_salary_prediction_possible():
    if 'ConvertedCompYearly' in clean_df.columns:
        assert not clean_df['ConvertedCompYearly'].isnull().all()
        assert len(clean_df['ConvertedCompYearly']) > 10

@pytest.mark.sanity
def test_no_duplicate_columns():
    assert len(clean_df.columns) == len(set(clean_df.columns))

@pytest.mark.regression
def test_all_rows_have_country():
    assert clean_df['Country'].notnull().all()

@pytest.mark.regression
def test_all_rows_have_language():
    assert clean_df['LanguageHaveWorkedWith'].notnull().all()

@pytest.mark.smoke
def test_minimum_row_count():
    assert len(clean_df) > 100

@pytest.mark.smoke
def test_minimum_column_count():
    assert len(clean_df.columns) > 5

@pytest.mark.sanity
def test_salary_column_type():
    if 'ConvertedCompYearly' in clean_df.columns:
        assert pd.api.types.is_numeric_dtype(clean_df['ConvertedCompYearly'])

@pytest.mark.regression
def test_country_value_types():
    assert clean_df['Country'].apply(lambda x: isinstance(x, str)).all()

@pytest.mark.regression
def test_language_value_types():
    assert clean_df['LanguageHaveWorkedWith'].apply(lambda x: isinstance(x, str)).all()

@pytest.mark.longrun
def test_salary_distribution():
    if 'ConvertedCompYearly' in clean_df.columns:
        mean = clean_df['ConvertedCompYearly'].mean()
        std = clean_df['ConvertedCompYearly'].std()
        assert mean > 0
        assert std >= 0
