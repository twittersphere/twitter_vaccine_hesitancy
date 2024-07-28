import re
import numpy as np
import pandas as pd

from src.scripts.utils import load_main_config


def calculate_perc_and_margin(total, total_margin, value, value_margin):
    perc = value / total
    min_margin = (value - value_margin) / (total + total_margin)
    max_margin = (value + value_margin) / (total - total_margin)

    # , np.abs(perc - min_margin), np.abs(perc - max_margin)
    return perc, min_margin, max_margin


def return_states(main_configs, df):
    new_array = []
    for i in range(df.shape[1]):
        if i % 2 == 0 and i != 0:
            val = df.iloc[0, i-1]
        else:
            val = df.iloc[0, i]
        new_val = main_configs['us_states_and_abbreviations'].get(val, val)
        new_array.append(new_val)

    return new_array


def get_number(value):
    number = "".join(re.findall(r'\d+', value))
    try:
        return int(number)
    except:
        return 0


def get_nonnative(main_configs, socio_economic_params_path):
    nonnative = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/CITIZENSHIP STATUS IN THE UNITED STATES.xlsx").iloc[5:, :].reset_index(drop=True)
    nonnative.iloc[0, :] = return_states(main_configs, nonnative)

    df = []
    for state in sorted(main_configs['us_51_state']):
        state_values = nonnative.loc[:, nonnative.iloc[0, :].values == state]

        total = get_number(state_values.iloc[2, 0])
        total_margin = get_number(state_values.iloc[2, 1])
        value = get_number(state_values.iloc[4, 0])
        value_margin = get_number(state_values.iloc[4, 1])

        df.append(calculate_perc_and_margin(
            total, total_margin, value, value_margin))

    nonnative = pd.DataFrame(
        df, columns=['nonnative_perc', 'nonnative_min_mar', 'nonnative_max_mar'])

    return nonnative


def get_education_plths_pgtbch(main_configs, socio_economic_params_path):
    education = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/EDUCATIONAL ATTAINMENT FOR THE POPULATION 25 YEARS AND OVER.xlsx").iloc[5:, :].reset_index(drop=True)
    education.iloc[0, :] = return_states(main_configs, education)

    df_plths, df_pgtbch = [], []
    for state in sorted(main_configs['us_51_state']):
        state_values = education.loc[:, education.iloc[0, :].values == state]

        total = get_number(state_values.iloc[2, 0])
        total_margin = get_number(state_values.iloc[2, 1])

        plths = np.sum([get_number(state_values.iloc[i, 0])
                        for i in range(3, 6)])
        plths_margin = np.mean(
            [get_number(state_values.iloc[i, 1]) for i in range(3, 6)])

        pgtbch = np.sum([get_number(state_values.iloc[i, 0])
                         for i in range(8, 10)])
        pgtbch_margin = np.mean(
            [get_number(state_values.iloc[i, 1]) for i in range(8, 10)])

        df_plths.append(calculate_perc_and_margin(
            total, total_margin, plths, plths_margin))
        df_pgtbch.append(calculate_perc_and_margin(
            total, total_margin, pgtbch, pgtbch_margin))

    df_plths = pd.DataFrame(
        df_plths, columns=['plths_perc', 'plths_min_mar', 'plths_max_mar'])
    df_pgtbch = pd.DataFrame(
        df_pgtbch, columns=['pgtbch_perc', 'pgtbch_min_mar', 'pgtbch_max_mar'])

    return df_plths, df_pgtbch


def get_unemployment(main_configs, socio_economic_params_path):
    unemployment = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/EMPLOYMENT STATUS FOR THE POPULATION 16 YEARS AND OVER.xlsx").iloc[5:, :].reset_index(drop=True)
    unemployment.iloc[0, :] = return_states(main_configs, unemployment)

    df = []
    for state in sorted(main_configs['us_51_state']):
        state_values = unemployment.loc[:,
                                        unemployment.iloc[0, :].values == state]

        total = get_number(state_values.iloc[4, 0])
        total_margin = get_number(state_values.iloc[4, 1])
        value = get_number(state_values.iloc[6, 0])
        value_margin = get_number(state_values.iloc[6, 1])

        df.append(calculate_perc_and_margin(
            total, total_margin, value, value_margin))

    unemployment = pd.DataFrame(
        df,
        columns=['unemployment_perc', 'unemployment_min_mar',
                 'unemployment_max_mar'])

    return unemployment


def get_median_household_income(main_configs, socio_economic_params_path):
    median_household = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2020 INFLATION-ADJUSTED DOLLARS).xlsx").iloc[5:, :].reset_index(drop=True)
    median_household.iloc[0, :] = return_states(main_configs, median_household)

    df = []
    for state in sorted(main_configs['us_51_state']):
        state_values = median_household.loc[:,
                                            median_household.iloc[0, :].values == state]

        value = get_number(state_values.iloc[2, 0])
        value_margin = get_number(state_values.iloc[2, 1])
        min_margin, max_margin = value - value_margin, value + value_margin

        df.append([value, min_margin, max_margin])

    median_household = pd.DataFrame(
        df,
        columns=['median_household', 'median_household_min_mar',
                 'median_household_max_mar'])

    return median_household


def get_poverty_rate(main_configs, socio_economic_params_path):
    poverty = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/POVERTY STATUS IN THE PAST 12 MONTHS BY AGE.xlsx").iloc[5:, :].reset_index(drop=True)
    poverty.iloc[0, :] = return_states(main_configs, poverty)

    df = []
    for state in sorted(main_configs['us_51_state']):
        state_values = poverty.loc[:, poverty.iloc[0, :].values == state]

        total = get_number(state_values.iloc[2, 0])
        total_margin = get_number(state_values.iloc[2, 1])
        value = get_number(state_values.iloc[3, 0])
        value_margin = get_number(state_values.iloc[3, 1])

        df.append(calculate_perc_and_margin(
            total, total_margin, value, value_margin))

    poverty = pd.DataFrame(
        df, columns=['poverty_perc', 'poverty_min_mar', 'poverty_max_mar'])

    return poverty


def get_black_population(main_configs, socio_economic_params_path):
    black = pd.read_excel(
        f"{socio_economic_params_path}/state_level_data/RACE.xlsx").iloc[5:, :].reset_index(drop=True)
    black.iloc[0, :] = return_states(main_configs, black)

    df = []
    for state in sorted(main_configs['us_51_state']):
        state_values = black.loc[:, black.iloc[0, :].values == state]

        total = get_number(state_values.iloc[2, 0])
        total_margin = get_number(state_values.iloc[2, 1])
        value = get_number(state_values.iloc[4, 0])
        value_margin = get_number(state_values.iloc[4, 1])

        df.append(calculate_perc_and_margin(
            total, total_margin, value, value_margin))

    black = pd.DataFrame(
        df, columns=['black_perc', 'black_min_mar', 'black_max_mar'])

    return black


def get_president_election(main_configs, socio_economic_params_path):
    parties_to_consider = set(['DEMOCRAT', 'REPUBLICAN'])
    president = pd.read_csv(
        f"{socio_economic_params_path}/state_level_data/1976-2020-president.csv")
    president = president[president['year'] == 2020][['state_po',
                                                      'candidatevotes', 'party_simplified']].reset_index(drop=True).rename(
        columns={'state_po': 'state'})
    total_votes = president.drop(columns=['party_simplified']).groupby(
        'state').agg('sum').drop('DC')['candidatevotes'].values
    president = president[president['party_simplified'].isin(
        parties_to_consider) & president['state'].isin(
        main_configs['us_51_state'])]
    president = president[president['party_simplified'] == 'REPUBLICAN'].sort_values('state')['candidatevotes'].values / total_votes - president[president[
        'party_simplified'] == 'DEMOCRAT'].sort_values('state')['candidatevotes'].values / total_votes
    president = pd.DataFrame({'president': president})

    return president


def get_social_capital(main_configs, socio_economic_params_path):
    social_capita = pd.read_csv(
        f"{socio_economic_params_path}/state_level_data/Social Capital Project Social Capital Index Data.xlsx - State Index.csv")
    social_capita.columns = social_capita.iloc[1, :]
    social_capita = social_capita.iloc[2:, :][['State', 'State Abbreviation',
                                               'State-Level Index']].rename(columns={'State-Level Index': 'social_capita'})
    social_capita = social_capita.sort_values('State Abbreviation')[
        social_capita['State Abbreviation'].isin(
            main_configs['us_51_state'])][
        ['social_capita']].reset_index(
        drop=True)
    social_capita['social_capita'] = social_capita['social_capita'].astype(
        np.float16)

    return social_capita


def get_cat_dog_ownership(main_configs, socio_economic_params_path):
    cat_dog_ownership = pd.read_csv(
        f"{socio_economic_params_path}/state_level_data/pet_ownership2.csv",
        delimiter=';')

    cleaned_cat_dog = []
    for i in cat_dog_ownership.columns[1:]:
        sub_data = []
        for j in cat_dog_ownership[i].values:
            if type(j) == str:
                sub_data.append(float(j[:-1]))
            else:
                sub_data.append(j)
        cleaned_cat_dog.append(sub_data)

    cat_dog_ownership.iloc[:, 1:] = np.array(cleaned_cat_dog).T
    cat_dog_ownership['state'] = [
        main_configs['us_states_and_abbreviations'][i]
        for i in cat_dog_ownership['state'].values]
    cat_dog_ownership = cat_dog_ownership.fillna(cat_dog_ownership.median())
    cat_dog_ownership['cat_dog_ratio'] = np.log(
        cat_dog_ownership['cat'].values / cat_dog_ownership['dog'].values)
    cat_dog_ownership = cat_dog_ownership.sort_values(
        'state').reset_index(drop=True)

    return cat_dog_ownership


def main():
    dataframes_path = "data/processed/data_frames"
    socio_economic_params_path = "data/raw/socio_enomic_params"

    fisher_exact_test_results_state = pd.read_parquet(
        f"{dataframes_path}/fisher_exact/state_us_fisher_values.parquet")

    main_configs = load_main_config()
    main_configs['us_51_state'].remove('DC')

    fisher_exact_test_results_state = fisher_exact_test_results_state[fisher_exact_test_results_state['state'].isin(
        main_configs['us_51_state'])].sort_values('state').reset_index(drop=True)

    # Socio-economic params
    nonnative = get_nonnative(main_configs, socio_economic_params_path)
    df_plths, df_pgtbch = get_education_plths_pgtbch(
        main_configs, socio_economic_params_path)
    unemployment = get_unemployment(main_configs, socio_economic_params_path)
    median_household = get_median_household_income(
        main_configs, socio_economic_params_path)
    poverty = get_poverty_rate(main_configs, socio_economic_params_path)
    black = get_black_population(main_configs, socio_economic_params_path)
    president = get_president_election(
        main_configs, socio_economic_params_path)
    social_capita = get_social_capital(
        main_configs, socio_economic_params_path)
    cat_dog_ownership = get_cat_dog_ownership(
        main_configs, socio_economic_params_path)

    # Combine DataFrames
    combined_df = pd.concat(
        [cat_dog_ownership.iloc[:, [0]],
         nonnative.iloc[:, [0]],
         df_plths.iloc[:, [0]],
         df_pgtbch.iloc[:, [0]],
         unemployment.iloc[:, [0]],
         median_household.iloc[:, [0]],
         poverty.iloc[:, [0]],
         black.iloc[:, [0]],
         president, social_capita, cat_dog_ownership.iloc[:, [-1]],
         fisher_exact_test_results_state.iloc[:, [1, -1]]],
        axis=1)

    combined_df.to_parquet(
        f"{dataframes_path}/combined_socio_economic_parameters.parquet",
        index=False)


if __name__ == '__main__':
    main()
