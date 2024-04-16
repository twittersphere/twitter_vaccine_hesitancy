import pandas as pd
from src.scripts.utils import load_main_config
from src.scripts.incremental_imputer import Imputer
import numpy as np

def load_vaccination_percentage(covid_data_path, main_configs):
    get_state_abbreviations = lambda x: main_configs['us_states_and_abbreviations'].get(x)

    vaccination_percentage = pd.read_parquet(
        f"{covid_data_path}/us_state_vaccinations.parquet")
    vaccination_percentage = vaccination_percentage[['date', 'location', 'people_vaccinated_per_hundred']]
    vaccination_percentage['date'] = pd.to_datetime(vaccination_percentage['date'])
    vaccination_percentage = vaccination_percentage[vaccination_percentage['date'] < '2022-01-01'].reset_index(drop=True)

    vaccination_percentage['location'] = vaccination_percentage['location'].replace({'New York State': 'New York'})
    vaccination_percentage['location'] = vaccination_percentage['location'].apply(get_state_abbreviations)

    return vaccination_percentage

def impute_dfs(incremental_imputer, vaccination_percentage, main_configs):
    imputed_dfs = [incremental_imputer.impute_data(
        vaccination_percentage[vaccination_percentage['location'] == state],
        'people_vaccinated_per_hundred', inplace=False
        ) for state in main_configs['us_51_state']]
    vaccination_percentage = pd.concat(imputed_dfs, axis=0).reset_index(drop=True)

    return vaccination_percentage

def read_fisher_exact(dataframes_path, main_configs):
    fisher_exact_test_results_state = pd.read_parquet(
        f"{dataframes_path}/fisher_exact_test_results_state.parquet")
    
    fisher_exact_test_results_state = fisher_exact_test_results_state[
        fisher_exact_test_results_state['state'].isin(main_configs['us_51_state'])
        ].sort_values('state')

    return fisher_exact_test_results_state

def vaccination_percentage_and_atv(vaccination_percentage, fisher_exact_test_results_state, main_configs):
    vaccination_percentage = []
    for state in sorted(main_configs['us_51_state']):
        vaccination_percentage.append(np.max(
            vaccination_percentage[vaccination_percentage['location'] == state]['people_vaccinated_per_hundred'].values
            ))
    vaccination_percentage = np.array(vaccination_percentage)

    vaccination_percentage = pd.DataFrame({'state':np.array(sorted(main_configs['us_51_state'])),
                                        'vaccination_percentage': vaccination_percentage,
                                        'odd_ratios': fisher_exact_test_results_state['odd_ratios'].values,
                                        'tweet_counts': fisher_exact_test_results_state['tweet_counts'].values})

    return vaccination_percentage

def main():
    dataframes_path = "/data/processed/data_frames"
    incremental_imputer = Imputer()

    main_configs = load_main_config()
    vaccination_percentage = load_vaccination_percentage(
        'data/raw/covid_data', main_configs)
    vaccination_percentage = impute_dfs(incremental_imputer, vaccination_percentage, main_configs)

    fisher_exact_test_results_state = read_fisher_exact(dataframes_path, main_configs)
    vaccination_percentage = vaccination_percentage_and_atv(vaccination_percentage, fisher_exact_test_results_state, main_configs)

    vaccination_percentage.to_parquet(f"{dataframes_path}/correlation_df_VaccinationPercentage_ATV.parquet", index=False)

if __name__ == "__main__":
    main()