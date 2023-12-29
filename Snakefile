include: "src/sentiment_analysis/Snakefile"
include: "src/table_scripts/Snakefile"
include: "src/df_scripts/Snakefile"
include: "src/fetching_data/Snakefile"
include: "src/figures/Snakefile"

rule all:
    input:
        rules.fetching_data_all.input,
        rules.df_scripts_all.input,
        rules.sentiment_analysis_all.input,
        rules.table_scripts_all.input,
        rules.figures_all.input
    default_target: True
