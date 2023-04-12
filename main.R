options(scipen = 999)
set.seed(123)

library(glue)
source("models.R")

simulation_name <- "simulation_benchmark"
sim_array <- c(0:99)
n_array <- c(50, 100, 200, 1000, 10000)

for (n in n_array) {
  for (sim in sim_array) {
    df_path <- glue("data/{simulation_name}/n_{n}/sim_{sim}_n_{n}.csv")
    df_results <- glue("raw_results/{simulation_name}/MLE/n_{n}/{simulation_name}_sim_{sim}_n_{n}.csv")
    dir.create(glue("raw_results/{simulation_name}/MLE/n_{n}"), recursive = TRUE, showWarnings = FALSE)
    MLE(df_path, df_results)
  }
}
