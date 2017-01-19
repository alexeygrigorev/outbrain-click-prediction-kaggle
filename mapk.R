# usage
# Rscript mapk.R pred.feather

library(methods)
library(data.table)
library(feather)

input_cmd_args = commandArgs(trailingOnly = TRUE)
path = path.expand(input_cmd_args[[1]])

message(Sys.time(), " reading ", path)
dt = read_feather(path)
setDT(dt)
dt[, p_neg:=-pred]
message(Sys.time(), " sorting")
setkey(dt, display_id, p_neg)
message(Sys.time(), " calculating map...")
map = dt[ , .(map_12 = 1 / which(clicked == 1)), by = display_id][['map_12']]
message(Sys.time(), " MAP@12 = ", mean(map))