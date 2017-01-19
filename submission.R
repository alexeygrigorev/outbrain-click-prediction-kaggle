# usage:
# Rscript submission.R final_submission.feather xgb_submission.csv

library(methods)
library(data.table)
library(feather)


input_cmd_args = commandArgs(trailingOnly = TRUE)

path = path.expand(input_cmd_args[[1]])

out_path = path.expand(input_cmd_args[[2]])
out_path = paste0(out_path, ".gz")



message(Sys.time(), " reading ", path)

dt = read_feather(path)
setDT(dt)
dt[, p_neg:=-pred]

message(Sys.time(), " sorting")
setkey(dt, display_id, p_neg)


message(Sys.time(), " generating submission")

submission = dt[ , .(ad_id = paste(ad_id, collapse = " ")), keyby = display_id]

write.table(submission, file = gzfile(out_path, compression = 1), row.names = F, quote = F, sep = ",", append = F)

message(Sys.time(), " DONE")