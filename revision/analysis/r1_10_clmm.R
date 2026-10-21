suppressMessages(library(ordinal))
d <- read.csv("c1_overall_for_R1-3.csv", stringsAsFactors=FALSE)
for (j in c("claude-opus-4-8","gpt-5.5","gemini-2.5-pro","qwen-3.5")) {
  m <- d[d$judge_id==j,]; m$y <- factor(m$overall_global_quality, ordered=TRUE); m$report <- factor(m$report_id)
  f <- tryCatch(clmm(y ~ 1 + (1|report), data=m, link="logit", nAGQ=10), error=function(e) NULL)
  if (is.null(f)) { cat(j,"ERROR\n"); next }
  s <- tryCatch(summary(f), error=function(e) NULL)
  v <- as.numeric(VarCorr(f)$report); icc <- v/(v+pi^2/3)
  ci <- tryCatch(confint(f, parm="ST", method="Wald"), error=function(e) NA)
  cat(j, "sd=",round(sqrt(v),3),"latentICC=",round(icc,4),"condHess=", if(!is.null(s)) signif(s$condHess,3) else NA, "conv=", f$optRes$convergence, "\n")
}
