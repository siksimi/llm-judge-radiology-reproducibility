# R1-3 ordinal analysis per SAP_appendix_R1-3_ordinal_v1.md
suppressMessages(library(ordinal))
args <- commandArgs(TRUE); MODE <- if (length(args)) args[1] else "low"
MEDFUN <- if (MODE == "high") function(x) { x <- sort(x); x[floor(length(x)/2)+1] } else function(x) { x <- sort(x); x[ceiling(length(x)/2)] }
OUTP <- paste0("out/", MODE, "_")
d <- read.csv("c1_overall_for_R1-3.csv", stringsAsFactors = FALSE)
d <- d[d$parse_ok %in% c(TRUE, "True", "TRUE"), ]
d$cell <- ifelse(d$group3 == "original", "original",
          ifelse(d$actual_text_changed %in% c(TRUE,"True","TRUE") & d$assigned_family %in% c("finding_omission","hallucinated_finding","readability"),
                 paste0(d$assigned_family, "_i", d$nominal_intensity), NA))
lv <- c("original", paste0(rep(c("finding_omission","hallucinated_finding","readability"), each = 3), "_i", 1:3))
judges <- c("claude-opus-4-8","gpt-5.5","gemini-2.5-pro","qwen-3.5")
# descriptive (all cells incl laterality)
d$cell_desc <- ifelse(is.na(d$cell), ifelse(d$assigned_family == "laterality_measure",
                     ifelse(d$actual_text_changed %in% c(TRUE,"True","TRUE"), "laterality_swap", "laterality_unchanged"), "other"), d$cell)
rep_means <- aggregate(overall_global_quality ~ judge_id + report_id + cell_desc, data = d, FUN = mean)
desc <- aggregate(overall_global_quality ~ judge_id + cell_desc, data = rep_means, FUN = function(x) c(mean = mean(x), n = length(x)))
desc <- do.call(data.frame, desc); names(desc) <- c("judge","cell","mean_score","n_reports")
write.csv(desc, paste0(OUTP,"r1_3_descriptive.csv"), row.names = FALSE)
coefs <- list(); contr <- list(); info <- list()
for (j in judges) {
  m <- d[d$judge_id == j & !is.na(d$cell), ]
  m$cell <- factor(m$cell, levels = lv); m$y <- factor(m$overall_global_quality, ordered = TRUE)
  m$report <- factor(m$report_id)
  method <- "clmm"; fit <- NULL; note <- ""
  within_var <- tapply(m$overall_global_quality, m$report, var)
  if (all(within_var == 0, na.rm = TRUE)) { method <- "clm_report_level"; note <- "no within-report variation" }
  if (method == "clmm") {
    fit <- tryCatch(clmm(y ~ cell + (1 | report), data = m, link = "logit"), error = function(e) NULL)
    ok <- !is.null(fit)
    if (ok) { s <- tryCatch(summary(fit), error=function(e) NULL); cn <- if (is.null(s)) Inf else s$condHess; se <- tryCatch(sqrt(diag(vcov(fit))), error=function(e) NA)
      ok <- is.finite(cn) && cn <= 1e6 && all(is.finite(se)) && (is.null(fit$optRes$convergence) || fit$optRes$convergence == 0)
      if (!ok) note <- paste0("clmm failed criteria (condHess=", signif(cn,3), ")") }
    else note <- "clmm error"
    if (!ok) { method <- "clm_report_level"; fit <- NULL }
  }
  if (method == "clm_report_level") {
    r <- aggregate(overall_global_quality ~ report + cell, data = m, FUN = MEDFUN)
    r$y <- factor(r$overall_global_quality, ordered = TRUE); r$cell <- factor(r$cell, levels = lv)
    fit <- clm(y ~ cell, data = r, link = "logit")
  }
  b <- coef(fit); V <- vcov(fit); nm <- grep("^cell", names(b), value = TRUE)
  se <- sqrt(diag(V))[nm]
  tab <- table(m$cell, m$y)
  for (k in nm) {
    lvl <- sub("^cell", "", k); row <- tab[lvl, ]; sep <- sum(row > 0) <= 1 || !is.finite(se[k]) || se[k] > 50
    coefs[[length(coefs)+1]] <- data.frame(judge = j, method = method, cell = lvl, beta = b[k], se = se[k],
        lo = b[k] - 1.959964*se[k], hi = b[k] + 1.959964*se[k], separation_flag = sep)
  }
  for (f in c("finding_omission","hallucinated_finding","readability")) {
    k <- paste0("cell", f, "_i", 1:3)
    for (pr in list(c(2,1), c(3,2), c(3,1))) {
      a <- k[pr[1]]; c0 <- k[pr[2]]; est <- b[a] - b[c0]; v <- V[a,a] + V[c0,c0] - 2*V[a,c0]
      contr[[length(contr)+1]] <- data.frame(judge = j, method = method, family = f, contrast = paste0("i", pr[1], "-i", pr[2]),
          estimate = est, lo = est - 1.959964*sqrt(v), hi = est + 1.959964*sqrt(v))
    }
    bb <- b[k]; info_mono <- all(diff(bb) <= 0)
    info[[length(info)+1]] <- data.frame(judge = j, method = method, family = f, monotone_decline_point = info_mono,
        beta_i1 = bb[1], beta_i2 = bb[2], beta_i3 = bb[3])
  }
  ranef_sd <- if (inherits(fit, "clmm")) sqrt(as.numeric(VarCorr(fit)$report)) else NA
  info[[length(info)+1]] <- data.frame(judge = j, method = method, family = "MODEL", monotone_decline_point = NA,
      beta_i1 = ranef_sd, beta_i2 = if (inherits(fit,"clmm")) summary(fit)$condHess else NA, beta_i3 = nrow(m))
  cat(j, method, note, "\n")
}
write.csv(do.call(rbind, coefs), paste0(OUTP,"r1_3_coefficients.csv"), row.names = FALSE)
write.csv(do.call(rbind, contr), paste0(OUTP,"r1_3_intensity_contrasts.csv"), row.names = FALSE)
write.csv(do.call(rbind, info), paste0(OUTP,"r1_3_monotonicity_and_model.csv"), row.names = FALSE)
cat("R", R.version$major, R.version$minor, "ordinal", as.character(packageVersion("ordinal")), "\n")
