library(sjemea)


x <- scan("/Users/maxwuerfek/code/diss/x.csv", sep='', what=numeric())
y <- scan("/Users/maxwuerfek/code/diss/y.csv", sep='', what=numeric())


tiling.corr(x, y, dt = 0.05, rec.time = c(0, 912))
