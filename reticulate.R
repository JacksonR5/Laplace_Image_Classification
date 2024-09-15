#install.packages("reticulate")
require("reticulate")

reticulate::py_install("scikit-learn")
reticulate::py_install("openpyxl")
reticulate::py_install("statsmodels")
reticulate::py_install("matplotlib")
reticulate::py_install("seaborn")
reticulate::py_install("sympy")
reticulate::py_install("opencv-python")

reticulate::py_install("pygbif")