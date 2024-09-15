#AUTOR JACKSON R. SILVA

##################################### PACOTES ##################################
library("caret")
library("mlbench")
library("pROC")
library("rpart")
library("caretEnsemble")
library("data.table")
library("lmtest")
require("nortest")
library("parallel") #PROCESSADORES
require("RWeka") #M5#require("rJava")
library("performance")
require("doParallel")
require("MLmetrics")


################################# data #########################################
data = fread("//home//jackson//Documentos//SensoreR//Art_SensoreR_20.csv", stringsAsFactor=TRUE)
edit(data)
########################### data TREINO E TESTE ################################
#data$D <- data$D / 100 #centímetros para metros
set.seed(100)  
trainIndex  = createDataPartition(y=data$Nome, p=.70, list=FALSE)               #Isso se deve ao fato de que o processo de Repeated K-fold Cross-Validation foi realizado estritamente no subconjunto de treinamento, que abrange 70% do conjunto de dados total. Essa etapa foi executada para identificar os hiperparâmetros ideais dos modelos, os quais foram subsequentemente aplicados ao subconjunto de teste, correspondendo aos 30% restantes do conjunto completo de dados.
trainingSet = data[ trainIndex, ]        
testSet     = data[ -trainIndex, ]
edit(testSet)
################### ATIVANDO NÚCLEOS DO PROCESSADOR ############################ CUIDADO AO UTILIZAR, PARA NÃO SUPERAQUECER O SISTEMA. DEIXEI DEFAULT EM F NO allowParallel
cl <- makeCluster(detectCores() / 2)
registerDoParallel(cl)
################################################################################
# Atualize o controlador de treinamento para problemas multiclasse
my_control <- trainControl(
  method = "repeatedcv",
  number = 4,
  repeats = 4,
  selectionFunction = "best",
  savePredictions = 'all', # all,
  #returnResamp = 'all',
  allowParallel = F,  # Paralelismo desativado temporariamente
  verboseIter = TRUE,
  #index = createResample(trainingSet$Nome, 5), #NULL
  classProbs = TRUE,  # Ativar probabilidades de classe
  summaryFunction = multiClassSummary #  # Usar multiClassSummary para métricas multiclasse
)


# Ajustar modelos e hiperparâmetros para classificação multiclasse
Metodos = c(
  #"naive_bayes",
  "ranger",
  #"glmnet",
  ##"lssvmRadial",
  ##"C5.0",
  "pda",
  "treebag",
  #"nb",
  "LMT",
  "bagEarthGCV",
  #"wsrf",
  #"rFerns",
  #"bagFDA",
  #"lda",
  #"J48",
  #"LogitBoost",
  #"bagFDAGCV",
  "bagEarth",
  #"cforest",
  "parRF",
  "Rborist",
  "RRF",
  "RRFglobal",
  #"xgbLinear",
  "kknn"
  #"stepQDA",
  #"gcvEarth",
  #"earth"
  #"xgbDART",
  #"xgbTree",
  #"gbm"
  #"stepLDA"
) 

# Remova o gaussprPoly por enquanto
Hiper_parametros = list(
  ranger = caretModelSpec(method = 'ranger', num.trees = 500, tuneLength = 5, importance ='permutation')
)

# Lista de modelos para classificação multiclasse
model_list <- list()

for (metodo in Metodos) {
  tryCatch({
    model <- train(
      Nome ~ filtro_passa_alta + kurt + skewness + filtro_passa_faixa + filtro_passa_baixa,
      data = trainingSet,
      method = metodo,
      trControl = my_control,
      tuneLength = 5,
      metric = "F1"
    )
    model_list[[metodo]] <- model
  }, error = function(e) {
    message(paste("O modelo", metodo, "falhou: ", e$message))
  })
}

stopCluster(cl) #DESATIVA MULTI NÚCLEOS 

######################### RESULTADOS DE VALIDAÇÃO ##############################
# Resample os modelos
resamples_results <- resamples(model_list)

# Obter o resumo das métricas de logLoss e Accuracy
summary_logLoss <- summary(resamples_results$values[, grepl("logLoss", names(resamples_results$values))])
summary_accuracy <- summary(resamples_results$values[, grepl("Accuracy", names(resamples_results$values))])

# Exibir os resumos
summary_logLoss
summary_accuracy



# Boxplot para logLoss
boxplot(resamples_results$values[, grepl("logLoss", names(resamples_results$values))], 
        main = "Distribuição de logLoss dos Modelos", 
        ylab = "logLoss",
        las = 2,    # Rotacionar rótulos do eixo x
        col = "lightblue")  # Cor dos boxplots


# Boxplot para Accuracy
boxplot(resamples_results$values[, grepl("Accuracy", names(resamples_results$values))], 
        main = "Distribuição de Accuracy dos Modelos", 
        ylab = "Accuracy",
        las = 3,    # Rotacionar rótulos do eixo x
        col = "lightgreen")  # Cor dos boxplots

# Ajustar os rótulos para melhor visualização
par(mar = c(10, 5, 4, 2))  # Aumenta a margem inferior para acomodar os rótulos
boxplot(resamples_results$values[, grepl("logLoss", names(resamples_results$values))], 
        main = "Distribuição de logLoss dos Modelos", 
        ylab = "logLoss",
        las = 2,    # Rotacionar rótulos do eixo x (90 graus)
        cex.axis = 0.7,  # Diminuir o tamanho dos rótulos do eixo
        col = "lightblue")

# Boxplot para Accuracy com ajuste de rótulos
par(mar = c(10, 5, 4, 2))  # Aumenta a margem inferior
boxplot(resamples_results$values[, grepl("Accuracy", names(resamples_results$values))], 
        main = "Distribuição de Accuracy dos Modelos", 
        ylab = "Accuracy",
        las = 2,    # Rotacionar rótulos do eixo x
        cex.axis = 0.7,  # Diminuir o tamanho dos rótulos do eixo
        col = "lightgreen")

# Ajustar os rótulos para melhor visualização e boxplot na horizontal
par(mar = c(5, 12, 4, 2))  # Aumenta a margem lateral esquerda para acomodar os rótulos
boxplot(resamples_results$values[, grepl("logLoss", names(resamples_results$values))], 
        xlab = "logLoss",
        horizontal = TRUE,  # Boxplot na horizontal
        las = 1,    # Deixar os rótulos na horizontal (para o eixo y neste caso)
        cex.axis = 0.7,  # Diminuir o tamanho dos rótulos
        col = "lightblue")


# Boxplot para Accuracy com ajuste de rótulos e horizontal
par(mar = c(5, 12, 4, 2))  # Aumenta a margem lateral esquerda
boxplot(resamples_results$values[, grepl("Accuracy", names(resamples_results$values))], 
        xlab = "Accuracy",
        horizontal = TRUE,  # Boxplot na horizontal
        las = 1,    # Deixar os rótulos na horizontal (para o eixo y neste caso)
        cex.axis = 0.7,  # Diminuir o tamanho dos rótulos
        col = "lightgreen") #lightgreen


#######################################

# Prever no conjunto de teste para LMT
predictions_lmt <- predict(model_list$LMT, testSet)

# Prever no conjunto de teste para PDA
predictions_pda <- predict(model_list$pda, testSet)

# Matriz de confusão para LMT
conf_matrix_lmt <- confusionMatrix(predictions_lmt, testSet$Nome)

# Matriz de confusão para PDA
conf_matrix_pda <- confusionMatrix(predictions_pda, testSet$Nome)

# Verificar a matriz de confusão
print(conf_matrix_lmt)
print(conf_matrix_pda)

# Transformar a matriz de confusão para formato longo para LMT
conf_matrix_lmt_table <- as.table(conf_matrix_lmt$table)
conf_matrix_lmt_melted <- melt(conf_matrix_lmt_table, varnames = c("Real", "Predito"))

# Plotar a matriz de confusão para o modelo LMT
ggplot(data = conf_matrix_lmt_melted, aes(x = Predito, y = Real, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = value), vjust = 1, color = "black") +
  labs(x = "Predito (LMT)", y = "Real", fill = "Contagem") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Transformar a matriz de confusão para formato longo para PDA
conf_matrix_pda_table <- as.table(conf_matrix_pda$table)
conf_matrix_pda_melted <- melt(conf_matrix_pda_table, varnames = c("Real", "Predito"))

# Plotar a matriz de confusão para o modelo PDA
ggplot(data = conf_matrix_pda_melted, aes(x = Predito, y = Real, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "steelblue") +
  geom_text(aes(label = value), vjust = 1, color = "black") +
  labs(x = "Predito (PDA)", y = "Real", fill = "Contagem") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))