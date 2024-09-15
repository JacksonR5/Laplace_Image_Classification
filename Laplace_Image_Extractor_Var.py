import cv2
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats import kurtosis, skew


# Função para criar uma máscara passa-baixa
def low_pass_mask(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    return mask

# Função para criar uma máscara passa-alta
def high_pass_mask(shape, cutoff):
    return 1 - low_pass_mask(shape, cutoff)

# Função para criar uma máscara passa-faixa
def band_pass_mask(shape, low_cutoff, high_cutoff):
    return low_pass_mask(shape, high_cutoff) - low_pass_mask(shape, low_cutoff)

# Função para extrair características usando transformada de Fourier
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fft_result = np.fft.fft2(img)
    spectrum = np.abs(fft_result)
    
    # Extrai características adicionais do espectro de frequência
    mean_freq = np.mean(spectrum)
    std_freq = np.std(spectrum)
    max_freq = np.argmax(spectrum)
    total_energy = np.sum(spectrum**2)  # Energia total do espectro de frequência
    entropy = -np.sum(spectrum * np.log(spectrum + 1e-8))  # Entropia da imagem
    
    # Aplicação das máscaras
    low_pass = spectrum * low_pass_mask(spectrum.shape, 30)  # Cutoff arbitrário
    high_pass = spectrum * high_pass_mask(spectrum.shape, 30)  # Cutoff arbitrário
    band_pass = spectrum * band_pass_mask(spectrum.shape, 10, 50)  # Low e high cutoff arbitrários
    
    # Extração das novas variáveis
    filtro_passa_baixa = np.sum(low_pass)
    filtro_passa_alta = np.sum(high_pass)
    filtro_passa_faixa = np.sum(band_pass)
    
    kurt = kurtosis(spectrum.flatten())
    skewness = skew(spectrum.flatten())
    
    return mean_freq, std_freq, max_freq, total_energy, entropy, filtro_passa_baixa, filtro_passa_alta, filtro_passa_faixa, kurt, skewness

# Definindo diretórios contendo imagens de cada classe
dir_1 = "/home/jackson/Imagens/COPAS/1"
dir_2 = "/home/jackson/Imagens/COPAS/2"
dir_3 = "/home/jackson/Imagens/COPAS/3"
dir_4 = "/home/jackson/Imagens/COPAS/4"
dir_5 = "/home/jackson/Imagens/COPAS/5"
dir_6 = "/home/jackson/Imagens/COPAS/6"
dir_7 = "/home/jackson/Imagens/COPAS/7"
dir_8 = "/home/jackson/Imagens/COPAS/8"
dir_9 = "/home/jackson/Imagens/COPAS/9"
dir_10 = "/home/jackson/Imagens/COPAS/10"
dir_11 = "/home/jackson/Imagens/COPAS/11"
dir_12  = "/home/jackson/Imagens/COPAS/12"
dir_13 = "/home/jackson/Imagens/COPAS/13"
dir_14 = "/home/jackson/Imagens/COPAS/11"
dir_15 = "/home/jackson/Imagens/COPAS/15"
dir_16 = "/home/jackson/Imagens/COPAS/16"
dir_17 = "/home/jackson/Imagens/COPAS/17"
dir_18 = "/home/jackson/Imagens/COPAS/18"
dir_19 = "/home/jackson/Imagens/COPAS/19"
dir_20 = "/home/jackson/Imagens/COPAS/20"
dir_22
dir_23
dir_24
dir_25
dir_26
dir_27
dir_28
dir_29
dir_30
dir_31
dir_32
dir_33
dir_34
dir_35
dir_36
dir_37
dir_38
dir_39
dir_40
dir_41
dir_42
dir_43
dir_44
dir_45
dir_46
dir_47
dir_48
dir_49
dir_50
dir_51
dir_52
dir_53
dir_54
dir_55
dir_56
dir_57
dir_58
# Obtendo caminhos completos de todas as imagens de cada classe
images_1 = [f"{dir_1}/{img}" for img in os.listdir(dir_1)]
images_2 = [f"{dir_2}/{img}" for img in os.listdir(dir_2)]
images_3 = [f"{dir_3}/{img}" for img in os.listdir(dir_3)]
images_4 = [f"{dir_4}/{img}" for img in os.listdir(dir_4)]
images_5 = [f"{dir_5}/{img}" for img in os.listdir(dir_5)]
images_6 = [f"{dir_6}/{img}" for img in os.listdir(dir_6)]
images_7 = [f"{dir_7}/{img}" for img in os.listdir(dir_7)]
images_8 = [f"{dir_8}/{img}" for img in os.listdir(dir_8)]
images_9 = [f"{dir_9}/{img}" for img in os.listdir(dir_9)]
images_10 = [f"{dir_10}/{img}" for img in os.listdir(dir_10)]
images_11 = [f"{dir_11}/{img}" for img in os.listdir(dir_11)]
images_12 = [f"{dir_12}/{img}" for img in os.listdir(dir_12)]
images_13 = [f"{dir_13}/{img}" for img in os.listdir(dir_13)]
images_14 = [f"{dir_14}/{img}" for img in os.listdir(dir_14)]
images_15 = [f"{dir_15}/{img}" for img in os.listdir(dir_15)]
images_16 = [f"{dir_16}/{img}" for img in os.listdir(dir_16)]
images_17 = [f"{dir_17}/{img}" for img in os.listdir(dir_17)]
images_18 = [f"{dir_18}/{img}" for img in os.listdir(dir_18)]
images_19 = [f"{dir_19}/{img}" for img in os.listdir(dir_19)]
images_20 = [f"{dir_20}/{img}" for img in os.listdir(dir_20)]

# Criando lista com todos os caminhos de imagens
images_list = images_1 + images_2 + images_3 + images_4 + images_5 + images_6 + images_7 + images_8 + images_9 + images_10 + images_11 + images_12 + images_13 + images_14 + images_15 + images_16 + images_17 + images_18 + images_19 + images_20

# Extraindo características de todas as imagens de cada classe
features = [extract_features(img) for img in images_list]

# Criando DataFrame com as características extraídas
data = pd.DataFrame({
    'mean_freq': [f[0] for f in features],
    'std_freq': [f[1] for f in features],
    'max_freq': [f[2] for f in features],
    'total_energy': [f[3] for f in features],
    'entropy': [f[4] for f in features],
    'filtro_passa_baixa': [f[5] for f in features],
    'filtro_passa_alta': [f[6] for f in features],
    'filtro_passa_faixa': [f[7] for f in features],
    'skewness': [f[8] for f in features],
    'kurt': [f[9] for f in features],
    'class': (
        ['1'] * len(images_1) + 
        ['2'] * len(images_2) + 
        ['3'] * len(images_3) + 
        ['4'] * len(images_4) + 
        ['5'] * len(images_5) + 
        ['6'] * len(images_6) + 
        ['7'] * len(images_7) + 
        ['8'] * len(images_8) + 
        ['9'] * len(images_9) + 
        ['10'] * len(images_10) + 
        ['11'] * len(images_11) + 
        ['12'] * len(images_12) + 
        ['13'] * len(images_13) + 
        ['14'] * len(images_14) + 
        ['15'] * len(images_15) + 
        ['16'] * len(images_16) + 
        ['17'] * len(images_17) + 
        ['18'] * len(images_18) + 
        ['19'] * len(images_19) + 
        ['20'] * len(images_20)
    )
})

# Salvando os dados em um arquivo CSV
data.to_csv('Art_SensoreR_20.csv', index=True)
