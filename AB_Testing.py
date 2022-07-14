#################################################################
#AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#################################################################

#############################
#İş Problemi
#############################

#Facebook kısa süre önce mevcut "maximum bidding" adı verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan "average bidding"’i tanıttı.
#Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test etmeye karar verdi ve average bidding'in maximum bidding'den daha fazla dönüşüm getirip
#getirmediğini anlamak için bir A/B testi yapmak istiyor. A/B testi 1 aydır devam ediyor ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.
#Bombabomba.com için nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.

##############################
#Veri Seti Hikayesi
##############################

#Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.
# Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna Maximum Bidding,
# test grubuna Average Bidding uygulanmıştır.

#Impression : Reklam görüntüleme sayısı
#Click : Görüntülenen reklama tıklama sayısı
#Purchase : Tıklanan reklamlar sonrası satın alınan ürün sayısı
#Earning : Satın alınan ürünler sonrası elde edilen kazanç

#############################################
#Görev 1: Veriyi Hazırlama ve Analiz Etme
#############################################
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore",category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#########
#Adım 1: ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
#########
Control_Group = pd.read_excel("C:/Users/ASUS/PycharmProjects/ödev/AB_Testi/ab_testing.xlsx", sheet_name='Control Group')  # maximum bidding
Test_Group = pd.read_excel("C:/Users/ASUS/PycharmProjects/ödev/AB_Testi/ab_testing.xlsx", sheet_name='Test Group')        # average bidding

########
#Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
########
Control_Group.head()
Control_Group.shape
Control_Group.describe().T
Control_Group.isnull().sum()
Test_Group.head()
Test_Group.shape
Test_Group.describe().T
Test_Group.isnull().sum()

########
#Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
########

Control_Group["Group"] = "Control_Group"
Test_Group["Group"] = "Test_Group"
df = pd.concat([Control_Group,Test_Group],axis=1)
df.head()

####################################################
#Görev 2: A/B Testinin Hipotezinin Tanımlanması
####################################################

##############
#Adım 1: Hipotezi tanımlayınız.
#H0 : M1 = M2  #istatistiksel olarak bir farklılık yoktur.
#H1 : M1!= M2  #istatistiksel olarak bir farklılık vardır.
##############

########
#Adım 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarını analiz ediniz.
########

Control_Group["Purchase"].mean() #550.894058
Test_Group["Purchase"].mean() #582.106096

###################################################
#Görev 3: Hipotez Testinin Gerçekleştirilmesi
###################################################

#########
#Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
#Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.
##########
#Normallik Varsayımı :
#H0: Normal dağılım varsayımı sağlanmaktadır.
#H1: Normal dağılım varsayımı sağlanmamaktadır.
#p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
#Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.
#Varyans Homojenliği :
#H0: Varyanslar homojendir.
#H1: Varyanslar homojen Değildir.
#p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
#Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
#Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = shapiro(Control_Group['Purchase'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9773, p-value = 0.5891

test_stat, pvalue = shapiro(Test_Group['Purchase'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9589, p-value = 0.1541
## p_value >0.05 olduğunda Ho reddedilmez, yani normallik varsayımı sağlanmaktadır.

#########
#Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
#########

# Varyans Homojenligi Varsayımı

# H0: Varyanslar Homojendir.
# H1: Varyanslar Homojen Değildir.

test_stat, pvalue = levene((Control_Group["Purchase"]), (Test_Group["Purchase"]))
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 2.6393, p-value = 0.1083 , Varyanslar Homojendir.

#########
#Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız
#########

test_stat, pvalue = ttest_ind((Control_Group["Purchase"]), (Test_Group["Purchase"]),
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = -0.9416, p-value = 0.3493
#HO Reddedilemez.
# H0 : M1 = M2
#Kontrol ve Test verisi arasında istatistiksel olarak anlamlı bir farklılık yoktur.

##############################
#Görev 4: Sonuçların Analizi
##############################
#Bağımsız t-testi kullandık. Çünkü,
#Bağımsız örneklem t testi, iki bağımsız grup arasında ortalamalara bakarak istatistiksel olarak anlamlı bir fark olup olmadığını anlamamıza yardımcı olur.
#Sonuç: Kontrol ve Test verisi arasında istatistiksel olarak anlamlı bir farklılık yoktur.
