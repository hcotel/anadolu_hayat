import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import catboost as cat
import matplotlib as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from scipy.special import inv_boxcox
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, VotingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import sklearn.tree as sktree
from sklearn.svm import SVC
from mlxtend.classifier import StackingCVClassifier, StackingClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn import tree
from dtreeviz.trees import dtreeviz
import optuna
from lofo import LOFOImportance, Dataset, plot_importance
from tqdm import tqdm
from rgf import RGFClassifier
from lgbm_imputer import LGBMImputer
from constants import (
    run_lofo,
    run_optuna,
    run_city,
    run_adv,
    use_nfolds,
    trial,
    validation_strategy,
    validation_files_index,
    check_val_results,
    run_imputer,
    use_scaler,
    optuna_trials,
    plot_importance,
    model_type,
    isolation_forest,
)
from fix_city import CityFix
import warnings
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None
warnings.filterwarnings(action="ignore")
from utils import downcast_df_int_columns, downcast_df_float_columns, plot_confusion_matrix

sample_submission = pd.read_csv("data/samplesubmission.csv")
test = pd.read_csv("data/test-utf8.csv")
train = pd.read_csv("data/train-utf8.csv")

all = pd.concat([train, test], axis=0)
all = all.rename(columns={'SUBAT_ODENEN_TU': 'SUBAT_ODENEN_TUTAR'})
vade_cols = all.columns[all.columns.str.contains("VADE")].tolist()
odenen_cols = all.columns[all.columns.str.contains("ODENEN")].tolist()
all.replace('nan', np.nan, inplace=True)
all.replace('NaN', np.nan, inplace=True)
if run_city:
    cityFix = CityFix()
    all = cityFix.fix_sehir(all)
    all.to_csv("data/all.csv", sep=",", index=True)
else:
    all = pd.read_csv("data/all.csv")
all.loc[all["POLICE_SEHIR"].isna(), "POLICE_COUNTRY"] = np.nan
all["POLICE_SEHIR"] = all["POLICE_SEHIR"].str.replace('İ','I').str.lower()
all["POLICE_SEHIR"] = all["POLICE_SEHIR"].str.replace(" ilçesi", "")

all.loc[all["KAPSAM_TIPI"] == "PENSION381", "KAPSAM_TIPI"] = "PENSION247"
all.loc[all["KAPSAM_TIPI"] == "PENSION260", "KAPSAM_TIPI"] = "PENSION305"
all.loc[all["KAPSAM_TIPI"] == "PENSION343", "KAPSAM_TIPI"] = "PENSION351"

#all = all[~all['ARALIK_VADE_TUTARI'].isin([73.0, 216.37, 88.0, 150.7, 241.85, 198.0, 131, 230.79]) | all["ARTIS_DURUMU"].isna()]

# set up the threshold percent
threshold_percent = 0.01




series = pd.value_counts(all['OFFICE_ID'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(OFFICE_ID = np.where(all['OFFICE_ID'].isin(series[mask].index), 'Other', all['OFFICE_ID']))

series = pd.value_counts(all['POLICE_SEHIR'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(POLICE_SEHIR = np.where(all['POLICE_SEHIR'].isin(series[mask].index), 'Other', all['POLICE_SEHIR']))

series = pd.value_counts(all['POLICE_COUNTRY'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(POLICE_COUNTRY = np.where(all['POLICE_COUNTRY'].isin(series[mask].index), 'Other', all['POLICE_COUNTRY']))

series = pd.value_counts(all['KAPSAM_TIPI'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(KAPSAM_TIPI = np.where(all['KAPSAM_TIPI'].isin(series[mask].index), 'Other', all['KAPSAM_TIPI']))

all["meslek-krlm"] = all['MESLEK'] + '_' + all['MESLEK_KIRILIM']

target_encoding_features = [
    "odeme_karakteri",
    "vade_artirma_karakteri",
    "vade_azaltma_karakteri",
    "odeme_karakteri_q",
    "vade_artirma_karakteri_q",
    "vade_big_artirma_karakteri_q",
    "vade_azaltma_karakteri_q",
    "vade_group",
    "odeme_group",
    'OFFICE_ID',
    'SIGORTA_TIP',
    'SOZLESME_KOKENI',
    #'SOZLESME_KOKENI_DETAY',
    # "sb_0",
    # "ss_0",
    'KAPSAM_TIPI',
    'KAPSAM_GRUBU',
    'DAGITIM_KANALI',
    'POLICE_SEHIR',
    "POLICE_COUNTRY",
    'CINSIYET',
    #'UYRUK',
    #'MEMLEKET',
    'MESLEK',
    'MESLEK_KIRILIM',
    "meslek-krlm",
    #"meslek_is_bill",
    'MUSTERI_SEGMENTI',
    'YATIRIM_KARAKTERI',
    #"vade_kusuratli",
    #"yatirim_k_null",
    'MEDENI_HAL',
    'EGITIM_DURUM'
    #"under_months_all_year",
    #"zero_months_all_year",
    #"well_paid_customer",
    #"vade_up",
    #"vade_down",
    #"vade_not_changed",
    #"up_first_quarter",
]

all["vade_group"] = all.groupby(vade_cols).ngroup()
all["odeme_group"] = all.groupby(odenen_cols).ngroup()
all["vade_nunique"] = all[vade_cols].nunique(axis=1)

odenen_df = all[odenen_cols]
vade_df = all[vade_cols]

all["vade_mean"] = vade_df.mean(axis=1)
all["vade_std"] = vade_df.std(axis=1)
all["vade_median"] = vade_df.median(axis=1)
all["vade_min"] = vade_df.min(axis=1)
all["vade_max"] = vade_df.max(axis=1)
all["vade_skew"] = vade_df.skew(axis=1)
all["vade_pct_change_mean"] = vade_df.pct_change(axis=1).mean(axis=1).fillna(0)
all["vade_pct_change_max"] = vade_df.pct_change(axis=1).max(axis=1).fillna(0)
all["vade_pct_change_min"] = vade_df.pct_change(axis=1).min(axis=1).fillna(0)

all["odeme_mean"] = odenen_df.mean(axis=1)
all["odeme_std"] = odenen_df.std(axis=1)
all["odeme_median"] = odenen_df.median(axis=1)
all["odeme_min"] = odenen_df.min(axis=1)
all["odeme_max"] = odenen_df.max(axis=1)
all["odeme_skew"] = odenen_df.skew(axis=1)
all["odenen_pct_change_mean"] = odenen_df.pct_change(axis=1).mean(axis=1).fillna(0)
all["odenen_pct_change_max"] = odenen_df.pct_change(axis=1).max(axis=1).fillna(0)
all["odenen_pct_change_min"] = odenen_df.pct_change(axis=1).min(axis=1).fillna(0)

threshold_percent = 0.01

all["yerli_milli"] = (all["UYRUK"] == "TR").astype(int)

all['GELIR'] = pd.to_numeric(all['GELIR'], errors='coerce').astype(float)
all.loc[all['GELIR'] == 9999999999, 'GELIR'] = np.nan
all.loc[all['GELIR'] == 999999999, 'GELIR'] = np.nan
all.loc[all['GELIR'] == 99999999, 'GELIR'] = np.nan
all.loc[all['GELIR'] == 111111111, 'GELIR'] = np.nan
all.loc[all['GELIR'] == 50005000, 'GELIR'] = 5000
all.loc[all['GELIR'] == 30003000, 'GELIR'] = 3000
all.loc[all['GELIR'] == 10001000, 'GELIR'] = 1000
all.loc[all['GELIR'] == 20002000, 'GELIR'] = 2000
all.loc[all['GELIR'] == 25002500, 'GELIR'] = 2500
all.loc[all['GELIR'] == 100010000, 'GELIR'] = 1000
all.loc[all['GELIR'] == 18001800, 'GELIR'] = 1800
all.loc[all['GELIR'] == 13001300, 'GELIR'] = 1300
all.loc[all['GELIR'] == 12001200, 'GELIR'] = 1200
all.loc[all['GELIR'] == 10002000, 'GELIR'] = np.nan
all.loc[all['GELIR'] == 32502000, 'GELIR'] = 3250
all.loc[all['GELIR'] ==200000000001500, 'GELIR'] = 1500
all.loc[all['GELIR'] ==3000000000, 'GELIR'] = 3000
all.loc[all['GELIR'] ==3000000000, 'GELIR'] = 3000
all.loc[all['GELIR'] ==500000000, 'GELIR'] = 5000
all.loc[all['GELIR'] < 0, 'GELIR'] = np.nan
all.loc[(all['GELIR'] >= 100000) & (all['GELIR'] <= 50000000), 'GELIR'] = all['GELIR'] / 1000
all.loc[all['GELIR'] >= 100000, 'GELIR'] = np.nan
all.loc[(all['GELIR'] == 0) & (all['KAPSAM_GRUBU'] != "EV HANIMI") & (all['KAPSAM_GRUBU'] != "GENÇ"), 'GELIR'] = np.nan
all.loc[(all['GELIR'] < 100) & (all['GELIR'] > 0), "GELIR"] = np.nan
all["BASLANGIC_TARIHI"] = pd.to_datetime(all["BASLANGIC_TARIHI"])
all["BASLANGIC_ay"] = all["BASLANGIC_TARIHI"].dt.month
all["BASLANGIC_yil"] = all["BASLANGIC_TARIHI"].dt.year
all["month_passed"] = ((pd.to_datetime("2021-01-01") - all["BASLANGIC_TARIHI"]).dt.days / 30).round()
all["year_passed"] = 2021 - all["BASLANGIC_yil"]
all["YAS"] = 2021 - all["DOGUM_TARIHI"]
all["BASLANGIC_YAS"] = all["YAS"] - all["month_passed"] / 12.0
#all[['MESLEK', 'MESLEK_KIRILIM', 'EGITIM_DURUM', 'KAPSAM_GRUBU']] = all[['MESLEK', 'MESLEK_KIRILIM', 'EGITIM_DURUM', 'KAPSAM_GRUBU']].fillna("NAN")



# all.loc[all['EGITIM_DURUM'] == "(Diğer)", 'EGITIM_DURUM'] = np.nan
egitim_mods = all.groupby('MESLEK_KIRILIM')['EGITIM_DURUM'].agg(pd.Series.mode).reset_index().rename(columns={'EGITIM_DURUM': 'egitim_mod'})
all = all.merge(egitim_mods, how='left', on='MESLEK_KIRILIM')
all.loc[all['EGITIM_DURUM'].isna(), 'EGITIM_DURUM'] = all['egitim_mod']
all.loc[all['EGITIM_DURUM'].isna(), 'EGITIM_DURUM'] = all['EGITIM_DURUM'].agg(pd.Series.mode).iloc[0]

all.loc[all['COCUK_SAYISI'].isna(), 'COCUK_SAYISI'] = 0
all.loc[all['MEDENI_HAL'].isna() & (all['COCUK_SAYISI'] > 0), 'MEDENI_HAL'] = "Married"
all.loc[all['MEDENI_HAL'].isna() & (all['COCUK_SAYISI'] == 0), 'MEDENI_HAL'] = "Single"

all.loc[all['DAGITIM_KANALI'].isna(), 'DAGITIM_KANALI'] = all['DAGITIM_KANALI'].agg(pd.Series.mode).iloc[0]
all.loc[all['MUSTERI_SEGMENTI'].isna(), 'MUSTERI_SEGMENTI'] = all['MUSTERI_SEGMENTI'].agg(pd.Series.mode).iloc[0]
all.loc[all['UYRUK'].isna(), 'UYRUK'] = all['UYRUK'].agg(pd.Series.mode).iloc[0]
all.loc[all['POLICE_SEHIR'].isna(), 'POLICE_SEHIR'] = all['POLICE_SEHIR'].agg(pd.Series.mode).iloc[0]

all.loc[all['SOZLESME_KOKENI'] == "TRANS", 'SOZLESME_KOKENI'] = "TRANS_C"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TRANS", 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'].isna(), 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TR_T2_TR", 'SOZLESME_KOKENI_DETAY'] = "TRANS_T2"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "INV_PROC", 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TRANS_C", 'SOZLESME_KOKENI_DETAY'] = "TRANS_TR"
all["sozlesme-koken"] = all['SOZLESME_KOKENI'] + '_' + all['SOZLESME_KOKENI_DETAY']

all["yatirim_k_null"] = all['YATIRIM_KARAKTERI'].isna()

all['YAS_range'] = pd.cut(all['YAS'], 10, labels=False)
gelir_medians = all[(all["BASLANGIC_yil"] >= 2014) & (all["POLICE_COUNTRY"] == ' Turkey') & (all["GELIR"] >= 2324)].groupby(['MESLEK_KIRILIM', 'YAS_range', 'EGITIM_DURUM'])['GELIR'].median().reset_index().rename(columns={'GELIR': 'gelir_median'})
all = all.merge(gelir_medians, how='left', on=['MESLEK_KIRILIM','YAS_range', 'EGITIM_DURUM'])
all.loc[all['GELIR'].isna(), 'GELIR'] = all['gelir_median']
gelir_medians = all[(all["BASLANGIC_yil"] >= 2014) & (all["POLICE_COUNTRY"] == ' Turkey') & (all["GELIR"] >= 2324)].groupby(['MESLEK_KIRILIM', 'YAS'])['GELIR'].median().reset_index().rename(columns={'GELIR': 'gelir_median_lvl2'})
all = all.merge(gelir_medians, how='left', on=['MESLEK_KIRILIM', 'YAS'])
all.loc[all['GELIR'].isna(), 'GELIR'] = all['gelir_median_lvl2']
gelir_medians = all[(all["BASLANGIC_yil"] >= 2014) & (all["POLICE_COUNTRY"] == ' Turkey') & (all["GELIR"] >= 2324)].groupby('YAS_range')['GELIR'].median().reset_index().rename(columns={'GELIR': 'gelir_median_lvl3'})
all = all.merge(gelir_medians, how='left', on='YAS_range')
all.loc[all['GELIR'].isna(), 'GELIR'] = all['gelir_median_lvl3']

all['vade/gelir'] = all['ARALIK_VADE_TUTARI'] / all['GELIR']
all['vade/gelir'] = np.clip(all['vade/gelir'], 0.0001, 0.5)

aylar = ["OCAK", "SUBAT","MART","NISAN","MAYIS","HAZIRAN","TEMMUZ","AGUSTOS","EYLUL","EKIM","KASIM","ARALIK"]
ilk_ceyrek = ["OCAK", "SUBAT","MART"]
all["zero_months"] = 0
all["under_months"] = 0
all["last_change_months_ago"] = 13
all["last_paid_months_ago"] = 13
all["changed_first_quarter"] = 0
all["odenen_vade_tutar"] = 0
all["odeme_karakteri"] = ""
all["vade_artirma_karakteri"] = ""
all["vade_big_artirma_karakteri"] = ""
all["vade_azaltma_karakteri"] = ""
all["vade_artirma_sayisi"] = 0
all["vade_azaltma_sayisi"] = 0
all.loc[(all[f"MART_VADE_TUTARI"] - all[f"OCAK_VADE_TUTARI"]) > 0, "up_first_quarter"] = 1
for idx, ay in enumerate(aylar):
    all["under_months"] = all["under_months"] + ((all[f"{ay}_ODENEN_TUTAR"] - (all[f"{ay}_VADE_TUTARI"] + 0.001).round()) < 0).astype(int)
    all["odeme_karakteri"] = all["odeme_karakteri"] + (all[f"{ay}_ODENEN_TUTAR"] - (all[f"{ay}_VADE_TUTARI"] + 0.001).round() >= 0).astype(int).astype(str)
    all["zero_months"] = all["zero_months"] + (all[f"{ay}_ODENEN_TUTAR"] == 0).astype(int)
    if ay != "OCAK":
        all["vade_artirma_karakteri"] = all["vade_artirma_karakteri"] + ((all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) > 0).astype(int).astype(str)
        all["vade_big_artirma_karakteri"] = all["vade_big_artirma_karakteri"] + ((all[f"{ay}_VADE_TUTARI"] / all[f"{aylar[idx-1]}_VADE_TUTARI"]) > 1.15).astype(int).astype(str)
        all["vade_azaltma_karakteri"] = all["vade_azaltma_karakteri"] + ((all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) < 0).astype(int).astype(str)
        all["vade_artirma_sayisi"] = all["vade_artirma_sayisi"] + ((all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) > 0).astype(int)
        all["vade_azaltma_sayisi"] = all["vade_azaltma_sayisi"] + ((all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) < 0).astype(int)
        all.loc[(all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) != 0, "last_change_months_ago"] = 12 - idx

    all.loc[(all[f"{ay}_ODENEN_TUTAR"] > 0), "last_paid_months_ago"] = 12 - idx
    all["odenen_vade_tutar"] = all["odenen_vade_tutar"] + all[f"{ay}_ODENEN_TUTAR"]

all["odeme_karakteri_q"] = (all["odeme_karakteri"].str[:2] != "00").astype(int).astype(str) + (all["odeme_karakteri"].str[2:5] != "000").astype(int).astype(str) + (all["odeme_karakteri"].str[5:8] != "000").astype(int).astype(str) + (all["odeme_karakteri"].str[8:11] != "000").astype(int).astype(str)
all["vade_artirma_karakteri_q"] = (all["vade_artirma_karakteri"].str[:2] != "00").astype(int).astype(str) + (all["vade_artirma_karakteri"].str[2:5] != "000").astype(int).astype(str) + (all["vade_artirma_karakteri"].str[5:8] != "000").astype(int).astype(str) + (all["vade_artirma_karakteri"].str[8:11] != "000").astype(int).astype(str)
all["vade_big_artirma_karakteri_q"] = (all["vade_big_artirma_karakteri"].str[:2] != "00").astype(int).astype(str) + (all["vade_big_artirma_karakteri"].str[2:5] != "000").astype(int).astype(str) + (all["vade_big_artirma_karakteri"].str[5:8] != "000").astype(int).astype(str) + (all["vade_big_artirma_karakteri"].str[8:11] != "000").astype(int).astype(str)
all["vade_azaltma_karakteri_q"] = (all["vade_azaltma_karakteri"].str[:2] != "00").astype(int).astype(str) + (all["vade_azaltma_karakteri"].str[2:5] != "000").astype(int).astype(str) + (all["vade_azaltma_karakteri"].str[5:8] != "000").astype(int).astype(str) + (all["vade_azaltma_karakteri"].str[8:11] != "000").astype(int).astype(str)
all['vade_kusuratli'] = (all['ARALIK_VADE_TUTARI'].round() != all['ARALIK_VADE_TUTARI']).astype(int)
all['vade_kusurati'] = all['ARALIK_VADE_TUTARI'] - all['ARALIK_VADE_TUTARI'].astype(int)
all['vade_increase_ratio'] = all['ARALIK_VADE_TUTARI'] / all['OCAK_VADE_TUTARI']

series = pd.value_counts(all['vade_group'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(vade_group = np.where(all['vade_group'].isin(series[mask].index), 'Other', all['vade_group']))
series = pd.value_counts(all['odeme_group'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(odeme_group = np.where(all['odeme_group'].isin(series[mask].index), 'Other', all['odeme_group']))
series = pd.value_counts(all['odeme_karakteri'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(odeme_karakteri = np.where(all['odeme_karakteri'].isin(series[mask].index), 'Other', all['odeme_karakteri']))
series = pd.value_counts(all['vade_artirma_karakteri'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(vade_artirma_karakteri = np.where(all['vade_artirma_karakteri'].isin(series[mask].index), 'Other', all['vade_artirma_karakteri']))
series = pd.value_counts(all['vade_azaltma_karakteri'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(vade_azaltma_karakteri = np.where(all['vade_azaltma_karakteri'].isin(series[mask].index), 'Other', all['vade_azaltma_karakteri']))

for target_encoding_feature in target_encoding_features:
    target_stats = all.groupby(target_encoding_feature)['ARTIS_DURUMU'].agg(['count','mean', 'std', 'median']).reset_index().rename(columns={'count': f'{target_encoding_feature}_count',
                                                                                                                                             'mean': f'{target_encoding_feature}_mean',
                                                                                                                                             'std': f'{target_encoding_feature}_std',
                                                                                                                                             'median': f'{target_encoding_feature}_median'
                                                                                                                                             })
    gelir_stats = all.groupby(target_encoding_feature)['GELIR'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index().rename(columns={
                                                                                                                                 'mean': f'{target_encoding_feature}_gelir_mean',
                                                                                                                                 'std': f'{target_encoding_feature}_gelir_std',
                                                                                                                                 'median': f'{target_encoding_feature}_gelir_median',
                                                                                                                                 'min': f'{target_encoding_feature}_gelir_min',
                                                                                                                                 'max': f'{target_encoding_feature}_gelir_max'
                                                                                                                                             })
    vade_stats = all.groupby(target_encoding_feature)['odenen_vade_tutar'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index().rename(columns={
                                                                                                                                 'mean': f'{target_encoding_feature}_vade_mean',
                                                                                                                                 'std': f'{target_encoding_feature}_vade_std',
                                                                                                                                 'median': f'{target_encoding_feature}_vade_median',
                                                                                                                                 'min': f'{target_encoding_feature}_vade_min',
                                                                                                                                 'max': f'{target_encoding_feature}_vade_max'
                                                                                                                                             })

    smooth = 100
    prior = all['ARTIS_DURUMU'].mean()
    n = target_stats[f'{target_encoding_feature}_count']
    mu = target_stats[f'{target_encoding_feature}_mean']
    mu_smoothed = (n * mu + smooth * prior) / (n + smooth)
    target_stats[f'{target_encoding_feature}_mean_s'] = mu_smoothed
    all = all.merge(target_stats[[target_encoding_feature, f'{target_encoding_feature}_mean',f'{target_encoding_feature}_mean_s', f'{target_encoding_feature}_count', f'{target_encoding_feature}_std']], on=target_encoding_feature, how='left')

    all = all.merge(gelir_stats[[target_encoding_feature, f'{target_encoding_feature}_gelir_mean',  f'{target_encoding_feature}_gelir_std',  f'{target_encoding_feature}_gelir_median',  f'{target_encoding_feature}_gelir_min',  f'{target_encoding_feature}_gelir_max']], on=target_encoding_feature, how='left')
    all = all.merge(vade_stats[[target_encoding_feature, f'{target_encoding_feature}_vade_mean',  f'{target_encoding_feature}_vade_std',  f'{target_encoding_feature}_vade_median',  f'{target_encoding_feature}_vade_min',  f'{target_encoding_feature}_vade_max']], on=target_encoding_feature, how='left')

time_steps = [
        ("OCAK", "SUBAT", "MART"),
        ("NISAN", "MAYIS", "HAZIRAN"),
        ("TEMMUZ", "AGUSTOS", "EYLUL"),
        ("EKIM", "KASIM", "ARALIK"),
    ]
for idx, ts in enumerate(time_steps):
    t_odenen_df = all[[col + "_ODENEN_TUTAR" for col in ts]]
    t_vade_df = all[[col + "_VADE_TUTARI" for col in ts]]

    all[f"ts_{idx + 1}_odenen_max"] = t_odenen_df.max(axis=1)
    all[f"ts_{idx + 1}_odenen_min"] = t_odenen_df.min(axis=1)
    all[f"ts_{idx + 1}_odenen_mean"] = t_odenen_df.mean(axis=1)
    all[f"ts_{idx + 1}_odenen_sum"] = t_odenen_df.sum(axis=1)

    all[f"ts_{idx + 1}_vade_max"] = t_vade_df.max(axis=1)
    all[f"ts_{idx + 1}_vade_min"] = t_vade_df.min(axis=1)
    all[f"ts_{idx + 1}_vade_mean"] = t_vade_df.mean(axis=1)
    all[f"ts_{idx + 1}_vade_sum"] = t_vade_df.sum(axis=1)

    vade_first_month_of_ts = ts[0] + "_VADE_TUTARI"
    vade_last_month_of_ts = ts[-1] + "_VADE_TUTARI"
    odenen_first_month_of_ts = ts[0] + "_ODENEN_TUTAR"
    odenen_last_month_of_ts = ts[-1] + "_ODENEN_TUTAR"
    all[f"vade_up_{idx}_th_quarter"] = (
        all[vade_last_month_of_ts] - all[vade_first_month_of_ts] > 0
    ).astype(int)
    all[f"odenen_up_{idx}_th_quarter"] = (
        all[odenen_last_month_of_ts] - all[odenen_first_month_of_ts] > 0
    ).astype(int)

all["vade_not_changed"] = all["last_change_months_ago"] == 13
all["under_months_all_year"] = all["under_months"] == 12
all["zero_months_all_year"] = all["zero_months"] == 12
all["well_paid_customer"] = all["under_months"] == 0
all["ratio_increase"] = all["SENE_SONU_HESAP_DEGERI"] / all["SENE_BASI_HESAP_DEGERI"]
all.loc[all["SENE_SONU_HESAP_DEGERI"] <= 0, "ratio_increase"] = np.nan
all.loc[all["SENE_BASI_HESAP_DEGERI"] <= 0, "ratio_increase"] = np.nan
all["ratio_increase"] = np.clip(all["ratio_increase"], 0.2, 5)
all["hesap_degisimi"] = all["SENE_SONU_HESAP_DEGERI"] - all["SENE_BASI_HESAP_DEGERI"]
all["ss_0"] = all["SENE_SONU_HESAP_DEGERI"] <= 0
all["sb_0"] = all["SENE_BASI_HESAP_DEGERI"] <= 0
all["para_cekilen_oran"] = (all["odenen_vade_tutar"] + all["SENE_BASI_HESAP_DEGERI"] - all["SENE_SONU_HESAP_DEGERI"]) / (all["odenen_vade_tutar"] + all["SENE_BASI_HESAP_DEGERI"])
all["hesap_degisimi_faiz"] = all["hesap_degisimi"] - all["odenen_vade_tutar"]
all["yuzde_kazanc"] = (all["hesap_degisimi"] - all["odenen_vade_tutar"]) / all["SENE_BASI_HESAP_DEGERI"]
all["vade_up"] = all["ARALIK_VADE_TUTARI"] > all["OCAK_VADE_TUTARI"]
all["vade_down"] = all["ARALIK_VADE_TUTARI"] < all["OCAK_VADE_TUTARI"]

dup_check_features = all.drop(['POLICY_ID','ARTIS_DURUMU'],axis=1).columns.tolist()

# all["is_duplicated"] = all[['GELIR','POLICE_SEHIR', 'MUSTERI_SEGMENTI', 'YAS', 'MESLEK', 'MESLEK_KIRILIM', 'MEDENI_HAL', 'COCUK_SAYISI', 'EGITIM_DURUM', 'CINSIYET', 'UYRUK']].duplicated()
# all["dup_id"] = all.groupby(['GELIR','POLICE_SEHIR', 'MUSTERI_SEGMENTI', 'YAS', 'MESLEK', 'MESLEK_KIRILIM', 'MEDENI_HAL', 'COCUK_SAYISI', 'EGITIM_DURUM', 'CINSIYET', 'UYRUK'], sort=False).ngroup() + 1

all["is_duplicated"] = all[dup_check_features].duplicated()
all["dup_id"] = all.groupby(dup_check_features, sort=False).ngroup() + 1
all.loc[all["dup_id"] ==0, "dup_id"] = np.nan
all.loc[all["dup_id"] ==0, "is_duplicated"] = False
dup_stats = all.groupby('dup_id')['ARTIS_DURUMU'].agg(['count', 'mean']).rename(columns={'count': 'dup_count', 'mean':'dup_mean'})
dup_stats = dup_stats[dup_stats['dup_count'] >= 1]
all = pd.merge(all, dup_stats, on="dup_id", how='left')
all.loc[(all['ARTIS_DURUMU'].isna()) & (all['dup_mean'].notna()), "target_overwrite"] = all[(all['ARTIS_DURUMU'].isna()) & (all['dup_mean'].notna())]["dup_mean"]
all = all[(all["is_duplicated"] == False) | ( all['ARTIS_DURUMU'].isna())]

all["is_wide_duplicated"] = all[['GELIR','POLICE_SEHIR', 'MUSTERI_SEGMENTI', 'YAS', 'MESLEK', 'MESLEK_KIRILIM', 'MEDENI_HAL', 'COCUK_SAYISI', 'EGITIM_DURUM', 'CINSIYET', 'UYRUK']].duplicated()
all["wide_dup_id"] = all.groupby(['GELIR','POLICE_SEHIR', 'MUSTERI_SEGMENTI', 'YAS', 'MESLEK', 'MESLEK_KIRILIM', 'MEDENI_HAL', 'COCUK_SAYISI', 'EGITIM_DURUM', 'CINSIYET', 'UYRUK'], sort=False).ngroup() + 1
all.loc[all["wide_dup_id"] ==0, "wide_dup_id"] = np.nan
all.loc[all["wide_dup_id"] ==0, "wide_is_duplicated"] = False
dup_stats = all.groupby('wide_dup_id')['ARTIS_DURUMU'].agg(['count', 'mean']).rename(columns={'count': 'wide_dup_count', 'mean':'wide_dup_mean'})
dup_stats = dup_stats[dup_stats['wide_dup_count'] > 1]
dup_stats = dup_stats[dup_stats['wide_dup_count'] < 7]
all = pd.merge(all, dup_stats, on="wide_dup_id", how='left')

all = downcast_df_int_columns(all)
all = downcast_df_float_columns(all)

target_feature = 'ARTIS_DURUMU'
all["is_train"] = all[target_feature].notnull()
all = all.replace([np.inf, -np.inf], np.nan)

features = [
    "vade_group",
    "odeme_group",
    "odeme_karakteri",
    "vade_increase_ratio",
    "vade_artirma_karakteri",
    "vade_big_artirma_karakteri",
    "vade_big_artirma_karakteri_q",
    "vade_azaltma_karakteri",
    "odeme_karakteri_q",
    "vade_artirma_karakteri_q",
    "vade_azaltma_karakteri_q",
    "vade_artirma_sayisi",
    "vade_azaltma_sayisi",
    "vade_pct_change_mean",
    "vade_pct_change_max",
    "vade_pct_change_min",
    "odenen_pct_change_mean",
    "odenen_pct_change_max",
    "odenen_pct_change_min",
    "vade_mean",
    "vade_nunique",
    "vade_std",
    "vade_median",
    "vade_min",
    "vade_max",
    "vade_skew",
    "odeme_mean",
    "odeme_std",
    "odeme_median",
    "odeme_min",
    "odeme_max",
    "odeme_skew",
    #'POLICY_ID',
    "OFFICE_ID",
    "SIGORTA_TIP",
    "yerli_milli",
    "SOZLESME_KOKENI",
    #"SOZLESME_KOKENI_DETAY",
    #"sozlesme-koken",
    "KAPSAM_TIPI",
    "KAPSAM_GRUBU",
    "DAGITIM_KANALI",
    "POLICE_SEHIR",
    "POLICE_COUNTRY",
    "CINSIYET",
    "UYRUK",
    "MEMLEKET",
     "MESLEK",
     "MESLEK_KIRILIM",
    "meslek-krlm",
    "MUSTERI_SEGMENTI",
    "YATIRIM_KARAKTERI",
    #"yatirim_k_null",
    "MEDENI_HAL",
    "EGITIM_DURUM",
    "GELIR",
    "COCUK_SAYISI",
    # "OCAK_ODENEN_TUTAR",
    # "OCAK_VADE_TUTARI",
    # "SUBAT_ODENEN_TUTAR",
    # "SUBAT_VADE_TUTARI",
    # "MART_ODENEN_TUTAR",
    # "MART_VADE_TUTARI",
    # "NISAN_ODENEN_TUTAR",
    # "NISAN_VADE_TUTARI",
    # "MAYIS_ODENEN_TUTAR",
    # "MAYIS_VADE_TUTARI",
    # "HAZIRAN_ODENEN_TUTAR",
    # "HAZIRAN_VADE_TUTARI",
    # "TEMMUZ_ODENEN_TUTAR",
    # "TEMMUZ_VADE_TUTARI",
    # "AGUSTOS_ODENEN_TUTAR",
    # "AGUSTOS_VADE_TUTARI",
    "EYLUL_ODENEN_TUTAR",
    # "EYLUL_VADE_TUTARI",
    "EKIM_ODENEN_TUTAR",
    # "EKIM_VADE_TUTARI",
    "KASIM_ODENEN_TUTAR",
    # "KASIM_VADE_TUTARI",
    #"ARALIK_ODENEN_TUTAR",
    "ARALIK_VADE_TUTARI",
    "vade_kusuratli",
    'vade/gelir',
    "SENE_BASI_HESAP_DEGERI",
    "SENE_SONU_HESAP_DEGERI",
    #"sb_0",
    #"ss_0",
    "BASLANGIC_ay",
    #"BASLANGIC_yil",
    "month_passed",
    #"year_passed",
    "YAS",
    "BASLANGIC_YAS",
    "under_months",
    "zero_months",
    "under_months_all_year",
    #"zero_months_all_year",
    #"well_paid_customer",
    "odenen_vade_tutar",
    "hesap_degisimi_faiz",
    #"ratio_increase",
    "vade_up",
    #"vade_down",
    "hesap_degisimi",
    "last_change_months_ago",
    "last_paid_months_ago",
    "vade_not_changed",
    'ts_1_odenen_max', 'ts_1_odenen_min',
    'ts_1_odenen_mean',
    'ts_1_odenen_sum',
    'ts_1_vade_max', 'ts_1_vade_min',
    'ts_1_vade_mean',
    'ts_1_vade_sum',
    'vade_up_0_th_quarter', 'odenen_up_0_th_quarter',
    'ts_2_odenen_max',
    'ts_2_odenen_min', 'ts_2_odenen_mean', 'ts_2_odenen_sum',
    'ts_2_vade_max',
    'ts_2_vade_min',
    'ts_2_vade_mean',
    'ts_2_vade_sum',
    'vade_up_1_th_quarter', 'odenen_up_1_th_quarter', 'ts_3_odenen_max', 'ts_3_odenen_min',
    'ts_3_odenen_mean',
    'ts_3_odenen_sum',
    'ts_3_vade_max', 'ts_3_vade_min',
    'ts_3_vade_mean',
    'ts_3_vade_sum',
    'vade_up_2_th_quarter', 'odenen_up_2_th_quarter', 'ts_4_odenen_max', 'ts_4_odenen_min',
    'ts_4_odenen_mean',
    'ts_4_odenen_sum',
    'ts_4_vade_max',
    'ts_4_vade_min',
    'ts_4_vade_mean',
    'ts_4_vade_sum'
    #"is_duplicated"
    #"up_first_quarter"
]

impute_features = [
    # "OFFICE_ID",
    # "SIGORTA_TIP",
    # "SOZLESME_KOKENI",
    # "SOZLESME_KOKENI_DETAY",
    # "KAPSAM_TIPI",
    # "KAPSAM_GRUBU",
    "DAGITIM_KANALI",
    #"POLICE_SEHIR",
    "CINSIYET",
    "UYRUK",
    "MEMLEKET",
    #"MESLEK",
    #"MESLEK_KIRILIM",
    "MUSTERI_SEGMENTI",
    #"YATIRIM_KARAKTERI",
    "MEDENI_HAL",
    "EGITIM_DURUM",
    "GELIR",
    # "COCUK_SAYISI",
    # "ARALIK_VADE_TUTARI",
    # "SENE_BASI_HESAP_DEGERI",
    # "SENE_SONU_HESAP_DEGERI",
    # "BASLANGIC_ay",
    # "BASLANGIC_yil",
    # "month_passed",
    # "YAS",
    # "BASLANGIC_YAS",
    # "under_months",
    # "zero_months",
    # "under_months_all_year",
    # "zero_months_all_year",
    # "well_paid_customer",
     #"ratio_increase",
    #"vade_up",
    #"vade_down",
    # "hesap_degisimi",
    # "last_change_months_ago",
    # "vade_not_changed",
]
categorical_features = [
    "vade_group",
    "odeme_group",
    'OFFICE_ID',
    'SIGORTA_TIP',
    'SOZLESME_KOKENI',
    "odeme_karakteri",
    "vade_artirma_karakteri",
    "vade_big_artirma_karakteri",
    "vade_big_artirma_karakteri_q",
    "vade_azaltma_karakteri",
    "odeme_karakteri_q",
    "vade_artirma_karakteri_q",
    "vade_azaltma_karakteri_q",
    #'SOZLESME_KOKENI_DETAY',
    # "sb_0",
    # "ss_0",
    'KAPSAM_TIPI',
    'KAPSAM_GRUBU',
    'DAGITIM_KANALI',
    'POLICE_SEHIR',
    "POLICE_COUNTRY",
    'CINSIYET',
    'UYRUK',
    'MEMLEKET',
    'MESLEK',
    'MESLEK_KIRILIM',
    "meslek-krlm",
    #"meslek_is_bill",
    'MUSTERI_SEGMENTI',
    'YATIRIM_KARAKTERI',
    #"vade_kusuratli",
    #"yatirim_k_null",
    'MEDENI_HAL',
    'EGITIM_DURUM',
    #"under_months_all_year",
    #"zero_months_all_year",
    #"well_paid_customer",
    #"vade_up",
    #"vade_down",
    #"vade_not_changed",
    #"up_first_quarter",
]

lol_features = [
    'KAPSAM_TIPI',
    'KAPSAM_GRUBU',
    'DAGITIM_KANALI',
    'POLICE_SEHIR',
    "POLICE_COUNTRY",
    'CINSIYET',
    'UYRUK',
    'MEMLEKET',
    'MESLEK',
    'MESLEK_KIRILIM',
]
for idx, categorical_feature in enumerate(lol_features):
    if idx < len(lol_features) -1:
        for other_id in range(idx+1, len(lol_features) -1):
            all[f"combination{idx}_{other_id}"] = all[lol_features[idx]].astype(str) + '_' + all[lol_features[other_id]].astype(str)

for idx, categorical_feature in enumerate(lol_features):
    if idx < len(lol_features) -1:
        for other_id in range(idx+1, len(lol_features) - 1):
            categorical_features.append(f"combination{idx}_{other_id}")
            features.append(f"combination{idx}_{other_id}")

for target_encoding_feature in target_encoding_features:
    features.append(f"{target_encoding_feature}_count")
    features.append(f"{target_encoding_feature}_mean_s")
    features.append(f"{target_encoding_feature}_std")
    features.append(f"{target_encoding_feature}_gelir_mean")
    features.append(f"{target_encoding_feature}_vade_mean")
    # features.append(f"{target_encoding_feature}_gelir_std")
    # features.append(f"{target_encoding_feature}_vade_std")
    # features.append(f"{target_encoding_feature}_gelir_min")
    # features.append(f"{target_encoding_feature}_vade_min")
    # features.append(f"{target_encoding_feature}_gelir_max")
    # features.append(f"{target_encoding_feature}_vade_max")
    # features.append(f"{target_encoding_feature}_gelir_median")
    # features.append(f"{target_encoding_feature}_vade_median")
    # features.append(f"{target_encoding_feature}_median")
    #features.remove(target_encoding_feature)
    #categorical_features.remove(target_encoding_feature)
    pass
ban_features = [
]
for ban_feature in ban_features:
    try:
        features.remove(ban_feature)
        categorical_features.remove(ban_feature)
    except:
        pass
numeric_cols = list(set(all[features].columns) - set(categorical_features))

# for col_name in categorical_features:
#     series = all[col_name]
#     label_encoder = LabelEncoder()
#     all[col_name] = pd.Series(
#         label_encoder.fit_transform(series[series.notnull()]),
#         index=series[series.notnull()].index
#     )

label_encoder = LabelEncoder()
all[categorical_features] = all[categorical_features].apply(label_encoder.fit_transform)

#all[categorical_features] = all[categorical_features].astype('category')

if run_imputer:
    imputer = LGBMImputer(verbose=True, feature_list=['GELIR'])
    all.loc[:, features] = imputer.fit_transform(all[features])



train_x = all[all.is_train == 1][features]
train_y = all[all.is_train == 1][target_feature]
test_x = all[all.is_train == 0][features]

N_FOLDS = 5

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=777)
y_oof = np.zeros(train_x.shape[0])
y_test = np.zeros(test_x.shape[0])
categorical_indices = [train_x.columns.get_loc(c) for c in features if c in categorical_features]
ix = 0

if run_lofo:
    sample_df = all[all.is_train==1].sample(frac=0.5, random_state=555)
    dataset = Dataset(df=sample_df, target=target_feature, features=features)
    lofo_imp = LOFOImportance(dataset, cv=skf, scoring="f1")
    importance_df = lofo_imp.get_importance()
    print(importance_df)
    plot_importance(importance_df, figsize=(12, 20))

def xgb_f1(y, t, threshold=0.5):
    t = t.get_label()
    y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
    return 'f1', 1 - f1_score(t,y_bin)

def lgb_f1_score(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred), True

def objective_hgbt(trial):
    y_oof = np.zeros(train_x.shape[0])

    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.05),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 50, 250, 5),
        "max_depth": trial.suggest_int("max_depth", 10, 20, 1),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, 1),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.3, 5),
        "loss": 'binary_crossentropy'
    }

    param["random_state"] = 61
    model = HistGradientBoostingClassifier(**param)
    ix = 0
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )
        w = tr_y.copy()
        w = np.where(w == 1, param["scale_pos_weight"], w)
        w = np.where(w == 0, 1, w)



        model.fit(
            tr_x,
            tr_y,
            sample_weight=w
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    score = f1_score(train_y, y_oof)

    return score

def objective_cat(trial):
    y_oof = np.zeros(train_x.shape[0])

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    param = {
        "iterations": 500,
        #"scale_pos_weight": 3.471475485069041,
        #"scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.3, 5),
        "objective": "Logloss",
        #"eval_metric": "F1",
        "task_type": "GPU",
        #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 5, 15),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        #"depth": 9,
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0001, 1.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 500),
    }

    param["random_state"] = 61
    param["eval_metric"] = "F1:use_weights=False"
    param["auto_class_weights"] = 'SqrtBalanced'
    model = CatBoostClassifier(**param)
    ix = 0
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            cat_features=categorical_features,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=100,
            verbose=500,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    score = f1_score(train_y, y_oof)

    return score

def objective_xgb(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, observation_key="validation_0-f1"
    )
    param = {
        "tree_method": "gpu_hist",  # Use GPU acceleration
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 1e2),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 1e2),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "scale_pos_weight": 3.471475485069041,
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, 100),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "random_state": 61,
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.1, 10),
        "objective": "binary:logistic",
        "enable_categorical": True,
    }
    model = XGBClassifier(**param)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric=xgb_f1,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=100,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    y_oof = (y_oof > 0.5).astype(int)
    score = f1_score(train_y, y_oof)

    return score

def objective_lgb(trial):

    # To select which parameters to optimize, please look at the XGBoost documentation:
    # https://xgboost.readthedocs.io/en/latest/parameter.html
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric="f1")
    param = {
        "random_state": 61,
        "metric": "custom",
        "n_estimators": 3000,
        "scale_pos_weight": 3.471475485069041,
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
        "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.1),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "num_leaves": trial.suggest_int("num_leaves", 1, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 300),
    }
    model = LGBMClassifier(**param)
    y_oof = np.zeros(train_x.shape[0])

    ix = 0
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric=lgb_f1_score,
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=100,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        ix = ix + 1

    y_oof = (y_oof > 0.5).astype(int)
    score = f1_score(train_y, y_oof)

    return score


lgb_params = {}
xgb_params = {}
cat_params = {}
if run_optuna:
    study = optuna.create_study(direction="maximize")
    if model_type == "cat":
        study.optimize(objective_cat, n_trials=optuna_trials)
        cat_params = study.best_params
    elif model_type == "xgb":
        study.optimize(objective_xgb, n_trials=optuna_trials)
        xgb_params = study.best_params
    elif model_type == "lgb":
        study.optimize(objective_lgb, n_trials=optuna_trials)
        lgb_params = study.best_params
    elif model_type == "hgbt":
        study.optimize(objective_hgbt, n_trials=optuna_trials)
        lgb_params = study.best_params
    else:
        exit(1)
else:
    lgb_params = {'reg_alpha': 2.0604665531147646, 'reg_lambda': 0.0508941398294875, 'colsample_bytree': 0.8, 'subsample': 0.7, 'learning_rate': 0.01, 'max_depth': 12, 'num_leaves': 1000, 'min_child_samples': 30}
    cat_params = {'depth': 9, 'bootstrap_type': 'Bernoulli', 'boosting_type': 'Plain', 'subsample': 0.993}
    #cat_params = {'one_hot_max_size': 48, 'colsample_bylevel': 0.05440878981719498, 'depth': 7, 'boosting_type': 'Plain', 'bootstrap_type': 'MVS'}
    #cat_params = {'bootstrap_type': 'Bernoulli', 'boosting_type': 'Plain', 'depth': 9, 'max_bin': 252, 'subsample': 0.898250286630696, 'l2_leaf_reg': 0.11017941019690093, 'min_data_in_leaf': 338}
    #cat_params = {'bootstrap_type': 'Bernoulli', 'boosting_type': 'Plain', 'depth': 8, 'max_bin': 246, 'subsample': 0.9686988180575133, 'l2_leaf_reg': 0.46043929647227483, 'min_data_in_leaf': 362}
    xgb_params = {'reg_lambda': 0.2716863446664029, 'reg_alpha': 32.796648528916805, 'colsample_bytree': 0.9, 'subsample': 1.0, 'learning_rate': 0.02987792485738012, 'n_estimators': 1000, 'max_depth': 7, 'min_child_weight': 2.7489363264568705}
    hgbt_params = {'learning_rate': 0.04654814855554209, 'max_leaf_nodes': 200, 'max_depth': 16, 'min_samples_leaf': 25}
lgb_params["metric"] = "custom"
lgb_params["n_estimators"] = 3000
lgb_params["random_state"] = 61
lgb_params["scale_pos_weight"] = 3.24
xgb_params["tree_method"] = "gpu_hist"
xgb_params["random_state"] = 61
xgb_params["scale_pos_weight"] = 1
xgb_params["enable_categorical"] = True
#cat_params["scale_pos_weight"] = 3.240440311
cat_params["objective"] = "Logloss"
cat_params["eval_metric"] = "F1:use_weights=False"
cat_params["task_type"] = "GPU"
cat_params["random_state"] = 61
cat_params["iterations"] = 5000
cat_params["verbose"] = 300
cat_params["auto_class_weights"] = 'SqrtBalanced'
hgbt_params["loss"] = 'binary_crossentropy'
#hgbt_params["scale_pos_weight"] = 3.240440311

if model_type == "cat":
    importance_list = []
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = CatBoostClassifier(**cat_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )
        model.fit(tr_x,tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=600,
            cat_features=categorical_features,
            verbose=500,
        )

        if plot_importance:

            importances = model.feature_importances_
            importance_list.append(importances)

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds
        y_oof = (y_oof >= 0.5).astype(int)

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1

    if plot_importance:
        importances = np.mean(importance_list, axis=0)
        indices = np.argsort(importances)
        for indice in indices:
            print(f"'{features[indice]}',")

        plt.set_cmap("inferno")
        plt.figure(figsize=(30, 10))
        plt.title("Feature Importances")
        plt.barh(
            range(len(indices)), importances[indices], color="b", align="center",
        )
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.show()
elif model_type == "xgb":
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = XGBClassifier(**xgb_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric=xgb_f1,
            early_stopping_rounds=600,
            verbose=100,
        )

        if plot_importance:
            if ix == 0:
                xgb.plot_tree(model, num_trees=2)
                fig = plt.gcf()
                fig.set_size_inches(100, 30)
                fig.savefig('tree.png')
            importances = model.feature_importances_
            indices = np.argsort(importances)
            indices = indices[-50:]

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1
elif model_type == "lgb":
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = LGBMClassifier(**lgb_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            eval_metric=lgb_f1_score,
            categorical_feature=categorical_features,
            early_stopping_rounds=100,
            verbose=100,
        )

        if plot_importance and ix == 0:
            lgb.plot_tree(model, figsize=(20,6), tree_index=0, dpi=600)
            importances = model.feature_importances_
            indices = np.argsort(importances)
            #indices = indices[-50:]

            plt.figure(figsize=(30, 10))
            plt.set_cmap("viridis")
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1
elif model_type == "hgbt":
    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model = HistGradientBoostingClassifier(**hgbt_params)
        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        w = tr_y.copy()
        w = np.where(w == 1, 3.24, w)
        w = np.where(w == 0, 1, w)

        model.fit(
            tr_x,
            tr_y,
            sample_weight=w
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1
if model_type == "stacking":

    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model_xgb = XGBClassifier(**xgb_params)
        model_cat = CatBoostClassifier(**cat_params)

        # xgb_params["scale_pos_weight"] = 1
        # lgb_params["scale_pos_weight"] = 1
        cat_params["scale_pos_weight"] = 5
        cat_params["auto_class_weights"] = None
        #
        # model_xgb_2 = XGBClassifier(**xgb_params)
        # model_lgb_2 = LGBMClassifier(**lgb_params)
        model_cat_2 = CatBoostClassifier(**cat_params)

        model = VotingClassifier(estimators = [('xgb', model_xgb),  ('cat', model_cat), ('cat_2', model_cat_2)], voting='hard', verbose=True)

        # model = StackingClassifier(classifiers=[model_xgb, model_lgb, model_cat],
        #                            use_probas=True,
        #                            verbose=100,
        #                            meta_classifier=LogisticRegression(class_weight={0: 0.08, 1: 0.92}))

        tr_x, val_x = (
            train_x.iloc[train_ind].reset_index(drop=True),
            train_x.iloc[val_ind].reset_index(drop=True),
        )
        tr_y, val_y = (
            train_y.iloc[train_ind].reset_index(drop=True),
            train_y.iloc[val_ind].reset_index(drop=True),
        )

        model.fit(
            tr_x,
            tr_y,
        )

        preds = model.predict(val_x)
        y_oof[val_ind] = y_oof[val_ind] + preds

        test_preds = model.predict(test_x)
        y_test = y_test + test_preds / N_FOLDS
        ix = ix + 1

val_score = f1_score(train_y, y_oof)
acc_score = accuracy_score(train_y, y_oof)
cm = confusion_matrix(train_y, y_oof)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm,
                      classes = class_names,
                      title = f'{model_type}_{trial} matrix')
plt.show()
print(f"Val F-1 Score: {val_score}")
print(f"Val Accuracy Score: {acc_score}")
print(f"Val Mean: {y_oof.mean()}")


sample_submission[target_feature] = y_test
sample_submission.loc[sample_submission[target_feature] >= 0.5, target_feature] = 1
sample_submission.loc[sample_submission[target_feature] < 0.5, target_feature] = 0
sample_submission[target_feature] = sample_submission[target_feature].astype(int)
sample_submission = pd.merge(sample_submission, all[['POLICY_ID', 'wide_dup_mean', 'target_overwrite', 'ARALIK_VADE_TUTARI', "vade_kusuratli", "zero_months", "vade_group_mean"]], on='POLICY_ID')
sample_submission.loc[sample_submission['wide_dup_mean'] == 0, target_feature] = 0
#sample_submission.loc[(sample_submission['wide_dup_mean'] > 0.5) & (sample_submission['zero_months'] == 0), target_feature] = 1
sample_submission.loc[sample_submission['ARALIK_VADE_TUTARI'].isin([73.0, 216.37, 88.0, 150.7, 241.85, 198.0, 131, 230.79]) & (sample_submission["zero_months"] == 0), target_feature] = 1
sample_submission.loc[(sample_submission['vade_kusuratli'] == 1) & (sample_submission['zero_months'] == 0), target_feature] = 1
sample_submission.loc[(sample_submission['vade_group_mean'] == 0), target_feature] = 0
sample_submission.loc[(sample_submission['vade_group_mean'] == 1), target_feature] = 1
sample_submission.loc[sample_submission['target_overwrite'].notna(), target_feature] = sample_submission['target_overwrite']
print(f"Submission mean: {sample_submission[target_feature].mean()}")
sample_submission[['POLICY_ID', target_feature]].to_csv(f"submissions/submission_{model_type}_{trial}_{val_score}.csv", sep=",", index=False)

with open("submissions/submission_notes.csv", "a") as file_object:
    file_object.write(f"\nsubmission_{model_type}_{trial},{val_score},{acc_score},{sample_submission[target_feature].mean()},{xgb_params['scale_pos_weight']},,{y_oof.mean()}")