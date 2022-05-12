import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy import stats
from scipy.special import inv_boxcox
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, VotingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.svm import SVC
from mlxtend.classifier import StackingCVClassifier, StackingClassifier
from sklearn.tree import ExtraTreeRegressor
import optuna
from tqdm import tqdm
from rgf import RGFClassifier
from constants import (
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

warnings.filterwarnings(action="ignore")
from utils import downcast_df_int_columns, downcast_df_float_columns

sample_submission = pd.read_csv("data/samplesubmission.csv")
test = pd.read_csv("data/test-utf8.csv")
train = pd.read_csv("data/train-utf8.csv")

all = pd.concat([train, test], axis=0)
all = all.rename(columns={'SUBAT_ODENEN_TU': 'SUBAT_ODENEN_TUTAR'})
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

# set up the threshold percent
threshold_percent = 0.005

# series = pd.value_counts(all['MESLEK_KIRILIM'])
# mask = (series / series.sum() * 100).lt(threshold_percent)
# all = all.assign(MESLEK_KIRILIM = np.where(all['MESLEK_KIRILIM'].isin(series[mask].index), 'Other', all['MESLEK_KIRILIM']))

series = pd.value_counts(all['POLICE_COUNTRY'])
mask = (series / series.sum() * 100).lt(threshold_percent)
all = all.assign(POLICE_COUNTRY = np.where(all['POLICE_COUNTRY'].isin(series[mask].index), 'Other', all['POLICE_COUNTRY']))
all["yerli_milli"] = all["POLICE_COUNTRY"] == " Turkey"

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
all.loc[(all['GELIR'] < 100) & (all['GELIR'] > 0), "GELIR"] = np.nan
all["BASLANGIC_TARIHI"] = pd.to_datetime(all["BASLANGIC_TARIHI"])
all["BASLANGIC_ay"] = all["BASLANGIC_TARIHI"].dt.month
all["BASLANGIC_yil"] = all["BASLANGIC_TARIHI"].dt.year
all["month_passed"] = ((pd.to_datetime("2021-01-01") - all["BASLANGIC_TARIHI"]).dt.days / 30).round()
all["YAS"] = 2021 - all["DOGUM_TARIHI"]
all["BASLANGIC_YAS"] = all["YAS"] - all["month_passed"] / 12.0
all.loc[all['COCUK_SAYISI'].isna(), 'COCUK_SAYISI'] = 0
all.loc[all['SOZLESME_KOKENI'] == "TRANS", 'SOZLESME_KOKENI'] = "TRANS_C"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TRANS", 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'].isna(), 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TR_T2_TR", 'SOZLESME_KOKENI_DETAY'] = "TRANS_T2"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "INV_PROC", 'SOZLESME_KOKENI_DETAY'] = "NEW"
all.loc[all['SOZLESME_KOKENI_DETAY'] == "TRANS_C", 'SOZLESME_KOKENI_DETAY'] = "TRANS_TR"
all["sozlesme-koken"] = all['SOZLESME_KOKENI'] + '_' + all['SOZLESME_KOKENI_DETAY']
all["meslek-krlm"] = all['MESLEK'] + '_' + all['MESLEK_KIRILIM']

aylar = ["OCAK", "SUBAT","MART","NISAN","MAYIS","HAZIRAN","TEMMUZ","AGUSTOS","EYLUL","EKIM","KASIM","ARALIK"]
ilk_ceyrek = ["OCAK", "SUBAT","MART"]
all["zero_months"] = 0
all["under_months"] = 0
all["last_change_months_ago"] = np.nan
all["changed_first_quarter"] = 0
all["odenen_vade_tutar"] = 0
all.loc[(all[f"MART_VADE_TUTARI"] - all[f"OCAK_VADE_TUTARI"]) > 0, "up_first_quarter"] = 1
for idx, ay in enumerate(aylar):
    all["under_months"] = all["under_months"] + ((all[f"{ay}_ODENEN_TUTAR"] - (all[f"{ay}_VADE_TUTARI"] + 0.001).round()) < 0).astype(int)
    all["zero_months"] = all["zero_months"] + (all[f"{ay}_ODENEN_TUTAR"] == 0).astype(int)
    all.loc[(all[f"{ay}_VADE_TUTARI"] - all[f"{aylar[idx-1]}_VADE_TUTARI"]) != 0, "last_change_months_ago"] = 12 - idx
    all["odenen_vade_tutar"] = all["odenen_vade_tutar"] + all[f"{ay}_ODENEN_TUTAR"]


all["vade_not_changed"] = all["last_change_months_ago"].isna()
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


# gelir_stats = all.groupby(['MESLEK', 'MESLEK_KIRILIM', 'CINSIYET', 'YAS'])['GELIR'].agg(['count', 'mean']).rename(columns={'count': 'gelir_count', 'mean':'gelir_mean'})
# gelir_stats = gelir_stats[gelir_stats['gelir_count'] > 10]
# all = pd.merge(all, gelir_stats, on=['MESLEK', 'MESLEK_KIRILIM', 'CINSIYET', 'YAS'], how='left')
# all.loc[all['GELIR'].isna(), 'GELIR'] = all['gelir_mean']

all = downcast_df_int_columns(all)
all = downcast_df_float_columns(all)

target_feature = 'ARTIS_DURUMU'
all["is_train"] = all[target_feature].notnull()

features = [
    #'POLICY_ID',
    "OFFICE_ID",
    "SIGORTA_TIP",
    "SOZLESME_KOKENI",
    "SOZLESME_KOKENI_DETAY",
    #"sozlesme-koken",
    "KAPSAM_TIPI",
    "KAPSAM_GRUBU",
    "DAGITIM_KANALI",
    "POLICE_SEHIR",
    #"POLICE_COUNTRY",
    "CINSIYET",
    "UYRUK",
    "MEMLEKET",
    "MESLEK",
    "MESLEK_KIRILIM",
    #"meslek-krlm",
    "MUSTERI_SEGMENTI",
    "YATIRIM_KARAKTERI",
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
    # "EYLUL_ODENEN_TUTAR",
    # "EYLUL_VADE_TUTARI",
    # "EKIM_ODENEN_TUTAR",
    # "EKIM_VADE_TUTARI",
    # "KASIM_ODENEN_TUTAR",
    # "KASIM_VADE_TUTARI",
    # "ARALIK_ODENEN_TUTAR",
    "ARALIK_VADE_TUTARI",
    "SENE_BASI_HESAP_DEGERI",
    "SENE_SONU_HESAP_DEGERI",
    # "sb_0",
    # "ss_0",
    "BASLANGIC_ay",
    "BASLANGIC_yil",
    "month_passed",
    "YAS",
    "BASLANGIC_YAS",
    "under_months",
    "zero_months",
    "under_months_all_year",
    "zero_months_all_year",
    "well_paid_customer",
    #"odenen_vade_tutar",
    #"hesap_degisimi_faiz",
    "ratio_increase",
    "vade_up",
    "vade_down",
    "hesap_degisimi",
    "last_change_months_ago",
    "vade_not_changed",
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
    'OFFICE_ID',
    'SIGORTA_TIP',
    'SOZLESME_KOKENI',
    'SOZLESME_KOKENI_DETAY',
    "sb_0",
    "ss_0",
    'KAPSAM_TIPI',
    'KAPSAM_GRUBU',
    'DAGITIM_KANALI',
    'POLICE_SEHIR',
    'CINSIYET',
    'UYRUK',
    'MEMLEKET',
    'MESLEK',
    'MESLEK_KIRILIM',
    'MUSTERI_SEGMENTI',
    'YATIRIM_KARAKTERI',
    'MEDENI_HAL',
    'EGITIM_DURUM',
    "under_months_all_year",
    "zero_months_all_year",
    "well_paid_customer",
    "vade_up",
    "vade_down",
    "vade_not_changed",
    "up_first_quarter",
]
numeric_cols = list(set(all[features].columns) - set(categorical_features))

for col_name in categorical_features:
    series = all[col_name]
    label_encoder = LabelEncoder()
    all[col_name] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    )


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

        model.fit(
            tr_x,
            tr_y,
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
        "iterations": 3000,
        "scale_pos_weight": 3.471475485069041,
        "objective": "Logloss",
        "eval_metric": "F1",
        "task_type": "GPU",
        #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        # "depth": trial.suggest_int("depth", 5, 10),
        "depth": 9,
        "boosting_type": "Plain",
        "bootstrap_type": "Bernoulli",
        "random_state": trial.suggest_categorical("random_state", [i for i in range(1000)]),
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        # param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        param["subsample"] = 0.993
    param["random_state"] = 777
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

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=100,
            verbose=100,
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
    lgb_params = {'reg_alpha': 2.0604665531147646, 'reg_lambda': 0.0508941398294875, 'colsample_bytree': 0.8, 'subsample': 0.7, 'learning_rate': 0.05340773677228382, 'max_depth': 15, 'num_leaves': 196, 'min_child_samples': 87}
    cat_params = {'scale_pos_weight': 3.471475485069041, 'depth': 9, 'bootstrap_type': 'Bernoulli', 'subsample': 0.9931401642932501}
    xgb_params = {'reg_lambda': 0.2716863446664029, 'reg_alpha': 32.796648528916805, 'colsample_bytree': 0.9, 'subsample': 1.0, 'learning_rate': 0.02987792485738012, 'n_estimators': 1500, 'max_depth': 7, 'min_child_weight': 2.7489363264568705}
lgb_params["metric"] = "custom"
lgb_params["n_estimators"] = 1000
lgb_params["random_state"] = 61
lgb_params["scale_pos_weight"] = 3.471475485069041
xgb_params["tree_method"] = "gpu_hist"
xgb_params["random_state"] = 61
cat_params["objective"] = "Logloss"
cat_params["eval_metric"] = "F1"
cat_params["task_type"] = "GPU"
cat_params["random_state"] = 61
cat_params["iterations"] = 1000
cat_params["verbose"] = False

if model_type == "cat":
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

        model.fit(
            tr_x,
            tr_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=400,
            verbose=100,
        )

        if plot_importance:
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
            early_stopping_rounds=100,
            verbose=100,
        )

        if plot_importance:
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
            early_stopping_rounds=100,
            verbose=100,
        )

        if plot_importance:
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
if model_type == "stacking":

    for train_ind, val_ind in skf.split(train_x, train_y):
        print(f"******* Fold {ix} ******* ")
        model_xgb = XGBClassifier(**xgb_params)
        model_lgb = LGBMClassifier(**lgb_params)
        model_cat = CatBoostClassifier(**cat_params)

        model = VotingClassifier(estimators = [('xgb', model_xgb), ('lgb', model_lgb), ('cat', model_cat)], voting='soft', verbose=True)

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
print(f"Val F-1 Score: {val_score}")
print(f"Val Accuracy Score: {acc_score}")


sample_submission[target_feature] = y_test
sample_submission.loc[sample_submission[target_feature] >= 0.5, target_feature] = 1
sample_submission.loc[sample_submission[target_feature] < 0.5, target_feature] = 0
sample_submission[target_feature] = sample_submission[target_feature].astype(int)
sample_submission = pd.merge(sample_submission, all[['POLICY_ID', 'wide_dup_mean', 'target_overwrite']], on='POLICY_ID')
sample_submission.loc[sample_submission['wide_dup_mean'] == 0, target_feature] = 0
sample_submission.loc[sample_submission['wide_dup_mean'] > 0.7, target_feature] = 1
sample_submission.loc[sample_submission['target_overwrite'].notna(), target_feature] = sample_submission['target_overwrite']
print(f"Submission mean: {sample_submission[target_feature].mean()}")
sample_submission[['POLICY_ID', target_feature]].to_csv(f"submissions/submission_{model_type}_{trial}_{val_score}.csv", sep=",", index=False)

with open("submissions/submission_notes.csv", "a") as file_object:
    # Append 'hello' at the end of file
    file_object.write(f"\nsubmission_{model_type}_{trial},{val_score},{acc_score},{sample_submission[target_feature].mean()},{xgb_params['scale_pos_weight']}")