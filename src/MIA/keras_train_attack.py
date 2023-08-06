from utils.seed import seed_everything
from utils.load_config import load_config
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# get metric and train, test support
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# get classifier models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb
# import lightgbm as lgb
from catboost import CatBoostClassifier
import os 
from os.path import join
from tensorflow.keras.models import load_model


# load config
config_path = "keras_config.yaml"
CFG = load_config("CFG", config_path)
CFG_ATTACK = load_config("CFG_ATTACK", config_path)

# seed for future replication
seed_everything(CFG.seed)



X, y = pd.DataFrame(), pd.DataFrame()

# for i in range(CFG.num_shadow_models) : 
for i in range(3) : 

    this_shadow_model_path = join(CFG_ATTACK.shadow_models_path, f"shadow_model_{i}")
    if not os.path.exists(this_shadow_model_path):
        os.makedirs(this_shadow_model_path)
    
    model_path = join(this_shadow_model_path, 'best_model.h5')
    attack_dset_path = join(this_shadow_model_path, f'attack_dset_shadow_{i}.csv')

    model = load_model(model_path)
    df_shadow = pd.read_csv(attack_dset_path)

    tmp_y = df_shadow["is_member"]
    tmp_X = df_shadow.drop(["is_member"], axis=1)

    X = pd.concat([X, tmp_X])
    y = pd.concat([y, tmp_y])

# to numpy
X = X.to_numpy()
y = y.to_numpy()


print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CFG_ATTACK.test_size, random_state=CFG.seed
)

print("x_train shape:", X_train.shape)
print("x_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# model = xgb.XGBClassifier(n_estimators=CFG_ATTACK.n_estimators, n_jobs=-1, random_state=CFG.seed)
# model = lgb.LGBMClassifier(n_estimators=CFG_ATTACK.n_estimators, n_jobs=-1, random_state=CFG.seed)
model = CatBoostClassifier(
    iterations=CFG_ATTACK.train_epoch,
    depth=2,
    learning_rate=CFG_ATTACK.learning_rate,
    loss_function="Logloss",
    verbose=True,
)  # https://catboost.ai/en/docs/concepts/loss-functions-classification

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
precision, recall, f1_score, _ = precision_recall_fscore_support(
    y_test, model.predict(X_test), average="binary"
)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1_score:", f1_score)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
print("mean fpr:", np.mean(fpr))
print("mean tpr:", np.mean(tpr))

# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate
# plot and save roc curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label="MIA ROC curve (area = %0.2f)" % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.show()
plt.savefig("roc.png")

if not os.path.exists(CFG_ATTACK.attack_model_path):
    os.makedirs(CFG_ATTACK.attack_model_path)

save_path = join(CFG_ATTACK.attack_model_path, "catboost_1")
model.save_model(save_path)

