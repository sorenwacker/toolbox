import missingno as msno
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import shap


from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GroupKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    plot_confusion_matrix,
    mean_squared_error,
)
from sklearn.decomposition import PCA


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy

from matplotlib import pyplot as pl


def hierarchical_clustering(
    df,
    vmin=None,
    vmax=None,
    show="scaled",
    figsize=(8, 8),
    top_height=2,
    left_width=2,
    xmaxticks=None,
    ymaxticks=None,
    metric="euclidean",
    cmap=None,
    scaling="standard",
    scaling_kws=None,
):
    """based on heatmap function from
    http://nbviewer.ipython.org/github/herrfz/dataanalysis/
    blob/master/week3/svd_pca.ipynb
    Generates a heatmap from the input matrix.
    """

    df_orig = df.copy()
    df = df.copy()

    if scaling is not None:
        if scaling_kws is None:
            scaling_kws = {}
        df = scale_dataframe(df, how=scaling, **scaling_kws)

    # cm = pl.cm
    # cmap = cm.rainbow(np.linspace(0, 0, 1))
    # hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])

    # Subplot sizes
    total_width, total_height = figsize

    main_h = 1 - (top_height / total_height)
    main_w = 1 - (left_width / total_width)

    gap_x = 0.1 / total_width
    gap_y = 0.1 / total_height

    left_h = main_h
    left_w = 1 - main_w

    top_h = 1 - main_h
    top_w = main_w

    ydim, xdim = df.shape

    if xmaxticks is None:
        xmaxticks = int(5 * main_w * total_width)
    if ymaxticks is None:
        ymaxticks = int(5 * main_h * total_height)

    dm = df.fillna(0).values

    D1 = squareform(pdist(dm, metric=metric))
    D2 = squareform(pdist(dm.T, metric=metric))

    fig = pl.figure(figsize=figsize)
    fig.set_tight_layout(False)

    # add left dendrogram
    ax1 = fig.add_axes([0, 0, left_w - gap_x, left_h], frameon=False)
    Y = linkage(D1, method="complete")
    Z1 = dendrogram(Y, orientation="left", color_threshold=0, above_threshold_color="k")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add top dendrogram
    ax2 = fig.add_axes([left_w, main_h + gap_y, top_w, top_h - gap_y], frameon=False)
    Y = linkage(D2, method="complete")
    Z2 = dendrogram(Y, color_threshold=0, above_threshold_color="k")
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = fig.add_axes([left_w, 0, main_w, main_h])
    idx1 = Z1["leaves"]
    idx2 = Z2["leaves"]

    if show == "scaled":
        D = dm[idx1, :]
        D = D[:, idx2]
    if show == "original":
        D = df_orig.iloc[idx1, :]
        D = D.iloc[:, idx2].values

    if cmap is None:
        cmap = "hot"
    fig = axmatrix.matshow(D[::-1], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    ax = pl.gca()
    ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()

    clustered = df_orig.iloc[Z1["leaves"][::-1], Z2["leaves"]]

    ndx_y = np.linspace(0, len(clustered.index) - 1, ymaxticks)
    ndx_x = np.linspace(0, len(clustered.columns) - 1, xmaxticks)
    ndx_y = [int(i) for i in ndx_y]
    ndx_x = [int(i) for i in ndx_x]

    _ = pl.yticks(ndx_y, clustered.iloc[ndx_y].index)
    _ = pl.xticks(ndx_x, clustered.columns[ndx_x], rotation=90)

    return clustered, fig


def scale_dataframe(df, how="standard", **kwargs):
    if how == "standard":
        scaler = StandardScaler
    elif how == "robust":
        scaler = RobustScaler
    df = df.copy()
    df.loc[:, :] = scaler(**kwargs).fit_transform(df)
    return df


def plot_missing_values(df, kind="matrix"):
    if kind == "matrix":
        return msno.matrix(df)
    if kind == "bar":
        return msno.bar(df)
    if kind == "heatmap":
        msno.heatmap(df)


def knn_score(df, var_names, tgt_name, **params):
    """
    Calculate the knn accuracy of a clustered dataset with known labels.
    """

    df = df.copy().reset_index(drop=True)

    X = df[var_names]
    y = df[tgt_name]

    kfold = KFold(n_splits=len(X))

    prediction = y.copy().to_frame()
    for ndx_train, ndx_valid in kfold.split(X):
        X_train, X_valid = X.loc[ndx_train], X.loc[ndx_valid]
        y_train, y_valid = y.loc[ndx_train], y.loc[ndx_valid]

        clf = KNeighborsClassifier(**params)
        clf.fit(X_train, y_train)
        prd = clf.predict(X_valid)

        prediction.loc[ndx_valid, "Prediction"] = prd

    classes = y.value_counts().index
    conf_ma = confusion_matrix(
        prediction[tgt_name], prediction["Prediction"], labels=classes
    )
    df_coma = pd.DataFrame(conf_ma, index=classes, columns=classes)
    accuracy = accuracy_score(prediction[tgt_name], prediction["Prediction"])

    return accuracy, df_coma, prediction


def sklearn_cv_classification(
    X, y, base_model, params={}, X_test=None, n_folds=5, seeds=None
):

    if seeds is None:
        seeds = [1]

    losses = []
    predic = None

    cv_predictions = X[[]].copy()
    cv_predictions["CV-pred"] = None
    cv_predictions["y"] = y

    n_models = len(seeds) * n_folds

    if X_test is not None:
        predictions = X_test[[]].copy()

    for seed in seeds:

        assert isinstance(seed, int)

        params["seed"] = seed

        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for _n, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):

            print("=" * 22 + f" Fold {_n:3d} " + "=" * 22)

            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y[ndx_train], y[ndx_valid]

            _model = base_model()
            _model.fit(_X_train, _y_train)

            _pred = _model.predict(_X_valid)
            _loss = accuracy_score(_y_valid, _pred)

            cv_predictions.iloc[ndx_valid, 0] = _pred

            print(f"Fold {_n} accuracy: {_loss}")

            losses.append(_loss)

            if X_test is not None:
                _pred_test = _model.predict(X_test)
                predictions[f"seed-{seed}-fold-{_n}"] = _pred_test.astype(int)

    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_predictions = cv_predictions.astype(int)
    print(f"CV-loss {loss_mean}+/-{loss_std}")

    return loss_mean, loss_std, cv_predictions, predictions


def sklearn_cv_binary_clf_roc(
    X,
    y,
    base_model,
    params={},
    X_test=None,
    n_folds=5,
    seeds=None,
    to_numpy=False,
    metric=None,
    fit_kws=None,
    framework=None,
):
    """
    sklearn model compatible binary classification procedure with
    probability estimatoin and ROC metric with cross-validation
    and multiple random seeds.
    """
    if seeds is None:
        seeds = [1]
    losses = []
    predic = None
    cv_predictions = X[[]].copy()
    cv_predictions["y"] = y
    if metric is None:
        metric = roc_auc_score
    if fit_kws is None:
        fit_kws = {}
    n_models = len(seeds) * n_folds
    if X_test is not None:
        predictions = X_test[[]].copy()
    else:
        predictions = None
    for n_seed, seed in enumerate(seeds):
        print("+" * 22 + f" Seed {n_seed:2d} " + "+" * 23)
        assert isinstance(seed, int)
        params["seed"] = seed
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for n_fold, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):
            print("=" * 22 + f" Fold {n_fold:2d} " + "=" * 23)
            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y.iloc[ndx_train], y.iloc[ndx_valid]
            if to_numpy:
                _X_train, _X_valid, _y_train, _y_valid = (
                    _X_train.values,
                    _X_valid.values,
                    _y_train.values,
                    _y_valid.values,
                )
            _model = base_model(**params)
            if framework == "lgbm":
                fit_kws["eval_set"] = [(_X_train, _y_train), (_X_valid, _y_valid)]
                fit_kws["eval_metric"] = "auc"
                fit_kws["eval_names"] = ["Train", "Valid"]
            _model.fit(_X_train, _y_train, **fit_kws)
            _pred = _model.predict_proba(_X_valid)[:, 1]
            _loss = metric(_y_valid, _pred)
            print(f"Fold {n_fold}: {_loss:1.4f}")
            cv_predictions.loc[ndx_valid, f"cv-pred-{n_seed}"] = _pred
            losses.append(_loss)
            if X_test is not None:
                _pred_test = _model.predict_proba(X_test.values)[:, 1]
                predictions[f"pred-s-{seed}"] = _pred_test
    cv_predictions["cv-pred"] = cv_predictions.filter(regex="cv-pred").mean(axis=1)
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_loss = metric(cv_predictions["y"].values, cv_predictions["cv-pred"].values)
    if X_test is not None:
        predictions["pred"] = predictions.filter(regex="pred").mean(axis=1)
    print(f"Avg-CV: {loss_mean:1.4f}+/-{loss_std:1.4f}; Final-CV: {cv_loss:1.4f}")
    return loss_mean, loss_std, cv_predictions, predictions


def tabnet_cv_classification(
    X,
    y,
    base_model,
    params={},
    X_test=None,
    n_folds=5,
    seeds=None,
    to_numpy=False,
    fit_kws=None,
    metric=None,
):
    if seeds is None:
        seeds = [1]
    if fit_kws is None:
        fit_kws = {}
    if metric is None:
        metric = roc_auc_score
    losses = []
    predic = None
    cv_predictions = X[[]].copy()
    cv_predictions["y"] = y
    n_models = len(seeds) * n_folds
    if X_test is not None:
        predictions = X_test[[]].copy()
    else:
        predictions = None
    for n_seed, seed in enumerate(seeds):
        assert isinstance(seed, int)
        print("+" * 22 + f" Seed {n_seed:2d} " + "+" * 23)
        params["seed"] = seed
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for n_fold, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):
            print("=" * 22 + f" Fold {n_fold:2d} " + "=" * 23)
            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y.iloc[ndx_train], y.iloc[ndx_valid]
            if to_numpy:
                _X_train, _X_valid, _y_train, _y_valid = (
                    _X_train.values,
                    _X_valid.values,
                    _y_train.values,
                    _y_valid.values,
                )
            _model = base_model(seed=seed)
            _model.fit(
                _X_train,
                _y_train,
                eval_set=[(_X_train, _y_train), (_X_valid, _y_valid)],
                eval_name=["train", "valid"],
                **fit_kws,
            )
            _pred = _model.predict_proba(_X_valid)[:, 1]
            _loss = metric(_y_valid, _pred)
            cv_predictions.loc[ndx_valid, f"cv-pred-{n_seed}"] = _pred
            print(f"Fold {n_fold}: {_loss}")
            losses.append(_loss)
            if X_test is not None:
                _pred_test = _model.predict_proba(X_test.values)[:, 1]
                predictions[f"pred-s-{seed}"] = _pred_test
            del _model
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_predictions["cv-pred"] = cv_predictions.filter(regex="cv-pred").mean(axis=1)
    cv_loss = metric(cv_predictions["y"].values, cv_predictions["cv-pred"].values)
    cv_predictions = cv_predictions.astype(float)
    if X_test is not None:
        predictions["pred"] = predictions.filter(regex="pred").mean(axis=1)
    print(f"Avg-CV: {loss_mean:1.4f}+/-{loss_std:1.4f}; Final-CV: {cv_loss:1.4f}")
    return loss_mean, loss_std, cv_predictions, predictions


def softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything
    """
    from scipy.special import logsumexp

    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)


def decode_prediction(df, encoder):
    return df.apply(encoder.inverse_transform)


def remove_features_with_anti_target(
    df, features, anti_target="is_test", target_auc=0.6
):

    features = copy(features)

    auc = 1

    history = dict(n_features=[], auc=[], features=[])

    while len(features) > 0:
        anti_target = "is_test"

        dtrain = xgb.DMatrix(df[features], df[anti_target])

        params = {
            "objective": "binary:logistic",
            "max_depth": 2,
        }

        cv = xgb.cv(dtrain=dtrain, params=params, metrics=["auc"])

        auc = cv["train-auc-mean"].mean()

        history["auc"].append(auc)
        history["n_features"].append(len(features))
        history["features"].append(copy(features))

        if auc < target_auc:
            break

        params["metric"] = "logloss"

        model = xgb.XGBClassifier(params=params, verbosity=0)
        model.fit(df[features], df[anti_target])

        fi = pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_}
        )
        fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)

        best_feature = fi.loc[0, "Feature"]
        features.remove(best_feature)

    if len(features) == 1:
        logging.warning(
            "Target AUC could not be reached, returning last remaining feature."
        )

    return pd.DataFrame(history)


def quick_pca(df, n_components=2, labels=None, plot=True, scale=True, **plot_kws):
    df = df.copy()
    if scale:
        scaler = StandardScaler()
        df.loc[:, :] = scaler.fit_transform(df)
    pca = PCA(n_components)
    res = pd.DataFrame(pca.fit_transform(df))
    res.columns = res.columns.values + 1
    res = res.add_prefix("PC-")
    if labels is not None:
        res["label"] = list(labels)
    if plot:
        sns.pairplot(res, hue="label" if labels is not None else None, **plot_kws)
    res.index = df.index
    return res


def pycaret_score_threshold_analysis(pycaret_prediction):

    score_thresholds = np.arange(0.5, 0.95, 0.01)
    accs = []
    ns = []

    for st in score_thresholds:
        tmp = pycaret_prediction[pycaret_prediction.Score > st]
        score = balanced_accuracy_score(tmp.DEATH_IND, tmp.Label)
        accs.append(score)
        ns.append(len(tmp) / len(pycaret_prediction))

    plot(score_thresholds, accs, color="C0")
    ylabel("Balanced accuracy", color="C0")
    xlabel("Score threshold")
    yticks(color="C0")

    ax1 = gca()
    ax2 = ax1.twinx()

    plot(score_thresholds, ns, color="C2")

    ylabel("Fraction of samples", color="C2")
    yticks(color="C2")
    grid()

    title("Score theshold analysis")


class ShapAnalysis:
    def __init__(self, model, df):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        self.df = df
        self.shap_values = shap_values
        self.instance_names = df.index.to_list()
        self.feature_names = df.columns.to_list()
        #
        # self.df_shap = pd.DataFrame(
        #    shap_values.values,
        #    columns=df.columns,
        #    index=df.index
        # )

    def waterfall(self, i, **kwargs):
        shap_values = self.shap_values
        self._base_values = shap_values[i][0].base_values
        self._values = shap_values[i].values
        shap_object = shap.Explanation(
            base_values=self._base_values,
            values=self._values,
            feature_names=self.feature_names,
            # instance_names = self._instance_names,
            data=shap_values[i].data,
        )
        shap.plots.waterfall(shap_object, **kwargs)

    def summary(self, df=None, **kwargs):
        shap.summary_plot(self.shap_values, df if df is not None else self.df, **kwargs)

    def bar(self, **kwargs):
        shap.plots.bar(self.shap_values, **kwargs)
        for ax in plt.gcf().axes:
            for ch in ax.get_children():
                try:
                    ch.set_color("0.3")
                except:
                    break
