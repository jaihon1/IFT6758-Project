import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from comet_ml import Experiment


def create_experiment(parameters):
    """
    Create experiment on comet and returns it
    Args:
        parameters: the parameters to log for the experiment

    Returns: the experiment

    """
    experiment = Experiment(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="ift6758-project",
        workspace="jaihon",
    )
    experiment.log_parameters(parameters)
    return experiment


def plot_roc_curve(pred_probs, true_y, linestyles, labels, name_file=None):
    """
    Create the roc curve plot
    Args:
        probas: list of arrays, each array contains de probability of an event to be a goal
        actual_y: list of arrays, each array contains the actual label of an event
        linestyles: list of strings, the linestyle to use for each curve
        labels: list of strings, labels to give in the legend
        name_file: str, if None, the figure isn't saved in a file, if it's a string, this is the name of the file.

    Returns:

    """
    sns.set_theme()
    plt.grid(True)
    for proba, y, linestyle, label in zip(pred_probs, true_y, linestyles, labels):
        score = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        plt.plot(fpr, tpr, linestyle=linestyle, label=label+f' (area={score:.2f})')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    if False:
        plt.savefig(name_file)
    plt.show()


def create_percentile_model(proba, actual_y):
    """
    Create the shot probability model percentile
    Args:
        proba: array of probabilities
        actual_y: actual label of each event

    Returns: the array of percentile, the array of probabilities that correspond to each percentile and
    a dataframe saying to which bin of percentile an event corresponds

    """
    percentile = np.arange(0, 102, 2)
    percentile_pred = np.percentile(proba, percentile)
    percentile_pred = np.concatenate([[0], percentile_pred])
    percentile_pred = np.unique(percentile_pred)

    y_valid_df = pd.DataFrame(actual_y)

    y_valid_df['bins_percentile'] = pd.cut(proba, percentile_pred, include_lowest=True)

    return percentile, percentile_pred, y_valid_df


def plot_goal_rate(probas, actual_y, labels, name_file=None):
    """
    Create the goal rate plot
    Args:
        probas: list of arrays, each array contains de probability of an event to be a goal
        actual_y: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        name_file: str, if None, the figure isn't saved in a file, if it's a string, this is the name of the file.

    Returns:

    """
    sns.set_theme()
    for proba, y, label in zip(probas, actual_y, labels):
        percentile, percentile_pred, y_valid_df = create_percentile_model(proba, y)
        bins = np.linspace(0, 100, len(y_valid_df['bins_percentile'].unique()))[1:]

        goal_rate_by_percentile = y_valid_df.groupby(by=['bins_percentile']).apply(lambda g: g['is_goal'].sum()/len(g))

        g = sns.lineplot(x=bins, y=goal_rate_by_percentile[1:]*100, label=label)
        ax = g.axes
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Goals / (Shots + Goals)')
    if False:
        plt.savefig(name_file)
    plt.show()


def plot_cumulative_sum(probas, actual_y, labels, name_file=None):
    """
    Create cumulative sum of goals plot
    Args:
        probas: list of arrays, each array contains de probability of an event to be a goal
        actual_y: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        name_file: str, if None, the figure isn't saved in a file, if it's a string, this is the name of the file.

    Returns:

    """
    sns.set_theme()
    for proba, y, label in zip(probas, actual_y, labels):
        percentile, percentile_pred, y_valid_df = create_percentile_model(proba, y)
        bins = np.linspace(0, 100, len(y_valid_df['bins_percentile'].unique()))[1:]
        total_number_goal = (y == 1).sum()
        sum_goals_by_percentile = y_valid_df.groupby(by='bins_percentile').apply(lambda g: g['is_goal'].sum()/total_number_goal)
        cum_sum_goals = sum_goals_by_percentile[::-1].cumsum(axis=0)[::-1]

        g = sns.lineplot(x=bins, y=cum_sum_goals[1:]*100, label=label)
        ax = g.axes
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.xlim(100, 0)
    plt.ylim(0, 100)
    plt.xlabel('Shot probability model percentile')
    plt.ylabel('Proportion')
    if False:
        plt.savefig(name_file)
    plt.show()


def plot_calibration(probas, actual_y, labels, name_file=None):
    """
    Create calibration plot
    Args:
        probas: list of arrays, each array contains de probability of an event to be a goal
        actual_y: list of arrays, each array contains the actual label of an event
        labels: list of strings, labels to give in the legend
        name_file: str, if None, the figure isn't saved in a file, if it's a string, this is the name of the file.

    Returns:

    """
    sns.set_theme()
    fig = plt.figure()
    ax = plt.axes()
    for proba, y, label in zip(probas, actual_y, labels):
        disp = CalibrationDisplay.from_predictions(y, proba, n_bins=25, ax=ax, name=label, ref_line=False)
    plt.xlim(0, 1)
    plt.legend(loc=9)
    if False:
        plt.savefig(name_file)
    plt.show()


def prep_data(data_train, selected_features, categorical_features, norm=None):
    """
    Preparation of data for training models. Transforms categorical data into one-hot encoding and
    standardizes numerical data.
    Args:
        data_train: pd.DataFrame, dataframe containing the training data
        selected_features: list of strings, the names of the features to select for training the model
        categorical_features: list of strings, the names of the features that are categorical
        norm: list of string, the name of the features to standardize

    Returns: pd.DrataFrame, the preprocessed data

    """
    data = data_train[selected_features]

    # Drop rows with NaN values
    data = data.dropna(subset=selected_features)

    if norm is not None:
        scaler = StandardScaler()
        data[norm] = scaler.fit_transform(data[norm])

    # Encoding categorical features into a one-hot encoding
    for feature in categorical_features:
        one_hot_encoder = OneHotEncoder(sparse=False)
        encoding_df = data[[feature]]

        one_hot_encoder.fit(encoding_df)

        df_encoded = one_hot_encoder.transform(encoding_df)

        df_encoded = pd.DataFrame(data=df_encoded, columns=one_hot_encoder.categories_)

        # Drop original feature and add encoded features
        data.drop(columns=[feature], inplace=True)

        data = pd.concat([data.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    return data


def plot_relation(x, ys, labels, xlabel, ylabel, name_file=None):
    sns.set_theme()
    for y, label in zip(ys,labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if name_file:
        plt.savefig(name_file)
    plt.show()
