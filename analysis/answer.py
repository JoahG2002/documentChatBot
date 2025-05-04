import sys

import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from numpy.typing import NDArray
from scipy.stats import skew, kurtosis
from pandas.core.generic import NDFrame
from statsmodels.base.model import Model
from statsmodels.api import add_constant
from pandas._libs.properties import AxisProperty
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats._mstats_basic import ttest_ind, pearsonr
from statsmodels.regression.linear_model import OLS, RegressionResults

from app.constant.constant import directories, csv_


class ResearchQuestion1:
    __slots__: tuple[str, ...] = ("__llm_experiment", "__metrics", "__t_test_results", "__regression_results")

    def __init__(self, llm_experiment: pd.DataFrame) -> None:
        self.__llm_experiment: pd.DataFrame = llm_experiment
        self.__metrics: list[str] = [
            "answerRelevanceScore", "answerFaithfulnessScore", "contextRelevanceScore", "practicalityScore"
        ]
        self.__t_test_results: dict[str, dict[str, float]] = {}
        self.__regression_results: dict[str, dict[str, float]] = {}

    def answer(self) -> None:
        """
        Runs the complete analysis for page size impact with only two sizes (512 and 1000).
        """
        self._print_info()

        self._conduct_comparative_analysis()

        self._conduct_statistical_test()

        self._conduct_regression_analysis()

        self._print_conclusions()

    def _print_info(self) -> None:
        """
        Writes basic statistics about the two-page-size groups to the standard output stream.
        """
        sys.stdout.write("===GENERAL INFORMATION 'PAGE SIZE'===\n\n")

        page_counts: pd.Series = self.__llm_experiment["pdfPageSize"].value_counts()
        sys.stdout.write(f"Number of samples per page size:\n{page_counts}\n")

        sys.stdout.write(
            f"\nSummary statistics for page size 512:\n"
            f"{self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 512][self.__metrics].describe()}\n\n"
        )
        sys.stdout.write(
            f"\nSummary statistics for page size 1000:\n"
            f"{self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 1_000][self.__metrics].describe()}\n\n"
        )

        self._plot_metrics_distribution()

    def _plot_metrics_distribution(self) -> None:
        """
        Plots the distribution of metrics by page size.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(self.__metrics):
            sns.boxplot(x="pdfPageSize", y=metric, data=self.__llm_experiment, ax=axes[i])

            axes[i].set_title(f"Distribution of {metric} by Page Size")
            axes[i].set_xlabel("Page Size (characters)")
            axes[i].set_ylabel(metric)

        plt.tight_layout()

        plt.savefig(f"{directories.ANALYSIS_PLOTS}/metrics_distribution_by_page_size.png")

        plt.show()

        plt.close()

    def _conduct_comparative_analysis(self) -> None:
        """
        Compares the metrics between the two-page-sizes.
        """
        metric_means: pd.DataFrame = self.__llm_experiment.groupby("pdfPageSize")[self.__metrics].mean().reset_index()

        first_512_row: pd.DataFrame = metric_means[metric_means["pdfPageSize"] == 512].iloc[0]
        first_1000_row: pd.DataFrame = metric_means[metric_means["pdfPageSize"] == 1_000].iloc[0]

        percentage_changes: dict[str, float] = {}

        for metric in self.__metrics:
            percentage_changes[metric] = (
                ((first_1000_row[metric] - first_512_row[metric]) / first_512_row[metric]) * 100
            )

        sys.stdout.write("\nPercentage changes from 512 to 1000 characters:\n")

        for metric, percentage_change in percentage_changes.items():
            direction: str = "increase" if (percentage_change > 0) else "decrease"

            sys.stdout.write(f"{metric}: {abs(percentage_change):.5f}% {direction}\n")

        self._plot_comparison(metric_means)

    def _plot_comparison(self, metric_means: pd.DataFrame) -> None:
        """
        Plots a direct comparison between the two-page sizes.
        """
        plt.figure(figsize=(12, 6))

        bar_width: float = 0.35
        index: NDArray[np.int64] = np.arange(len(self.__metrics))

        plt.bar(
            index,
            metric_means[metric_means["pdfPageSize"] == 512][self.__metrics].values[0],
            bar_width,
            label="512 characters"
        )
        plt.bar(
            (index + bar_width),
            metric_means[metric_means["pdfPageSize"] == 1000][self.__metrics].values[0],
            bar_width,
            label="1000 characters"
        )

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Comparison of Metrics by Page Size")
        plt.xticks((index + (bar_width / 2.0)), self.__metrics, rotation=45)
        plt.legend()

        plt.tight_layout()
        
        plt.savefig(f"{directories.ANALYSIS_PLOTS}/metrics_comparison_by_page_size.png")

        plt.show()

        plt.close()

    def _conduct_statistical_test(self) -> None:
        """
        Performs a statistical T-tests to compare the two-page sizes.
        """
        sys.stdout.write("\n==STATISTICAL COMPARISON (T-TEST)==\n\n")

        for metric in self.__metrics:
            _512_group: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 512][metric]
            _1000_group: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 1000][metric]

            t_statistic, p_value = ttest_ind(a=_512_group, b=_1000_group, equal_var=False)

            self.__t_test_results[metric] = {"t_statistic": t_statistic, "p_value": p_value}

            mean_1000_group: float = _1000_group.mean()
            mean_512_group: float = _512_group.mean()

            effect_size: float = (
                (mean_1000_group - mean_512_group) /
                np.sqrt(
                    ((_512_group.std() ** 2.0) + (_1000_group.std() ** 2.0))
                    / 2
                )
            )

            sys.stdout.write(
                f"\n{metric}:\n"
                f"  512 chars (n={len(_512_group)}): mean = {mean_512_group:.4f}, std = {_512_group.std():.4f}\n"
                f"  1000 chars (n={len(_1000_group)}): mean = {mean_1000_group:.4f}, std = {_1000_group.std():.4f}\n"
                f"  t-statistic: {t_statistic}\n"
                f"  p-value: {p_value}\n"
                f"  Cohen's d effect size: {effect_size}\n"
                f"  Significance: {"Significant" if (p_value < csv_.ALPHA) else "Not significant"} at α={csv_.ALPHA}\n"
            )

            if (p_value < csv_.ALPHA):
                effect_readable: str = "large"

                absolute_effect_size: float = abs(effect_size)

                if (absolute_effect_size) > 0.8:
                    if (absolute_effect_size > 0.5):
                        effect_readable = "medium"
                    else:
                        effect_readable = "small"

                direction: str = "higher" if (mean_1000_group > mean_512_group) else "lower"

                sys.stdout.write(
                    f"\n\tInterpretation: {metric} is {direction} with 1000 characters, with a {effect_readable} effect size.\n"
                    f"{'-' * 85}\n"
                )
            else:
                sys.stdout.write(f"No significance found for {metric}\n{'-' * 85}\n")

    def _conduct_regression_analysis(self) -> None:
        """
        Conducts a regression analysis to control for other variables.
        """
        sys.stdout.write(f"\n{'-' * 85}\n==REGRESSION ANALYSIS (controlling for confounding variables)==\n\n")

        control_variables: list[str] = ["promptLength", "instructionLength", "queryLength"]

        self.__llm_experiment["is_page_size_1000"] = (self.__llm_experiment["pdfPageSize"] == 1_000).astype(np.int8)

        for metric in self.__metrics:
            formula_basic: str = f"{metric} ~ is_page_size_1000"
            model_basic: Model = smf.ols(formula=formula_basic, data=self.__llm_experiment).fit()

            formula_full: str = f"{metric} ~ is_page_size_1000 + " + " + ".join(control_variables)
            model_full: Model = smf.ols(formula=formula_full, data=self.__llm_experiment).fit()

            self.__regression_results[metric] = {
                "basic_model": {
                    "coefficient": model_basic.params["is_page_size_1000"],
                    "p_value": model_basic.pvalues["is_page_size_1000"],
                    "r_squared": model_basic.rsquared
                },
                "full_model": {
                    "coefficient": model_full.params["is_page_size_1000"],
                    "p_value": model_full.pvalues["is_page_size_1000"],
                    "r_squared": model_full.rsquared
                }
            }

            sys.stdout.write(
                f"\nn{'-' * 85}\nRegression for {metric}:\n"
                f"  Basic model (page size only):\n"
                f"    Page Size Coefficient: {model_basic.params["is_page_size_1000"]:.6f}\n"
                f"    Page Size p-value: {model_basic.pvalues["is_page_size_1000"]:.6f}\n"
                f"    R-squared: {model_basic.rsquared:.4f}\n"
    
                f"  Full model (with control variables):\n"
                f"    Page Size Coefficient: {model_full.params["is_page_size_1000"]:.6f}\n"
                f"    Page Size p-value: {model_full.pvalues["is_page_size_1000"]:.6f}\n"
                f"    R-squared: {model_full.rsquared:.4f}\n\n"
            )

            significant_change: bool = (model_basic.pvalues["is_page_size_1000"] < csv_.ALPHA) != (model_full.pvalues["is_page_size_1000"] < csv_.ALPHA)

            if (significant_change):
                sys.stdout.write(f"\tNote: Controlling for variables changed statistical significance!\n")

    def _print_conclusions(self) -> None:
        """
        Prints the research conclusions to the standards output stream.
        """
        sys.stdout.write("\n=== RESEARCH QUESTION CONCLUSIONS ===\n\n")
        sys.stdout.write(
            "Research Question: How does retrieval page size impact the relevance and completeness of technical requirement generation?\n\n"
        )

        significant_metrics = [metric for metric in self.__metrics if (self.__t_test_results[metric]["p_value"] < csv_.ALPHA)]

        if (significant_metrics):
            sys.stdout.write("\n==FINDINGS==:\n1. Significant differences found in the following metrics:\n\n")

            for metric in significant_metrics:
                _512_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 512][metric].mean()
                _1000_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 1000][metric].mean()

                direction: str = "higher" if (_512_group_mean > _1000_group_mean) else "lower"

                sys.stdout.write(
                    f"\t- {metric}: {direction} with 1000 characters (p={self.__t_test_results[metric]["p_value"]:.6f})"
                )
        else:
            sys.stdout.write(
                "\n==FINDINGS==:\n1. No statistically significant differences found between 512 and 1000 character page sizes.\n"
            )

        if (self.__regression_results):
            sys.stdout.write("\n2. After controlling for other factors:\n\n")

            significant_metrics_after_control: list[str] = [
                metric for metric in self.__metrics if (self.__regression_results[metric]["full_model"]["p_value"] < csv_.ALPHA)
            ]

            if (significant_metrics_after_control):
                for metric in significant_metrics_after_control:
                    coefficient: float = self.__regression_results[metric]["full_model"]["coefficient"]

                    direction: str = "increases" if (coefficient > 0.0) else "decreases"

                    sys.stdout.write(
                        f"\t- Larger page size {direction} {metric} (p={self.__regression_results[metric]["full_model"]["p_value"]:.4f})\n\n"
                    )
            else:
                sys.stdout.write("\t- No significant effects of page size after controlling for other variables.\n\n")

        sys.stdout.write("\nRecommendations:")

        if (significant_metrics):
            advantages_512: int = 0
            advantages_1000: int = 0

            for metric in significant_metrics:
                _512_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 512][metric].mean()
                _1000_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 1000][metric].mean()

                if (_1000_group_mean > _512_group_mean):
                    advantages_1000 += 1
                else:
                    advantages_512 += 1

            if (advantages_1000 > advantages_512):
                sys.stdout.write(
                    "1. Use the larger page size (1000 characters) for better overall technical requirement generation.\n"
                    f"\tThis size shows advantages in {advantages_1000} out of {len(significant_metrics)} significant metrics.\n"
                )
            elif (advantages_512 > advantages_1000):
                sys.stdout.write(
                    "1. Use the smaller page size (512 characters) for better overall technical requirement generation.\n"
                    f"\tThis size shows advantages in {advantages_512} out of {len(significant_metrics)} significant metrics.\n"
                )
            else:
                sys.stdout.write(
                    "1. Both page sizes (512 and 1000 characters) show mixed results.\t"
                    "The choice depends on which specific metrics are prioritized:\n"
                )

                for metric in (significant_metrics):
                    _512_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 512][metric].mean()
                    _1000_group_mean: float = self.__llm_experiment[self.__llm_experiment["pdfPageSize"] == 1000][metric].mean()

                    better_size: str = "1000" if (_1000_group_mean > _512_group_mean) else "512"

                    sys.stdout.write(f"\t- For {metric}: Use {better_size} characters\n")
        else:
            sys.stdout.write(
                "1. Based on this analysis, the page size (between 512 and 1000 characters) does not"
                "\tsignificantly impact the quality of technical requirement generation."
                "\tYou may choose either size based on other considerations (e.g., processing time).\n\n"
            )


class ResearchQuestion2:
    __slots__: tuple[str, ...] = (
        "__llm_experiment", "__relevant_features", "__performance_metrics", "__key_rag_variables"
    )

    def __init__(self, llm_experiment: pd.DataFrame) -> None:
        self.__llm_experiment: pd.DataFrame = llm_experiment

        self.__relevant_features: list[str] = [
            "largeLanguageModel", "modelName", "topKSimilarDocuments", "pdfPageSize", "promptLength", "instructionLength",
            "queryLength"
        ]

        self.__performance_metrics: list[str] = [
            "answerFaithfulnessScore", "answerRelevanceScore", "contextRelevanceScore", "sourcedCited",
            "practicalityScore", "llmResponseLength"
        ]

        self.__key_rag_variables: list[str] = ["queryLength", "topKSimilarDocuments", "pdfPageSize", "promptLength"]

    def answer(self) -> None:
        """
        Runs the full analysis for research question 2: How do different large language models respond to variable changes within a RAG system?
        """
        self._explore_data()

        self._compare_models()

        self._analyze_correlations()

        self._analyze_rag_variables_impact()

        self._summarize_findings()

    def _explore_data(self) -> None:
        """
        Explores the dataset to understand its structure and content.
        """
        """
        Explores the dataset to understand its structure and distribution of the two models.
        """
        sys.stdout.write("\n===1. DATA EXPLORATION===\n")

        sys.stdout.write(
            f"Dataset shape: {self.__llm_experiment.shape}"
            f"\nDistribution of models in dataset:\n{self.__llm_experiment["modelName"].value_counts()}\n"
            f"\nBasic statistics for RAG variables:\n{(self.__llm_experiment[self.__key_rag_variables].describe())}\n"
        )

        plt.figure(figsize=(10, 5))
        sns.countplot(x="modelName", data=self.__llm_experiment, palette=["#3498db", "#e74c3c"])
        plt.title("Distribution of Models in Dataset")
        plt.xlabel("Model Name")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.show()

        plt.figure(figsize=(20, 10))

        variable_count: int = self.__key_rag_variables.__len__()

        for i, variable in enumerate(self.__key_rag_variables):
            plt.subplot((variable_count // 2), (variable_count // 2), (i + 1))
            sns.histplot(self.__llm_experiment[variable], kde=True)
            plt.title(f"Distribution of {variable}")

        plt.tight_layout()

        plt.show()

    def _compare_models(self) -> None:
        """
        Compares the performance metrics between the two models.
        """
        sys.stdout.write("\n===2. MODEL COMPARISON ACROSS METRICS===\n")

        model_statistics: pd.DataFrame = self.__llm_experiment.groupby("modelName")[self.__performance_metrics].agg(["mean", "std", "count"])

        sys.stdout.write(f"\nPerformance metrics by model:\n{model_statistics}\n")

        plt.figure(figsize=(15, 10))

        for i, metric in enumerate(self.__performance_metrics):
            plt.subplot(2, 3, (i + 1))

            sns.boxplot(x="modelName", y=metric, data=self.__llm_experiment, palette=["#3498db", "#e74c3c"])
            plt.title(f"Comparison of {metric}")
            plt.xlabel("Model")
            plt.ylabel(metric)

            chat_gpt_4o_data = self.__llm_experiment[self.__llm_experiment["modelName"] == "gpt-4o"][metric]
            gemini_flash_data = self.__llm_experiment[self.__llm_experiment["modelName"] == "2.0 flash"][metric]

            t_statistic, p_value = ttest_ind(chat_gpt_4o_data, gemini_flash_data, equal_var=False)

            plt.annotate(
                f"p-value: {p_value:4f}",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )

        plt.tight_layout()
        plt.show()

        self._plot_radar_chart()

    def _plot_radar_chart(self) -> None:
        """
        Creates a radar chart comparing the two models across all metrics.
        """
        minimum_response_length: int = self.__llm_experiment["llmResponseLength"].min()
        maximum_response_length: int = self.__llm_experiment["llmResponseLength"].max()

        self.__llm_experiment["llmResponseLengthNormalised"] = (
            (self.__llm_experiment["llmResponseLength"] - minimum_response_length)
            / (maximum_response_length - minimum_response_length)
        )
        self.__performance_metrics.append("llmResponseLengthNormalised")

        model_comparison = self.__llm_experiment.groupby("modelName")[self.__performance_metrics].mean().reset_index()

        normalised_metrics: list[str] = [
            "llmResponseLengthNormalised", "answerFaithfulnessScore", "answerRelevanceScore", "contextRelevanceScore",
            "sourcedCited", "practicalityScore", "llmResponseLength"
        ]

        variable_count: int = len(normalised_metrics)
        metric_angles: list[float] = [float(n) / float(variable_count) * 2.0 * np.pi for n in range(variable_count)]
        metric_angles += metric_angles[:1]

        figure, axis = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        plt.xticks(metric_angles[:-1], normalised_metrics, size=12)

        axis.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
        plt.ylim(0, 1)

        colors: list[str] = ["#3498db", "#e74c3c"]

        for i, model in enumerate(model_comparison["modelName"]):
            values: list[float] = model_comparison.loc[model_comparison["modelName"] == model, normalised_metrics].values.flatten().tolist()
            values += values[:1]

            axis.plot(metric_angles, values, linewidth=2, linestyle="solid", label=model, color=colors[i])
            axis.fill(metric_angles, values, alpha=0.2, color=colors[i])

        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title("Model Comparison Across Performance Metrics", size=15)
        plt.tight_layout()

        plt.show()

        sys.stdout.write(
            f"\nModel comparison - mean values: {model_comparison[["modelName"] + self.__performance_metrics]}\n\n"
        )

    def _analyze_correlations(self) -> None:
        """
        Analyzes correlations between RAG variables and performance metrics by model."""
        sys.stdout.write("\n===3. CORRELATION ANALYSIS===\n\n")

        models: np.ndarray = self.__llm_experiment["modelName"].unique()

        for model in models:
            sys.stdout.write(f"\nCorrelations for model: {model}\n\n")
            model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

            correlation_matrix: pd.DataFrame = model_data[self.__key_rag_variables + self.__performance_metrics].corr()

            sys.stdout.write("\nCorrelations between RAG variables and performance metrics:\n\n")

            for rag_variable in self.__key_rag_variables:
                sys.stdout.write(f"\n{rag_variable} correlations:\n")

                for perf_metric in self.__performance_metrics:
                    correlation_value: float = correlation_matrix.loc[rag_variable, perf_metric]
                    correlation, p_value = pearsonr(model_data[rag_variable], model_data[perf_metric])

                    significance_marker = "*" if (p_value < csv_.ALPHA) else ""

                    sys.stdout.write(f"\t- {perf_metric}: {correlation_value:.6f} {significance_marker}\n")

            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title(f"Correlation Matrix for {model}")
            plt.tight_layout()

            plt.show()

    def _analyze_rag_variables_impact(self) -> None:
        """
        Analyzed how RAG variables impact performance metrics for each model.
        """
        sys.stdout.write("\n===4. RAG VARIABLES IMPACT ANALYSIS===\n\n")

        focus_metrics: list[str] = ["answerFaithfulnessScore", "answerRelevanceScore", "practicalityScore"]

        for variable in self.__key_rag_variables:
            sys.stdout.write(f"\nAnalyzing impact of {variable} on model performance:\n")

            plt.figure(figsize=(15, (len(focus_metrics) * 5)))

            for i, metric in enumerate(focus_metrics):
                plt.subplot(len(focus_metrics), 1, (i + 1))

                for model, color in (("gpt-4o", "#3498db"), ("gemini-2.0-flash", "#e74c3c")):
                    model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

                    plt.scatter(
                        model_data[variable],
                        model_data[metric],
                        alpha=0.6,
                        label=f"{model} (n={len(model_data)})",
                        color=color
                    )

                    sns.regplot(x=variable, y=metric, data=model_data, scatter=False, ci=None, color=color)

                    correlation, p_value = pearsonr(model_data[variable], model_data[metric])
                    plt.annotate(
                        f"{model}: r={correlation:.4f},"
                        f" p={p_value:.4f}",
                        xy=(0.02, 0.95 - i * csv_.ALPHA),
                        xycoords="axes fraction",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )

                plt.title(f"Impact of {variable} on {metric}")
                plt.xlabel(variable)
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            plt.show()

    def _compare_responses_to_parameters(self) -> None:
        """
        Compares how the two models respond differently to the same RAG parameter changes.
        """
        sys.stdout.write("\n===5. COMPARATIVE RESPONSE TO RAG PARAMETERS===\n\n")

        focus_metrics: list[str] = ["answerFaithfulnessScore", "answerRelevanceScore"]
        models: np.ndarray = self.__llm_experiment["modelName"].unique()

        i: int = 0
        slope_data: list[dict[str, float]] = [{}] * (len(self.__key_rag_variables) * len(focus_metrics) * len(models))

        for rag_variable in self.__key_rag_variables:
            for metric in focus_metrics:
                regression_coefficients_per_model: dict[str, dict[str, float]] = {}

                for model in models:
                    model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

                    X: np.array = add_constant(model_data[rag_variable])
                    y: pd.DataFrame = model_data[metric]

                    regression: RegressionResults = OLS(y, X).fit()

                    slope: float = regression.params[rag_variable]
                    p_value: float = regression.pvalues[rag_variable]
                    r_squared: float = regression.rsquared

                    regression_coefficients_per_model[model] = {"slope": slope, "p_value": p_value, "r_squared": r_squared}

                    slope_data[i] = {
                        "Variable": rag_variable,
                        "Metric": metric,
                        "Model": model,
                        "Slope": slope,
                        "P_value": p_value,
                        "R_squared": r_squared,
                        "Significant": (p_value < csv_.ALPHA)
                    }

                    i += 1

                sys.stdout.write(f"\n{'-' * 85}\n{rag_variable} -> {metric} relationship:\n")

                for model, results in regression_coefficients_per_model.items():
                    significance_marker: str = "*" if results["p_value"] < csv_.ALPHA else ""

                    sys.stdout.write(
                        f"\t- {model}: slope = {results["slope"]:.4f} {significance_marker},"
                        f" p-value = {results["p_value"]:.4f}, R² = {results["r_squared"]:.3f}\n"
                    )

                models_: list[str] = list(models)
                slope_difference: float = abs(
                    regression_coefficients_per_model[models_[0]]["slope"] -
                    regression_coefficients_per_model[models_[1]]["slope"]
                )
                sys.stdout.write(f"\t- Difference in slopes: {slope_difference:.6f}\n\n")

        slope_differences: pd.DataFrame = pd.DataFrame(slope_data)
        unique_var_metrics: pd.Series = slope_differences[["Variable", "Metric"]].drop_duplicates()

        plt.figure(figsize=(12, len(unique_var_metrics) * 3))

        for i, (_, row) in enumerate(unique_var_metrics.iterrows()):
            variable: pd.Series = row["Variable"]
            metric: pd.Series = row["Metric"]

            plot_data: pd.DataFrame = slope_differences[(slope_differences["Variable"] == variable) & (slope_differences["Metric"] == metric)]

            plt.subplot(len(unique_var_metrics), 1, (i + 1))

            bars: plt.BarContainer = plt.bar(plot_data["Model"], plot_data["Slope"], color=["#3498db", "#e74c3c"])

            for j, bar in enumerate(bars):
                if plot_data.iloc[j]["Significant"]:
                    plt.text(
                        (bar.get_x() + bar.get_width() / 2.0),
                        (bar.get_height() + 0.01) if (bar.get_height() >= 0.0) else (bar.get_height() - 0.03),
                        "*",
                        ha="center",
                        va="center",
                        fontsize=12
                    )

            plt.title(f"Model Sensitivity: {variable} → {metric}")
            plt.ylabel("Coefficient (Slope)")
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            plt.show()

    def _conduct_statistical_tests(self) -> None:
        """
        Conducts statistical tests to compare the two models across different conditions.
        """
        sys.stdout.write("\n===6. STATISTICAL COMPARISON TESTS===\n\n")

        sys.stdout.write("\nOverall model comparison (t-tests):\n\n")
        
        for metric in self.__performance_metrics:
            chatgpt_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == "gpt-4o"][metric]
            gemini_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == "gemini-2.0-flash"][metric]

            t_statistic, p_value = ttest_ind(chatgpt_data, gemini_data, equal_var=False)
            significance_marker: str = "*" if (t_statistic < csv_.ALPHA) else ""

            sys.stdout.write(
                f"{metric}: t={t_statistic:.3f}, p={t_statistic:.3f} {significance_marker}"
                f"\t- ChatGPT-4o: mean={chatgpt_data.mean():.5f}, std={chatgpt_data.std():.5f}, n={len(chatgpt_data)}\n"
                f"\t- Gemini 2.0 Flash: mean={gemini_data.mean():.5f}, std={gemini_data.std():.5f}, n={len(gemini_data)}\n\n"
            )

        sys.stdout.write("\nConditional analysis - model performance by parameter ranges:\n")

        for variable in self.__key_rag_variables:
            median_val: float = self.__llm_experiment[variable].median()
            low_range: pd.DataFrame = self.__llm_experiment[self.__llm_experiment[variable] <= median_val]
            high_range: pd.DataFrame = self.__llm_experiment[self.__llm_experiment[variable] > median_val]

            sys.stdout.write(f"\nAnalysis for {variable} (median split at {median_val:.2f}):\n")

            for metric in self.__key_rag_variables[:3]:
                sys.stdout.write(f"\n  {metric}:\n")

                gpt_low_range: pd.DataFrame = low_range[low_range["modelName"] == "gpt-4o"][metric]
                gemini_low_range: pd.DataFrame = low_range[low_range["modelName"] == "2.0 Flash"][metric]

                t_statistic, p_value = ttest_ind(gpt_low_range, gemini_low_range, equal_var=False)
                significance_marker: str = "*" if (p_value < csv_.ALPHA) else ""

                sys.stdout.write(
                    f"\t- Low {variable} (≤{median_val:.2f}): GPT={gpt_low_range.mean():.4f},"
                    f" Gemini={gemini_low_range.mean():.4f}, p={p_value:.4f}{significance_marker}\n"
                )

                gpt_high_range: pd.DataFrame = high_range[high_range["modelName"] == "gpt-4o"][metric]
                gemini_high_range: pd.DataFrame = high_range[high_range["modelName"] == "gemini-2.0-flash"][metric]

                t_statistic, p_value = ttest_ind(gpt_high_range, gemini_high_range, equal_var=False)
                significance_marker: str = "*" if (p_value < csv_.ALPHA) else ""

                sys.stdout.write(
                    f"\t- High {variable} (>{median_val:.2f}): GPT={gpt_high_range.mean():.3f},"
                    f" Gemini={gemini_high_range.mean():.3f}, p={p_value:.3f}{significance_marker}\n"
                )

            break

    def _run_model_specific_regressions(self) -> None:
        """
        Runs a separate regression analyses for each model to identify key variables affecting performance.
        """
        sys.stdout.write("\n===7. MODEL-SPECIFIC REGRESSION ANALYSIS===\n\n")

        target_metrics: list[str] = ["answerRelevanceScore", "answerFaithfulnessScore"]
        predictors: list[str] = self.__key_rag_variables + ["promptLength", "instructionLength", "queryLength"]

        for target in target_metrics:
            sys.stdout.write(f"\nRegression analysis for target: {target}\n")

            formula: str = f"{target} ~ {" + ".join(predictors)}"

            for model in ("gpt-4o", "2.0 Flash"):
                sys.stdout.write(f"\n  Model: {model}\n")

                model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model].copy()

                regression_model: Model = smf.ols(formula, data=model_data).fit()

                significance_predictors: pd.DataFrame = pd.DataFrame(
                    {
                        "Variable": regression_model.params.index,
                        "Coefficient": regression_model.params.values,
                        "P-value": regression_model.pvalues.values,
                        "Significant": (regression_model.pvalues.values < csv_.ALPHA)
                    }
                )
                significance_predictors = significance_predictors[significance_predictors["Variable"] != "Intercept"]
                significance_predictors = significance_predictors.sort_values("P-value")

                sys.stdout.write(
                    f"\tR-squared: {regression_model.rsquared:.3f}\n"
                    f"\tAdjusted R-squared: {regression_model.rsquared_adj:.3f}\n"
                    f"\tF-statistic: {regression_model.fvalue:.3f}, p-value: {regression_model.f_pvalue:.4f}\n"
                )

                sys.stdout.write("\n\tVariable impacts:\n")

                for _, row in significance_predictors.iterrows():
                    significance_marker: str = "*" if (row["Significant"]) else ""

                    sys.stdout.write(
                        f"\t- {row["Variable"]}: {row["Coefficient"]:.4f} {significance_marker}, p={row["P-value"]:.4f}\n"
                    )

    def _summarize_findings(self) -> None:
        """
        Summarizes the key findings from all analyses.
        """
        sys.stdout.write("\n===8. SUMMARY OF FINDINGS===\n")

        model_performance: pd.DataFrame = self.__llm_experiment.groupby("modelName")[self.__performance_metrics].agg(["mean", "std"])

        sys.stdout.write(f"\nOverall performance comparison:\n{model_performance}\n")

        model_means: pd.DataFrame = self.__llm_experiment.groupby("modelName")[self.__performance_metrics].mean()

        better_counts: dict[str, int] = {model: 0 for model in model_means.index}

        for metric in self.__performance_metrics:
            better_model: str = model_means[metric].idxmax()
            better_counts[better_model] += 1

        sys.stdout.write("\nModel with better performance by metric count:\n")

        for model, count in better_counts.items():
            sys.stdout.write(f"- {model}: better in {count}/{len(self.__performance_metrics)} metrics\n")

        sys.stdout.write("\nKey findings on RAG variable impacts:\n")

        significant_correlations: list[dict[str, str | float]] = []

        for model in self.__llm_experiment["modelName"].unique():
            model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

            for rag_variable in self.__key_rag_variables:
                for metric in self.__performance_metrics:
                    correlation, p_value = pearsonr(model_data[rag_variable], model_data[metric])

                    if (p_value < csv_.ALPHA):
                        significant_correlations.append(
                            {
                                "model": model,
                                "variable": rag_variable,
                                "metric": metric,
                                "correlation": correlation,
                                "p_value": p_value
                            }
                        )

        significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        sys.stdout.write("\nStrongest significant correlations between RAG variables and performance:\n")

        for i, correlation in enumerate(significant_correlations[:5]):
            sys.stdout.write(
                f"\n{i + 1}. {correlation["model"]}: {correlation["variable"]} → {correlation["metric"]} "
                f"(r = {correlation["correlation"]:.6f}, p = {correlation["p_value"]:.6f})\n"
            )

        sys.stdout.write("\n\nDifferences in model sensitivity to RAG parameters:\n")

        for rag_variable in self.__key_rag_variables:
            for metric in ("answerRelevanceScore", "answerFaithfulnessScore"):
                model_sensitivities: dict[str, float] = {}

                for model in self.__llm_experiment["modelName"].unique():
                    model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

                    X: np.ndarray = add_constant(model_data[rag_variable])
                    y: pd.DataFrame = model_data[metric]

                    regression: RegressionResults = OLS(y, X).fit()

                    model_sensitivities[model] = {
                        "slope": regression.params[rag_variable],
                        "p_value": regression.pvalues[rag_variable],
                        "significant": (regression.pvalues[rag_variable] < csv_.ALPHA)
                    }

                models: list[str] = list(model_sensitivities.keys())

                slope_differance: float = abs(model_sensitivities[models[0]]["slope"] - model_sensitivities[models[1]]["slope"])

                if (
                    (model_sensitivities[models[0]]["significant"])
                    or (model_sensitivities[models[1]]["significant"])
                    and (slope_differance > 0.01)
                ):
                    sys.stdout.write(
                        f"\n- {rag_variable} -> {metric}: {models[0]} (slope={model_sensitivities[models[0]]["slope"]:.6f}) vs "
                        f"{models[1]} (slope={model_sensitivities[models[1]]["slope"]:.4f}), difference={slope_differance:.6f}\n"
                    )

        sys.stdout.write("\n\nKey takeaways and recommendations:\n")

        for model in self.__llm_experiment["modelName"].unique():
            sys.stdout.write(f"\n\nFor {model}:\n")

            model_data: pd.DataFrame = self.__llm_experiment[self.__llm_experiment["modelName"] == model]

            for metric in ("answerRelevanceScore", "answerFaithfulnessScore"):
                sys.stdout.write(f"\n- To optimize {metric}:\n")

                for rag_variable in self.__key_rag_variables:
                    correlation, p_value = pearsonr(model_data[rag_variable], model_data[metric])

                    if (p_value < csv_.ALPHA):
                        direction: str = "increase" if (correlation > 0.0) else "decrease"

                        sys.stdout.write(f"\n  * {direction} {rag_variable} (r={correlation:.3f}, p={p_value:.4f})\n")

        sys.stdout.write(
            "\n\nConclusion:"
            "\nThis analysis has examined how GPT-4o and Gemini 2.0 Flash respond to various RAG system configurations."
            "\nThe findings indicate that these models demonstrate different sensitivities to RAG parameters,"
            "\nsuggesting that optimal RAG system design should be tailored to the specific LLM being used.\n"
        )

        if ((better_counts) and (max(better_counts.values()) > min(better_counts.values()))):
            better_model: str = max(better_counts.items(), key=lambda x: x[1])[0]

            sys.stdout.write(
                f"\n\nOverall, {better_model} demonstrated superior performance across more metrics in our RAG evaluation,"
                "\nbut the optimal choice may depend on specific use case requirements and parameter configurations.\n"
            )


class ResearchQuestion3:
    def __init__(self, llm_experiment: pd.DataFrame) -> None:
        self.__llm_experiment: pd.DataFrame = llm_experiment.copy()

        self.__llm_experiment["composite_score"] = (
            0.3 * self.__llm_experiment["answerFaithfulnessScore"] +
            0.3 * self.__llm_experiment["answerRelevanceScore"] +
            0.1 * self.__llm_experiment["contextRelevanceScore"] +
            0.3 * self.__llm_experiment["practicalityScore"]
        )

        self.__llm_experiment = self.__llm_experiment.drop(
            columns=[
                "pdfDocument", "timestamp", "sentenceEmbeddingModel", "dimensionCountVectors", "instructionContextText",
                "query", "rephrasedQuestions", "topKDocuments", "lmmResponse", "tokenLimit", "sourcedCited"
            ]
        )

        self.__parameter_columns: list[str] = ["pdfPageSize", "topKSimilarDocuments", "promptLength", "queryLength"]

        self.__categorical_parameters = ["modelName", "largeLanguageModel"]
        self.__llm_experiment_encoded: pd.DataFrame = pd.get_dummies(
            self.__llm_experiment,
            columns=self.__categorical_parameters,
            drop_first=False
        )

        self.__target_variables = [
            "answerFaithfulnessScore", "answerRelevanceScore", "contextRelevanceScore", "practicalityScore",
            "composite_score"
        ]

    def answer(self) -> None:
        """
        The main method to analyze which variable configuration leads to the most understandable and actionable LLM answers for technical requirements.
        """
        sys.stdout.write(
            "Analyzing which variable configurations produce the most understandable and actionable technical requirements...\n\n"
        )

        eda_results: dict[str, NDFrame | np.ndarray] = self._conduct_exploratory_data_analysis()
        sys.stdout.write(f"\n\n===EDA RESULTS===\n\n{eda_results}\n\n")

        correlation_results: dict[str, pd.DataFrame] = self._correlation_analysis()

        feature_importance: dict[str, pd.DataFrame] = self._feature_importance_analysis()

        regression_results: dict[str, Any] = self._regression_analysis()

        optimal_configurations: dict[str, dict[str, pd.Series]] = self._determine_optimal_configurations()

        self._visualize_results(feature_importance)

        results: dict[str, Any] = {
            "summary": "Analysis of LLM parameter configurations for optimal technical requirement generation",
            "key_findings": self._generate_key_findings(correlation_results, feature_importance, regression_results),
            "optimal_configurations": optimal_configurations,
            "feature_importance": feature_importance,
            "regression_results": regression_results,
            "recommendations": self._generate_recommendations(optimal_configurations)
        }

        self._print_summary(results)

    def _conduct_exploratory_data_analysis(self) -> dict[str, NDFrame | np.ndarray]:
        """
        Performs an exploratory data analysis.
        """
        eda_results: dict[str, NDFrame | np.ndarray] = {
            "target_stats": self.__llm_experiment[self.__target_variables].describe(),
            "parameter_stats": self.__llm_experiment[self.__parameter_columns].describe()
        }

        for target_variable in self.__target_variables:
            eda_results[f"{target_variable}_skewness"] = skew(self.__llm_experiment[target_variable])
            eda_results[f"{target_variable}_kurtosis"] = kurtosis(self.__llm_experiment[target_variable])

        eda_results["score_by_model"] = self.__llm_experiment.groupby("modelName")[self.__target_variables].mean()

        parameter_correlations: dict[str, dict[str, float]] = {}

        for parameter in self.__parameter_columns:
            parameter_correlations[parameter] = {}

            for target in self.__target_variables:
                parameter_correlations[parameter][target] = self.__llm_experiment[[parameter, target]].corr().iloc[0, 1]

        eda_results["parameter_score_correlations"] = parameter_correlations

        return eda_results

    def _correlation_analysis(self) -> dict[str, pd.DataFrame]:
        """
        AnalyzeS correlations between parameters and target variables.
        """
        results: dict[str, pd.DataFrame] = {}

        correlation_matrix: pd.DataFrame = self.__llm_experiment[self.__parameter_columns + self.__target_variables].corr()

        for target in self.__target_variables:
            results[f"{target}_correlations"] = correlation_matrix[target][correlation_matrix.index.isin(self.__parameter_columns)].sort_values(
                ascending=False
            )

        return results

    def _feature_importance_analysis(self) -> dict[str, pd.DataFrame]:
        """
        Calculatest the feature importance using Random Forest regression.
        """
        feature_importance_variables: dict[str, pd.DataFrame] = {}

        feature_columns: list[str] = [
            column for column in self.__llm_experiment_encoded.columns if any(param in column for param in self.__parameter_columns)
        ]

        X: pd.DataFrame = self.__llm_experiment_encoded[feature_columns]

        for target in self.__target_variables:
            y: pd.DataFrame = self.__llm_experiment_encoded[target]

            random_forest_regressor: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
            random_forest_regressor.fit(X, y)

            feature_importance_variables[target] = pd.DataFrame(
                {
                    "Feature": feature_columns,
                    "Importance": random_forest_regressor.feature_importances_
                }
            ).sort_values("Importance", ascending=False).head(10)

        return feature_importance_variables

    def _regression_analysis(self) -> dict[str, Any]:
        """
        Performs a regression analysis to model the relationship between parameters and scores.
        """
        regression_results: dict[str, Any] = {}

        feature_columns: list[str] = [
            column for column in self.__llm_experiment_encoded.columns
            if (
                any(param in column for param in self.__parameter_columns)
                and (column not in self.__target_variables)
            )
        ]

        X: pd.DataFrame = self.__llm_experiment_encoded[feature_columns].select_dtypes(include=[np.number])
        X: pd.DataFrame = self.__llm_experiment_encoded[feature_columns].select_dtypes(include=[np.number])

        for target in self.__target_variables:
            y: pd.DataFrame = self.__llm_experiment_encoded[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            linear_regression: LinearRegression = LinearRegression()
            linear_regression.fit(X_train, y_train)
            y_predicted_linear_regression: np.ndarray = linear_regression.predict(X_test)

            random_forest_regressor: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
            random_forest_regressor.fit(X_train, y_train)
            y_pred_random_forest: np.ndarray = random_forest_regressor.predict(X_test)

            if (len(feature_columns) != len(linear_regression.coef_)):
                length_difference: int = (len(feature_columns) - len(linear_regression.coef_))
                average_coefficient: float = linear_regression.coef_.mean()
                linear_regression.coef_ = list(linear_regression.coef_) + ([average_coefficient] * length_difference)

            regression_results[target] = {
                "linear_regression": {
                    "r2": r2_score(y_test, y_predicted_linear_regression),
                    "mse": mean_squared_error(y_test, y_predicted_linear_regression),
                    "coefficients": pd.DataFrame(
                        {
                            "Feature": feature_columns,
                            "Coefficient": linear_regression.coef_
                        }
                    ).sort_values("Coefficient", ascending=False)
                },
                "random_forest": {
                    "r2": r2_score(y_test, y_pred_random_forest),
                    "mse": mean_squared_error(y_test, y_pred_random_forest)
                }
            }

            X_train_sm: np.ndarray = add_constant(X_train)
            ols: RegressionResults = OLS(y_train, X_train_sm).fit()

            regression_results[target]["ols_summary"] = {
                "r2": ols.rsquared,
                "adj_r2": ols.rsquared_adj,
                "significant_features": ols.pvalues[ols.pvalues < csv_.ALPHA].index.tolist()
            }

        return regression_results

    def _determine_optimal_configurations(self) -> dict[str, dict[str, pd.Series]]:
        """
        Determines the optimal parameter configurations.
        """
        optimal_parameter_configurations: dict[str, dict[str, pd.Series]] = {}

        top_n_configurations: pd.DataFrame = self.__llm_experiment.sort_values("composite_score", ascending=False).head(5)

        for i, row in top_n_configurations.iterrows():
            configuration: dict[str, pd.Series] = {}

            for parameter in self.__parameter_columns:
                if (parameter in row.index):
                    configuration[parameter] = row[parameter]

            for target in self.__target_variables:
                if (target in row.index):
                    configuration[target] = row[target]

            optimal_parameter_configurations[f"Configuration_{i}"] = configuration

        for parameter in self.__parameter_columns:
            self.__llm_experiment[f"{parameter}_bin"] = pd.qcut(self.__llm_experiment[parameter], 5, duplicates="drop")

            bin_means: pd.DataFrame = self.__llm_experiment.groupby(f"{parameter}_bin")["composite_score"].mean().sort_values(ascending=False)

            if (not bin_means.empty):
                best_bin: AxisProperty = bin_means.index[0]
                optimal_parameter_configurations[f"best_{parameter}_range"] = (best_bin.left, best_bin.right)

            for target in self.__target_variables[:4]:
                target_bin_means = self.__llm_experiment.groupby(f"{parameter}_bin")[target].mean().sort_values(
                    ascending=False)

                if (not target_bin_means.empty):
                    best_target_bin = target_bin_means.index[0]
                    optimal_parameter_configurations[f"best_{parameter}_range_for_{target}"] = (
                        best_target_bin.left,
                        best_target_bin.right
                    )

            self.__llm_experiment = self.__llm_experiment.drop(f"{parameter}_bin", axis=1)

        return optimal_parameter_configurations

    def _visualize_results(self, feature_importance: dict[str, pd.DataFrame]) -> None:
        """
        Creates visualizations for the results.
        """
        plt.style.use("seaborn-v0_8-whitegrid")

        plt.figure(figsize=(12, 8))
        feature_importances: pd.DataFrame = feature_importance["composite_score"]
        sns.barplot(x="Importance", y="Feature", data=feature_importances)
        plt.title("Feature Importance for Composite Score")
        plt.tight_layout()
        plt.savefig(f"{directories.ANALYSIS_PLOTS}/feature_importance_composite.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 8))
        correlation_matrix: pd.DataFrame = self.__llm_experiment[self.__parameter_columns + self.__target_variables].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=.5
        )
        plt.title("Correlation Between Parameters and Performance Scores")
        plt.tight_layout()
        plt.savefig(f"{directories.ANALYSIS_PLOTS}/parameter_score_correlation_heatmap.png")
        plt.show()
        plt.close()

        for parameter in self.__parameter_columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=parameter, y="composite_score", data=self.__llm_experiment, alpha=0.6)
            plt.title(f"{parameter} vs Composite Score")
            plt.tight_layout()
            plt.savefig(f"{directories.ANALYSIS_PLOTS}/{parameter}_vs_composite_score.png")
            plt.close()

        sns.pairplot(
            self.__llm_experiment,
            x_vars=self.__parameter_columns,
            y_vars=self.__target_variables,
            height=3,
            aspect=1.5
        )
        plt.suptitle("Parameter vs Score Relationships", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{directories.ANALYSIS_PLOTS}/parameter_score_pairplots.png")
        plt.show()
        plt.close()

    @staticmethod
    def _generate_key_findings(correlation_results: dict[str, pd.DataFrame],
                               feature_importance: dict[str, pd.DataFrame],
                               regression_results: dict[str, Any]) -> list[str]:
        """
        Generates key findings focused on our parameters and their impact on scores.
        """
        findings: list[str] = []

        top_features: list[str] = feature_importance["composite_score"]["Feature"].tolist()[:4]
        findings.append(f"The most important parameters for overall quality are: {', '.join(top_features)}")

        composite_correlations: pd.DataFrame = correlation_results["composite_score_correlations"]
        positive_correlations: list[float] = [
            correlation for correlation in composite_correlations.index if (composite_correlations[correlation] > 0.0)
        ]
        negative_correlations: list[float] = [
            correlation for correlation in composite_correlations.index if (composite_correlations[correlation] < 0.0)
        ]

        if (positive_correlations):
            findings.append(f"Parameters positively correlated with better outcomes: {', '.join(positive_correlations)}")
        if (negative_correlations):
            findings.append(f"Parameters negatively correlated with outcomes: {', '.join(negative_correlations)}")

        for target in ("answerFaithfulnessScore", "answerRelevanceScore", "contextRelevanceScore", "practicalityScore"):
            target_importance: list[str] = feature_importance[target]["Feature"].tolist()[:2]
            findings.append(f"For {target}, the most important parameters are: {', '.join(target_importance)}")

        if ("composite_score" in regression_results):
            r2_random_fores: float = regression_results["composite_score"]["random_forest"]["r2"]
            findings.append(f"Our predictive model explains {r2_random_fores:.2f} of the variance in the composite quality score")

            if ("ols_summary" in regression_results["composite_score"]):
                sig_features = regression_results["composite_score"]["ols_summary"]["significant_features"]
                if sig_features:
                    findings.append(f"Statistically significant parameters: {', '.join(sig_features[:5])}")

        return findings

    def _generate_recommendations(self, optimal_configurations: dict[str, dict[str, pd.Series]]) -> list[str]:
        """
        Generates recommendations based on optimal configurations.
        """
        recommendations: list[str] = []

        for parameter in self.__parameter_columns:
            range_key: str = f"best_{parameter}_range"

            if (range_key not in optimal_configurations):
                continue

            minimum_value, maximum_value = optimal_configurations[range_key]
            recommendations.append(
                f"Set {parameter} between {minimum_value:.1f} and {maximum_value:.1f} for best overall results"
            )

        for target in ("answerFaithfulnessScore", "answerRelevanceScore", "contextRelevanceScore", "practicalityScore"):
            recommendations.append(f"\nTo optimize {target}:")

            for parameter in self.__parameter_columns:
                range_key: str = f"best_{parameter}_range_for_{target}"

                if (range_key not in optimal_configurations):
                    continue

                minimum_value, maximum_value = optimal_configurations[range_key]
                recommendations.append(f"- Set {parameter} between {minimum_value:.1f} and {maximum_value:.1f}")

        if ("Configuration_0" in optimal_configurations):
            recommendations.append(
                "\nBest overall configuration found:\n" +
                "\n".join(
                    f"- {key}: {value}" for key, value in optimal_configurations["Configuration_0"].items()
                    if key in self.__parameter_columns
                )
            )

        return recommendations

    def _print_summary(self, results: dict[str, Any]) -> None:
        """
        Prints a summary of the results to the standard output stream.
        """
        sys.stdout.write("\n\n===RESEARCH QUESTION 3 ANALYSIS SUMMARY===\n\nKEY FINDINGS:\n")

        for i, finding in enumerate(results["key_findings"], 1):
            sys.stdout.write(f"{i}. {finding}\n\n")

        sys.stdout.write("\n===RECOMMENDATIONS===\n")

        for i, recommendation in enumerate(results["recommendations"], 1):
            sys.stdout.write(f"{i}. {recommendation}\n\n")

        sys.stdout.write("\n===OPTIMAL CONFIGURATION===\n\n")

        if ("Configuration_0" in results["optimal_configurations"]):
            configuration: dict[str, float] = results["optimal_configurations"]["Configuration_0"]

            for parameter in self.__parameter_columns:
                if (parameter not in configuration):
                    continue

                sys.stdout.write(f"- {parameter}: {configuration[parameter]}\n")

        sys.stdout.write("\n===MODEL PERFORMANCE===\n\n")

        for parameter in self.__parameter_columns:
            sys.stdout.write(f"\n{parameter} optimal ranges:\n")

            for target in self.__target_variables:
                range_key: str = f"best_{parameter}_range_for_{target}" if (target != "composite_score") else f"best_{parameter}_range"

                if (range_key not in results["optimal_configurations"]):
                    continue

                minimum_value, maximum_value = results["optimal_configurations"][range_key]

                target_name = "Overall" if (target == "composite_score") else target

                sys.stdout.write(f"- {target_name}: {minimum_value:.1f} to {maximum_value:.1f}\n")
