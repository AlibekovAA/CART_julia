import random
from typing import List, Dict
import io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree


def calculate_impurity_and_split(data: np.ndarray, target: np.ndarray, max_depth: int) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(data, target)
    return model


def evaluate_tree_accuracy(data: np.ndarray, target: np.ndarray, max_depth: int, output_path: str) -> float:
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    model = calculate_impurity_and_split(x_train, y_train, max_depth)
    predictions = model.predict(x_test)
    correct_predictions = sum(predictions == y_test)

    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=load_iris().feature_names, class_names=load_iris().target_names)
    plt.title(f"Дерево решений (max_depth={max_depth})")

    tree_image_path = f"decision_tree_depth_{max_depth}.png"
    plt.savefig(tree_image_path)
    plt.close()

    with open(output_path, "a", encoding='utf-8') as f:
        f.write(f"## Дерево решений для глубины {max_depth}\n")
        f.write(f"![Дерево решений (max_depth={max_depth})]({tree_image_path})\n\n")

    return accuracy_score(y_test, predictions), correct_predictions


def stability_test(data: np.ndarray, target: np.ndarray, max_depth: int, iterations: int = 3) -> List[Dict[str, int]]:
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    results = []

    for i in range(iterations):
        sample_indices = random.sample(range(len(x_train)), int(0.9 * len(x_train)))
        x_train_reduced, y_train_reduced = x_train[sample_indices], y_train[sample_indices]

        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(x_train_reduced, y_train_reduced)

        predictions = model.predict(x_test)
        correct = sum(predictions == y_test)
        results.append({
            'Iteration': i + 1,
            'Train Size': len(x_train_reduced),
            'Test Size': len(x_test),
            'Correct': correct,
            'Incorrect': len(y_test) - correct
        })
    return results


def display_results(accuracies: List[float], depths: List[int], correct_counts: List[int], output_path: str) -> None:
    depth_accuracy_data = pd.DataFrame({
        'Глубина': depths,
        'Точность': accuracies,
        'Количество правильно классифицированных': correct_counts,
    })

    plt.figure(figsize=(12, 7))
    plt.plot(depths, accuracies, marker='o', linestyle='-', color='blue', markersize=8, linewidth=2, label='Точность')
    plt.title("Зависимость точности от глубины дерева", fontsize=16)
    plt.xlabel("Глубина дерева", fontsize=14)
    plt.ylabel("Точность", fontsize=14)
    plt.xticks(depths)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, txt in enumerate(correct_counts):
        plt.annotate(txt, (depths[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='black')

    plt.legend()
    plt.tight_layout()

    result_plot_path = "results_plot.png"
    plt.savefig(result_plot_path)
    plt.close()

    with open(output_path, "a", encoding='utf-8') as f:
        f.write("## Результаты\n")
        f.write(f"![График зависимости точности от глубины дерева]({result_plot_path})\n\n")
        f.write("### Таблица с глубиной, точностью и количеством правильно классифицированных:\n")
        f.write(depth_accuracy_data.to_markdown(index=False))
        f.write("\n")


def find_best_depth(depths: List[int], accuracies: List[float]) -> int:
    max_accuracy = max(accuracies)
    best_depth_candidates = [depth for depth, accuracy in zip(depths, accuracies) if accuracy == max_accuracy]
    return min(best_depth_candidates)


def run_decision_tree_experiments(data: np.ndarray, target: np.ndarray, depths: List[int], output_path: str) -> None:
    accuracies = []
    correct_counts = []

    for depth in depths:
        accuracy, correct_count = evaluate_tree_accuracy(data, target, depth, output_path)
        accuracies.append(accuracy)
        correct_counts.append(correct_count)

    display_results(accuracies, depths, correct_counts, output_path)

    best_depth = find_best_depth(depths, accuracies)
    stability_results = stability_test(data, target, best_depth)
    stability_df = pd.DataFrame(stability_results)

    with open(output_path, "a", encoding='utf-8') as f:
        f.write(f"### Лучшая глубина: {best_depth}\n")
        f.write("### Устойчивость для лучшей глубины:\n")
        f.write(stability_df.to_markdown(index=False))
        f.write("\n")


def load_data() -> pd.DataFrame:
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df = df.drop_duplicates()
    return df


def perform_eda(data: pd.DataFrame, output_path: str = "eda_report.md") -> None:
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("# EDA\n")
        f.write("## Общая информация о данных\n")
        info_buffer = data.info(buf=None)
        info_buffer = io.StringIO()
        data.info(buf=info_buffer)
        f.write(f"```\n{info_buffer.getvalue()}\n```\n")

        f.write("## Статистическое описание данных\n")
        f.write(f"```\n{data.describe()}\n```\n")

        f.write("## Пропущенные значения\n")
        f.write(f"```\n{data.isnull().sum()}\n```\n")

        f.write("## Количество дубликатов\n")
        duplicate_count = data.duplicated().sum()
        f.write(f"- Количество дубликатов: {duplicate_count}\n\n")

        f.write("## Распределение классов\n")
        class_counts = data['target'].value_counts()
        f.write(f"```\n{class_counts}\n```\n")

    pairplot_path = "pairplot.png"
    sns.pairplot(data, hue="target", markers=["o", "s", "D"])
    plt.suptitle("Распределение признаков и классов", y=1.02)
    plt.savefig(pairplot_path)
    plt.close()

    heatmap_path = "heatmap.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.drop(columns="target").corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Корреляционная матрица признаков")
    plt.savefig(heatmap_path)
    plt.close()

    with open(output_path, "a", encoding='utf-8') as f:
        f.write("## Графики\n")
        f.write(f"### Распределение признаков и классов\n![Pairplot]({pairplot_path})\n\n")
        f.write(f"### Корреляционная матрица признаков\n![Heatmap]({heatmap_path})\n")


df = load_data()
perform_eda(df)

decision_tree_report_path = "decision_tree_report.md"
with open(decision_tree_report_path, "w", encoding='utf-8') as f:
    f.write("# Отчет по дереву решений\n")

run_decision_tree_experiments(data=df.drop(columns='target').values, target=df['target'].values, depths=[1, 3, 5, 7, 10],
                              output_path=decision_tree_report_path)
