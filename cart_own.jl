using Random
using DataFrames
using Statistics
using RDatasets
using Graphs
using GraphPlot
using Colors
using CategoricalArrays
using Plots
using PrettyTables

struct Node
    feature::Int
    threshold::Real
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    value::Union{Int, Nothing}
end

function count_elements(arr::Vector{Int})::Dict{Int, Int}
    counts = Dict{Int, Int}()
    for elem in arr
        counts[elem] = get(counts, elem, 0) + 1
    end
    return counts
end

function gini_impurity(y::Vector{Int})::Float64
    class_counts = count_elements(y)
    total_count = length(y)
    return 1.0 - sum((count / total_count)^2 for count in values(class_counts))
end

function best_split(X::Matrix{Float64}, y::Vector{Int})::Tuple{Int, Real, Vector{Int}, Vector{Int}}
    best_gini = Inf
    best_feature = -1
    best_threshold = 0.0
    best_left_indices = Int[]
    best_right_indices = Int[]

    for feature in 1:size(X, 2)
        thresholds = sort(unique(X[:, feature]))

        for threshold in thresholds
            left_indices = findall(X[:, feature] .<= threshold)
            right_indices = findall(X[:, feature] .> threshold)

            if isempty(left_indices) || isempty(right_indices)
                continue
            end

            left_y = y[left_indices]
            right_y = y[right_indices]

            gini = (length(left_y) * gini_impurity(left_y) + length(right_y) * gini_impurity(right_y)) / length(y)

            if gini < best_gini
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices
            end
        end
    end

    return best_feature, best_threshold, best_left_indices, best_right_indices
end

function build_tree(X::Matrix{Float64}, y::Vector{Int}, max_depth::Int, depth::Int)::Node
    if depth >= max_depth || length(unique(y)) == 1
        return Node(-1, -1.0, nothing, nothing, mode(y))
    end

    feature, threshold, left_indices, right_indices = best_split(X, y)

    if feature == -1
        return Node(-1, -1.0, nothing, nothing, mode(y))
    end

    left_node = build_tree(X[left_indices, :], y[left_indices], max_depth, depth + 1)
    right_node = build_tree(X[right_indices, :], y[right_indices], max_depth, depth + 1)

    return Node(feature, threshold, left_node, right_node, nothing)
end

function mode(y::Vector{Int})::Int
    class_counts = count_elements(y)
    max_class = argmax(class_counts)[1]
    return max_class
end

function predict(tree::Node, x::Vector{Float64})::Int
    if tree.value !== nothing
        return tree.value
    end

    if x[tree.feature] <= tree.threshold
        return predict(tree.left, x)
    else
        return predict(tree.right, x)
    end
end

function fit(X::Matrix{Float64}, y::Vector{Int}, max_depth::Int)::Node
    return build_tree(X, y, max_depth, 0)
end

function score(tree::Node, X::Matrix{Float64}, y::Vector{Int})::Tuple{Float64, Int}
    predictions = [predict(tree, X[i, :]) for i in 1:size(X, 1)]
    correct_predictions = sum(predictions .== y)
    accuracy = correct_predictions / length(y)
    return accuracy, correct_predictions
end

function visualize_tree(tree::Node, graph::SimpleDiGraph, labels::Dict{Int, String}, current_id::Int)::Int
    if tree.value !== nothing
        labels[current_id] = "Class: $(tree.value)"
    else
        labels[current_id] = "X$(tree.feature) <= $(tree.threshold)"
    end

    if tree.left !== nothing
        add_vertex!(graph)
        add_edge!(graph, current_id, nv(graph))
        current_id = visualize_tree(tree.left, graph, labels, nv(graph))
    end

    if tree.right !== nothing
        add_vertex!(graph)
        add_edge!(graph, current_id, nv(graph))
        current_id = visualize_tree(tree.right, graph, labels, nv(graph))
    end

    return current_id
end

function plot_tree(tree::Node)
    graph = SimpleDiGraph()
    add_vertex!(graph)
    labels = Dict{Int, String}()
    visualize_tree(tree, graph, labels, 1)

    node_colors = distinguishable_colors(nv(graph), colorant"lightblue")

    node_labels_vector = [get(labels, i, "") for i in 1:nv(graph)]

    g = gplot(graph, nodelabel=node_labels_vector, nodefillc=node_colors, nodesize=0.5, nodelabelsize=10)

    display(g)
end

function load_and_clean_data()::DataFrame
    iris = dataset("datasets", "iris")
    cleaned_data = dropmissing(iris)
    return unique(cleaned_data)
end

function prepare_data(data::DataFrame)::Tuple{Matrix{Float64}, Vector{Int}}
    X = Matrix{Float64}(data[:, 1:end-1])
    Y = Int.(levelcode.(data[:, end]))
    return X, Y
end

function print_table(depths::Vector{Int}, accuracies::Vector{Float64}, correct_counts::Vector{Int}, filename::String)
    data = hcat(depths, round.(accuracies, digits=2), correct_counts)
    headers = ["Глубина", "Точность", "Кол-во правильно классифиц."]

    md_content = String[]

    push!(md_content, "| $(join(headers, " | ")) |")
    push!(md_content, "| $(join(fill("---", length(headers)), " | ")) |")

    for row in eachrow(data)
        formatted_row = "| "
        formatted_row *= string(row[1], " ") * " | "
        formatted_row *= string(round(row[2], digits=2), " ") * " | "
        formatted_row *= string(row[3], " ") * " |"
        push!(md_content, formatted_row)
    end

    open(filename, "w") do file
        write(file, join(md_content, "\n"))
    end
end

function evaluate_tree_depths(X::Matrix{Float64}, Y::Vector{Int}, depths::Vector{Int}, filename::String)
    Random.seed!(21)
    indices = shuffle(1:size(X, 1))
    train_size = round(Int, 0.7 * length(indices))
    train_indices = indices[1:train_size]
    test_indices = indices[train_size + 1:end]

    X_train = X[train_indices, :]
    Y_train = Y[train_indices]
    X_test = X[test_indices, :]
    Y_test = Y[test_indices]

    accuracies = Float64[]
    correct_counts = Int[]

    for depth in depths
        tree = fit(X_train, Y_train, depth)
        accuracy, correct_count = score(tree, X_test, Y_test)
        push!(accuracies, accuracy)
        push!(correct_counts, correct_count)
    end

    if filename == "results.md"
        print_table(depths, accuracies, correct_counts, filename)
    end
    return accuracies
end

function find_best_depth(X::Matrix{Float64}, Y::Vector{Int}, depths::Vector{Int}, filename::String)::Tuple{Int, Float64}
    accuracies = Float64[]

    for depth in depths
        accuracy = mean(evaluate_tree_depths(X, Y, [depth], filename))
        push!(accuracies, accuracy)
    end

    best_index = argmax(accuracies)[1]
    best_depth = depths[best_index]
    best_accuracy = accuracies[best_index]

    open(filename, "a") do file
        write(file, "Лучший уровень глубины: $best_depth, Точность: $best_accuracy\n")
    end

    return best_depth, best_accuracy
end

function plot_accuracy_vs_depth(depths::Vector{Int}, accuracies::Vector{Float64}, filename::String)
    plot(depths, accuracies, label="Точность", xlabel="Глубина дерева", ylabel="Точность",
         title="Зависимость точности от глубины дерева", legend=:topright,
         marker=:circle, linecolor=:blue, grid=true)

    plot_path = "accuracy_vs_depth.png"
    savefig(plot_path)

    md_content = """


    ![Зависимость точности от глубины дерева](accuracy_vs_depth.png)
    """

    open(filename, "a") do file
        write(file, md_content)
    end
end

function evaluate_stability(X::Matrix{Float64}, Y::Vector{Int}, best_depth::Int, n_removals::Int, n_trials::Int)
    stability_results = DataFrame(Trial=Int[], Total_Train=Int[], Total_Test=Int[], Correct=Int[], Incorrect=Int[])

    for trial in 1:n_trials
        indices = shuffle(1:size(X, 1))
        train_size = round(Int, 0.7 * length(indices))
        train_indices = indices[1:train_size]
        test_indices = indices[train_size + 1:end]

        X_train = X[train_indices, :]
        Y_train = Y[train_indices]
        X_test = X[test_indices, :]
        Y_test = Y[test_indices]

        removed_indices = sort(indices[1:n_removals])
        keep_indices = setdiff(train_indices, removed_indices)

        X_train_reduced = X[keep_indices, :]
        Y_train_reduced = Y[keep_indices]

        tree = fit(X_train_reduced, Y_train_reduced, best_depth)
        _, correct_count = score(tree, X_test, Y_test)
        incorrect_count = length(Y_test) - correct_count

        push!(stability_results, (trial, size(X_train_reduced, 1), size(X_test, 1), correct_count, incorrect_count))
    end

    return stability_results
end

function plot_stability_results(results::DataFrame, filename::String)
    header = ["Испытание", "Общее количество в обучающей выборке", "Общее количество в тестовой выборке", "Правильно классифицировано", "Неправильно классифицировано"]

    table_string = join(header, " | ") * "\n" * join(fill("---", length(header)), " | ") * "\n"

    for row in eachrow(results)
        row_string = join(row, " | ")
        table_string *= row_string * "\n"
    end

    open(filename, "a") do file
        write(file, table_string)
    end
end

iris = load_and_clean_data()
X, Y = prepare_data(iris)
depths_tree = [1, 3, 5, 7, 10]
accuracies = evaluate_tree_depths(X, Y, depths_tree, "results.md")
plot_accuracy_vs_depth(depths_tree, accuracies, "results.md")
best_depth, best_accuracy = find_best_depth(X, Y, depths_tree, "results_stab.md")
stability_results = evaluate_stability(X, Y, best_depth, 10, 3)
plot_stability_results(stability_results, "results_stab.md")
