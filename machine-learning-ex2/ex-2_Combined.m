%% Logistic Regression (Linear + Regularized)
% Combined Script: Includes all parts of Exercise 2

clear; close all; clc

%% ========== Function Definitions ==========

function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end

function [J, grad] = costFunction(theta, X, y)
    m = length(y);
    h = sigmoid(X * theta);
    J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h));
    grad = (1/m) * (X' * (h - y));
end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);
    h = sigmoid(X * theta);
    theta_reg = [0; theta(2:end)];
    J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + ...
        (lambda / (2 * m)) * sum(theta_reg .^ 2);
    grad = (1/m) * (X' * (h - y)) + (lambda / m) * theta_reg;
end

function p = predict(theta, X)
    p = sigmoid(X * theta) >= 0.5;
end

function out = mapFeature(X1, X2)
    degree = 6;
    out = ones(size(X1(:, 1)));
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X1.^(i-j)) .* (X2.^j);
        end
    end
end

function plotData(X, y)
    pos = find(y == 1); 
    neg = find(y == 0);
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    hold on;
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
end

function plotDecisionBoundary(theta, X, y)
    plotData(X(:,2:3), y);
    hold on
    if size(X, 2) <= 3
        plot_x = [min(X(:,2))-2, max(X(:,2))+2];
        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
        plot(plot_x, plot_y)
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
    else
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);
        z = zeros(length(u), length(v));
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j)) * theta;
            end
        end
        z = z';
        contour(u, v, z, [0, 0], 'LineWidth', 2)
        legend('y = 1', 'y = 0', 'Decision boundary')
    end
    hold off
end

%% ========== Part 1: Logistic Regression ==========

fprintf('\n=== Logistic Regression (Linearly Separable Data) ===\n');
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
plotData(X, y);
xlabel('Exam 1 score'); ylabel('Exam 2 score');
legend('Admitted', 'Not admitted');
pause;

[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Initial Cost: %f\n', cost);
fprintf('Initial Gradients:\n'); disp(grad);
pause;

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('Optimized Cost: %f\n', cost);
fprintf('Optimized Theta:\n'); disp(theta);

plotDecisionBoundary(theta, X, y);
xlabel('Exam 1 score'); ylabel('Exam 2 score');
legend('Admitted', 'Not admitted', 'Decision Boundary');
pause;

prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, probability of admission: %f\n', prob);
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
pause;

%% ========== Part 2: Regularized Logistic Regression ==========

fprintf('\n=== Regularized Logistic Regression (Non-linearly Separable Data) ===\n');
data = load('ex2data2.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
plotData(X, y);
xlabel('Microchip Test 1'); ylabel('Microchip Test 2');
legend('y = 1', 'y = 0');
pause;

X = mapFeature(X(:,1), X(:,2));
initial_theta = zeros(size(X, 2), 1);

for lambda = [0 1 100]
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
    
    fprintf('\nLambda = %d\n', lambda);
    fprintf('Cost: %f\n', J);
    fprintf('Theta:\n'); disp(theta);
    
    plotDecisionBoundary(theta, X, y);
    title(sprintf('Decision boundary (lambda = %g)', lambda));
    xlabel('Microchip Test 1'); ylabel('Microchip Test 2');
    pause;
    
    p = predict(theta, X);
    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    pause;
end
