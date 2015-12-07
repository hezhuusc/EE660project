function [J,grad] = cost_func(theta, X, y, lambda)
% Computer Likelihood Function and Gradient
m = length(y); % training examples
hx = sigmoid(X*theta);
J = (1./m)*sum(-y.*log(hx)-(1.0-y).*log(1.0-hx)) + (lambda./(2*m)*norm(theta(2:end))^2);
regularize = (lambda/m).*theta;
regularize(1) = 0;
grad = (1./m) .* X' * (y-hx) - regularize;
end