function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

H = X*theta;
theta_noZero = [0;theta(2:size(theta))];

J = (1/m)*(-y'*log(sigmoid(H))-(1-y)'*log(1-sigmoid(H))) + ((lambda/(2*m)) * theta_noZero'*theta_noZero);

%EYEWITHZERO = eye(size(X,2));
%EYEWITHZERO(1) = 0;


grad = (1/m) * (X' * (sigmoid(H) - y) + lambda * theta_noZero);

%grad = pinv(X'*X + lambda * EYEWITHZERO) * X'*y;

% =============================================================

end
