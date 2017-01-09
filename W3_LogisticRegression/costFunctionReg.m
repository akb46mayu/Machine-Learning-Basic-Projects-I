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

hXtheta = sigmoid(X*theta);
TEMP1 = -y.*log(hXtheta) - (1-y).*log(1-hXtheta);
TEMP2 = theta(2:end);
J = sum(TEMP1)/m + (lambda/(2*m))*sum(TEMP2.^2);
grad(1) = sum(hXtheta-y)/m;
for j = 2:size(X,2)
    grad(j) = subpTheta(hXtheta, X, y, theta, j, lambda);
    
end


% =============================================================

end


function thetap = subpTheta(hXtheta, X, y, theta, j, lambda)

m = length(y); % number of training examples
thetap = sum((hXtheta-y).*X(:,j))/m + (lambda/m)*theta(j);

end
