function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%temp=1/(2*m);
%temp1=0;
%for k = 1 : m
%temp1=temp1+(((theta(1)+(theta(2)*X(k,2)))-y(k))^2);
%end

%J=temp*temp1;

%J=(1/(2*m))*sum(((theta'*X')'-y).^2);
J=(1/(2*m))*sum(((X*theta)-y).^2);


% =========================================================================

end
