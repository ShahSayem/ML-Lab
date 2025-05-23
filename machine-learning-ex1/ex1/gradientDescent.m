function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    %temp2=0;
    %temp_t1=0;
    %temp_t2=0;
    %for l=1:m
    %    temp2=temp2+((theta(1)+(theta(2)*X(l,2)))-y(l));
    %end
    %temp_t1=theta(1)-(alpha*(1/m)*temp2);

    %temp2=0;
    %for o=1:m
    %    temp2=temp2+(((theta(1)+(theta(2)*X(o,2)))-y(o))*X(o,2));
    %end
    %temp_t2=theta(2)-(alpha*(1/m)*temp2);

    %theta(1)=temp_t1;
    %theta(2)=temp_t2;
    delta=(1/m)*(((X*theta)-y)'*X)';
    
    theta=theta-alpha*delta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
