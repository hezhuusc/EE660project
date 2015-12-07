function f=mapFeature(x)
% Map features to high dimension
degree = 6;
f = ones(size(x(:,1)));  
for i = 1:degree  
    for j = 0:i  
        f(:, end+1) = (x(:,1).^(i-j)).*(x(:,2).^j);
    end  
end
end
