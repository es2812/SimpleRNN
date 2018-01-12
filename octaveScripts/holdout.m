function [training, test] = holdout(data)
  
  tam_training = ceil((size(data)(1))*(2/3));
  tam_test = size(data)(1) - tam_training;
  
  indices_training = [1:tam_training];
  training = data(indices_training,:);
  data(indices_training,:) = [];
  test = data;
  
  return;  
endfunction