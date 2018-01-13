function [training, test]=preprocessing(lines,size_window)

  num_days = length(lines(:,1));

  num_sequences = num_days-size_window;
  data =  zeros(num_sequences,size_window+1);
  
  for i=1:num_sequences
    data(i,:) = lines(i:i+size_window)';
  endfor

  [training, test] = holdout(data);

endfunction