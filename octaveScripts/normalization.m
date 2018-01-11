function [normalized_data, max, min] = normalize(data, margin_percent)
  %will normalize vectors
  max = max(data);
  min = min(data);
  
  max += margin_percent*max;
  min += margin_percent*min;
  
  normalized_data = zeros(size(data));
  
  for i=1:size(data)
    normalized_data(i) = (data(i) - min)/(max-min);
  endfor
  
endfunction

function result = denormalize(data,max,min)
  
  result = data*(max-min)+min
  
endfunction