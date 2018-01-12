function result = denormalize(data,max,min)
  result = data*(max-min)+min;
  return;
endfunction