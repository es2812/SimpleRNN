#Lecturade fichero
lines = csvread("../originalFiles/dat_entrada.csv");
num_dias = length(lines(:,1));

longitud_ventana = 20; #marca el numero de valores que cosiderar para cada prediccion.

num_secuencias = num_dias-longitud_ventana;
test =  zeros(num_secuencias,longitud_ventana+1);

#extraemos todas las posibles secuencias, con el ultimo valor de la secuencia 
#siendo el valor a predecir
for i=1:num_secuencias
  test(i,:) = lines(i:i+longitud_ventana)';
endfor

#2/3 para entrenamiento
tam_training = ceil(num_secuencias*(2/3));
#1/3 para test
tam_test = num_secuencias - tam_training;

#Hold-out NO randomizado
indices_training = [1:tam_training];
training = test(indices_training,:);
test(indices_training,:) = [];

%SE NORMALIZA CADA SECUENCIA POR SEPARADO

capa_oculta = 4;
capa_entrada = 1; #solo hay una neurona en la capa de entrada
capa_auxiliar = capa_oculta; #una neurona por cada neurona de la capa oculta
capa_salida =  1;

factor_aprendizaje = 0.3;
factor_inercia = 0.3;
bias = 1;
max_epocas = 500;

#pesos de la conexion entre la capa de entrada y la capa oculta
wx = (rand(capa_oculta,capa_entrada)*10)-5;
#peso del bias de la capa oculta
wxb = (rand(capa_oculta,1))*10-5;
#pesos de la conexión entre capa auxiliar y capa oculta
wo = (rand(capa_oculta,capa_auxiliar)*10)-5;
#pesos de la conexion entre la capa oculta y la neurona de salida
ws = (rand(capa_salida,capa_oculta)*10)-5;
#pesos de la conexion bias de la neurona de salida
wsb = (rand(capa_salida,1)*10)-5;

incremento_s = zeros(size(ws));
incremento_s_b = zeros(size(wsb));
incremento_x = zeros(size(wx));
incremento_x_b = zeros(size(wxb));
incremento_o = zeros(size(wo));

#inercias
inercia_s = zeros(size(ws));
inercia_s_b = zeros(size(wsb));
inercia_x = zeros(size(wx));
inercia_x_b = zeros(size(wxb));
inercia_o = zeros(size(wo));

grafica_error = zeros(max_epocas,1);

function result = sigmoid(x)
  result = (1./(1+exp(-x)));
  return;
endfunction

function result = dsigmoid(f)
  result = (f.*(1.-f));
  return;
endfunction


for epoca=1:max_epocas

  #entrada de retroalimentacion
  #para la primera secuencia es 0
  o = zeros(1,capa_auxiliar);
  
  #salida actual de las neuronas de la capa oculta
  y = zeros(1,capa_oculta);
  #salida actual de la neurona de salida
  z = zeros(1,capa_salida);
    
  for i=1:tam_training  
    #cada secuencia X(i) = {x(i+1),x(i+2),...,x(i+19)->x(i+20)}
    #compararemos la salida de la red con x(i+20)
    
    secuencia = training(i,1:longitud_ventana);
    salida_deseada = training(i,longitud_ventana+1);
    
    %Normalizacion con desbordamiento del 20%
    min_des = min([secuencia,salida_deseada]);
    #min_des -= 0.05*min_des;
    max_des = max([secuencia,salida_deseada]);
    #max_des += 0.05*max_des;

    for j=1:longitud_ventana
      secuencia(j) = (secuencia(j)-min_des)/(max_des-min_des);  
    endfor
    salida_deseada = (salida_deseada-min_des)/(max_des-min_des);
    
    %fase hacia delante
    %se introduce valor a valor X(i) a la red
    for t=1:longitud_ventana
      x = secuencia(t); #x(i+t)
      o = y;
      y = sigmoid(x*wx'+wxb'+o*wo'); 
      #no es necesario calcular la salida de la capa de salida porque no calcularemos los delta hasta el ultimo valor.    
    endfor
    %salida a comparar
    z = sigmoid(y*ws'+wsb);
    
    %fase hacia atras    
    ds = (salida_deseada-z)*dsigmoid(z);#un número
    incremento_s = ds*y;
    incremento_s_b = ds;
    
    dd = (ds*ws').*(dsigmoid(y))';
    incremento_x = dd.*x;
    incremento_x_b = dd;
    incremento_o = dd*o;
    
    wx += factor_aprendizaje*incremento_x + factor_inercia*inercia_x;
    wxb += factor_aprendizaje*incremento_x_b + factor_inercia*inercia_x_b;
    wo += factor_aprendizaje*incremento_o + factor_inercia*inercia_o;
    ws += factor_aprendizaje*incremento_s + factor_inercia*inercia_s;
    wsb += factor_aprendizaje*incremento_s_b + factor_inercia*inercia_s_b;

    inercia_x = incremento_x;
    inercia_x_b = incremento_x_b;
    inercia_o = incremento_o;
    inercia_s = incremento_s;
    inercia_s_b = incremento_s_b;
  endfor
  
  %Validacion
  errores = 0;
  
  o = zeros(1,capa_auxiliar);
  y = zeros(1,capa_oculta);
  z = zeros(1,capa_salida);
  
  for i=1:tam_test
    secuencia = test(i,1:longitud_ventana);
    d = test(i,longitud_ventana+1);#salida deseada
    
    %Normalizacion
    min_des = min([secuencia,d]);
    #min_des -= 0.05*min_des;
    max_des = max([secuencia,d]);
    #max_des += 0.05*max_des;

    for j=1:longitud_ventana
      secuencia(j) = (secuencia(j)-min_des)/(max_des-min_des);  
    endfor
    d = (d-min_des)/(max_des-min_des);
    
    for t=1:longitud_ventana
      x = secuencia(t); #x(i+t)
      o = y;
      y = sigmoid(x*wx'+wxb'+o*wo');
    endfor
    z = sigmoid(y*ws'+wsb);    
  
    #se desnormaliza la salida de la red   
    z = (z*(max_des-min_des))+min_des;
    d = (d*(max_des-min_des))+min_des;
    #consideramos un acierto si la diferencia es menor que 0.05 centimos
    #printf("deseada %f obtenida %f\n",d,z);
    fflush(stdout);
    if abs(z-d)>0.05
      errores++;
    endif
  endfor
  grafica_error(epoca) = (errores/tam_test)*100;
  printf("Epoca %d tasa error %f %s\n",epoca,grafica_error(epoca),"%");
  fflush(stdout);
end

plot(grafica_error);