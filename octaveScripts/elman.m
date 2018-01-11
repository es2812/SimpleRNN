#Lectura de fichero
lines = csvread("../originalFiles/dat_entrada.csv");
num_dias = length(lines(:,1));

longitud_ventana = 20; #marca el numero de valores que cosiderar para cada prediccion.

num_secuencias = num_dias-longitud_ventana;
data =  zeros(num_secuencias,longitud_ventana+1);

#extraemos todas las posibles secuencias, con el ultimo valor de la secuencia 
#siendo el valor a predecir
for i=1:num_secuencias
  data(i,:) = lines(i:i+longitud_ventana)';
endfor

[training, test] = holdout(data);

capa_oculta = 10;
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

tam_training = size(training)(1);
tam_test = size(test)(1);

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
    
    secuencia = training(i,1:longitud_ventana+1);

    %Normalizacion con margen 0%    
    [secuencia,min,max] = normalize(secuencia,0);
    
    salida_deseada = secuencia(longitud_ventana+1);
    secuencia(longitud_ventana+1) = [];
    
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
    
    secuencia = test(i,1:longitud_ventana+1);
    
    [secuencia,max,min] = normalize(secuencia,0);   
    
    d = secuencia(longitud_ventana+1);
    secuencia(longitud_ventana+1) = [];
    
    for t=1:longitud_ventana
      x = secuencia(t); #x(i+t)
      o = y;
      y = sigmoid(x*wx'+wxb'+o*wo');
    endfor
    z = sigmoid(y*ws'+wsb);    
  
    #se desnormaliza la salida de la red  
    z = denormalize(z,max,min);
    d = denormalize(d,max,min); 
    
    #consideramos un acierto si la diferencia es menor que 0.05 centimos
    printf("deseada %f obtenida %f\n",d,z);
    fflush(stdout);
    if abs(z-d)>0.05
      errores++;
    endif
  endfor
  grafica_error(epoca) = (errores/tam_test)*100;
  printf("Epoca %d tasa error %f %s\n",epoca,grafica_error(epoca),"%");
  fflush(stdout);
end

#plot(grafica_error);