#Lectura de fichero
lines = csvread("dat_entrada.csv");
num_dias = length(lines(:,1));
#entrada = [[1:num_dias]',lines]#se anade una columna que nos indica el indice

longitud_ventana = 20; #marca el número de valores que cosiderar para cada predicción.
ventana = zeros(longitud_ventana,1);

%Normalización con desbordamiento del 20%
min_des = min(lines);
min_des -= 0.2*min_des;
max_des = max(lines);
max_des += 0.2*max_des;

for i=1:num_dias
  lines(i) = (lines(i)-min_des)/(max_des-min_des);
endfor

capa_oculta = 20;
capa_entrada = longitud_ventana + capa_oculta; #la capa entrada incluye la salida de la capa oculta para t-1
capa_salida =  1;

#pesos de la conexión entre la capa de entrada y la capa oculta
ww = (rand(capa_oculta,capa_entrada)*10)-5;
#pesos de la conexión entre la capa oculta y la neurona de salida
ws = (rand(capa_salida,capa_oculta)*10)-5;

#salida actual de las neuronas de la capa oculta
yy = zeros(1,capa_oculta);
#salida de la época anterior de las neuronas de la capa oculta. Para t=0 es 0
yyt = zeros(1,capa_oculta);
#salida actual de la neurona de salida
ys = zeros(1,capa_salida);

iteraciones = num_dias-longitud_ventana;

for i=1:iteraciones
  ventana = lines(i:i+longitud_ventana-1);
  
  entrada = [ventana',yyt];
  
  %fase hacia delante
  yy = 1./(1.+exp(-(entrada*ww')));
  yt = yy;
  
  ys = atan(yy*ws');
  
end