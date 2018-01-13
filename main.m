#anyadimos los paths de las funciones
addpath(genpath('octaveScripts/'));

capa_oculta = 10;
capa_entrada = 1; #solo hay una neurona en la capa de entrada
capa_salida =  1;

factor_aprendizaje = 0.1;
factor_inercia = 0.1;
max_epocas = 5000;
tam_ventana = 20;

ruta_fichero = "originalFiles/dat_entrada.csv";
#leemos fichero solo una vez
lines = csvread(ruta_fichero);

grafica_error_elman = zeros(max_epocas,1);
grafica_error_jordan = zeros(max_epocas,1);

printf("Evaluando la red Elman...\n");
grafica_error_elman = elman(lines,tam_ventana,capa_oculta,capa_entrada,capa_salida,factor_aprendizaje,factor_inercia, max_epocas);

figure(1);
plot(grafica_error_elman,'r-','LineWidth',4);
grid on;
title('Evolucion de la tasa de aciertos en el tiempo');
xlabel('Epoca');
ylabel('% aciertos');
legend ("Elman","location", "south") ;

printf("Evaluando la red Jordan...\n");
grafica_error_jordan = jordan(lines,tam_ventana,capa_oculta,capa_entrada,capa_salida,factor_aprendizaje,factor_inercia, max_epocas);

figure(2);
plot(grafica_error_jordan,'b-','LineWidth',4);
grid on;
title('Evolucion de la tasa de aciertos en el tiempo');
xlabel('Epoca');
ylabel('% aciertos');
legend ('Jordan',"location", "south") ;

figure(3);
plot(grafica_error_elman,'r-','LineWidth',4,grafica_error_jordan,'b-','LineWidth',4);
grid on;
title('Evolucion de la tasa de aciertos en el tiempo');
xlabel('Epoca');
ylabel('% aciertos');
legend ('Elman','Jordan','location', 'south') ;