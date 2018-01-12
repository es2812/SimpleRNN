function grafica_error = jordan(data_filename,longitud_ventana,capa_oculta,capa_entrada,capa_salida,factor_aprendizaje,factor_inercia, max_epocas)
  [training,test] = preprocessing(data_filename,longitud_ventana);


  [wx,wxb,ws,wsb] = inicializeWeights(capa_entrada,capa_oculta,capa_salida);
  capa_auxiliar = capa_salida;
  wo = (rand(capa_oculta,capa_auxiliar)*10)-5;

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
        o = z;
        y = sigmoid(x*wx'+wxb'+o*wo'); 
        z = sigmoid(y*ws'+wsb);
      endfor
      
      %fase hacia atras    
      ds = (salida_deseada-z)*dsigmoid(z);#un numero
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
    aciertos = 0;
    
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
        o = z;
        y = sigmoid(x*wx'+wxb'+o*wo');
        z = sigmoid(y*ws'+wsb);
      endfor  
    
      #se desnormaliza la salida de la red  
      z = denormalize(z,max,min);
      d = denormalize(d,max,min); 
      
      #consideramos un acierto si la diferencia es menor que 0.05 centimos
      #printf("deseada %f obtenida %f\n",d,z);
      fflush(stdout);
      if abs(z-d)<0.05
        aciertos++;
      endif
    endfor
    grafica_error(epoca) = (aciertos/tam_test)*100;
    printf("Epoca %d tasa aciertos %f %s\n",epoca,grafica_error(epoca),"%");
    fflush(stdout);
  endfor
endfunction