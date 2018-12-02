f = @(x) 1.89*log(x)-1.518 ;

x=[0.01:0.1:450];
y = f(x);
plot(x,y);
plot(x,y);grid on;