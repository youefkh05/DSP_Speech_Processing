clear; 
clc; 
close all;

epsilon1=1e-12;
N1=1024;
n=0:N1-1;
W_Rec1=ones(1,N1);
W_Han1=0.5-0.5*cos(2*pi*n/N1);
W_Ham1=0.54-0.46*cos(2*pi*n/N1);

W_Rec1=fft(W_Rec1,N1);
W_Han1=fft(W_Han1,N1);
W_Ham1=fft(W_Ham1,N1);

W_Rec1_dB=20*log10(abs(W_Rec1)+epsilon1);
W_Han1_dB=20*log10(abs(W_Han1));
W_Ham1_dB=20*log10(abs(W_Ham1));

W_Rec1_dB=fftshift(W_Rec1_dB);
W_Han1_dB=fftshift(W_Han1_dB);
W_Ham1_dB=fftshift(W_Ham1_dB);

fs1=16000;
f1=(-N1/2:N1/2-1)*(fs1/N1);
figure;
plot(f1,W_Rec1_dB);hold on;
plot(f1,W_Han1_dB);
plot(f1,W_Ham1_dB);
xlabel('Frequncy (Hz)')
ylabel('Magnitude (dB)')
title('Frequency domain of the three signals')
legend('W_Rec1','W_Han1','W_Ham1')
grid on;



