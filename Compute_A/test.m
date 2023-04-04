clear;clear all; close all;
I = phantom(512,512);
% I = double(rgb2gray(imread('001.bmp')));
figure
imshow(normimage(I))
% return

pixSpacing = 200/(size(I,1)*11/16); %%��λmm ��־����ͷ��ֱ��Ϊ200mm���� ռSheep-Loganͼ���11/16 ���Լ���ͷ��ÿ�����صĳ�Ϊ 200/11*16/256 = 1.1346
% SO = 682;   %��Դ����ת���ĵľ��� (mm)
% D = SO/pixSpacing;   %%��Դ����ת���ĵľ���(��λ������)��������ͶӰ 
D = round(512/382*540);
senSpacing = 0.1; % in degrees
rotIncr = 1; % in degrees

% D = round(512/382*540);

% t0 = cputime;
projData = fanbeam(I,D,'FanSensorSpacing',senSpacing,'FanRotationIncrement',rotIncr);%ʹ��fanbeam��ͶӰ����������ͼ
% t2 = cputime-t0
figure
imshow(normimage(projData))
% return
im1 = ifanbeam(projData,D, 'FanSensorSpacing',senSpacing, 'FanRotationIncrement',rotIncr);
figure
imshow(normimage(im1(3:end-2, 3:end-2)))

t0 = cputime;
% projData = fanbeamMy2(I,D,senSpacing,rotIncr,4);
% t1 = cputime-t0
% figure
% imshow(normimage(projData))

% t0 = cputime;
[im2, alpha3D, d2Grid3D, projDataWF] = ifanbeanMy(projData, D, senSpacing, rotIncr, size(I));
A = computeAmatrix(alpha3D, d2Grid3D, size(projDataWF,1));
im3 = reshape(A*reshape(projDataWF,[],1),size(I));
max(max(abs(im3-im2)))
t2 = cputime-t0
figure
imshow(normimage(im3))

