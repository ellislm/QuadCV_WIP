close all
clear all
clc

redThresh = 0.1;

% Motion Capture Data File
%MC_data = dlmread('20150619_ft_4_mw.txt',',');
videoFReader = vision.VideoFileReader('20150619_ft_4_wc_cut2.mp4');
videoPlayer = vision.VideoPlayer;
videoFWriter = vision.VideoFileWriter('20150619_ft_1_cut_out.avi');
hblob = vision.BlobAnalysis('AreaOutputPort', false, ... % Set blob analysis handling
    'CentroidOutputPort', true, ...
    'BoundingBoxOutputPort', true', ...
    'MinimumBlobArea', 100, ...
    'MaximumBlobArea', 3000, ...
    'MaximumCount', 10);
hshapeinsBox = vision.ShapeInserter('BorderColorSource', 'Input port', ... % Set box handling
    'Fill', true, ...
    'FillColorSource', 'Input port', ...
    'Opacity', 0.4);
htextinsRed = vision.TextInserter('Text', 'Red   : %2d', ... % Set text for number of blobs
    'Location',  [5 2], ...
    'Color', [1 0 0], ... // red color
    'Font', 'Courier New', ...
    'FontSize', 14);
htextinsCent = vision.TextInserter('Text','+X:%2d,Y:%2d', ... % set text for centroid
    'LocationSource', 'Input port', ...
    'Color', [0 0 0], ... // black color
    'Font', 'Courier New', ...
    'FontSize', 20);
htextinsCent2 = vision.TextInserter('Text','+X:%2f,Y:%2f', ... % set text for centroid
    'LocationSource', 'Input port', ...
    'Color', [0 0 0], ... // black color
    'Font', 'Courier New', ...
    'FontSize', 20);
y_pix_max = 480;%1080;
x_pix_max = 640;%1920;
x_pix = x_pix_max/2;
y_pix = y_pix_max/2;
hVideoIn = vision.VideoPlayer('Name','Final Video','Position', [0 0 y_pix_max x_pix_max]);


% initialize kalman filter
phi = [1 1 0 0 0 0 0 0;
       0 1 0 0 0 0 0 0;
       0 0 1 1 0 0 0 0;
       0 0 0 1 0 0 0 0;
       0 0 0 0 1 1 0 0;
       0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 1 1;
       0 0 0 0 0 0 0 1];
   
h = [1 0 0 0 0 0 0 0;
     0 0 1 0 0 0 0 0;
     0 0 0 0 1 0 0 0;
     0 0 0 0 0 0 1 0];
 
r = 0.25;
q = eye(8)*0.1;
pm = eye(8);
xm = [10; 0; 10; 0; 1.4; 0; 0; 0;];

% initialize variables
centRed_prev = [];
nFrame = 1;
frame_rate = 30; % Hz
qLoc = [9.7,10.5,1.4,-5/360*pi];

time = 0;
l_mean = 260;

%%% TDF ADDED %%%
l_mean_initial = l_mean;

%%%%%%%%%%%%%%%%%
index = 0;
while ~isDone(videoFReader) && index < 900
    index = index + 1;
    rgbFrame = step(videoFReader);
    % rgbFrame = step(htextinsCent, rgbFrame, [uint16(x_pix) uint16(y_pix)], [uint16(x_pix-6) uint16(y_pix-9)]);
    clear centroidRed bboxRed
    diffFrameRed = imsubtract(rgbFrame(:,:,1), rgb2gray(rgbFrame)); % Get red component of the image
    diffFrameRed = medfilt2(diffFrameRed, [3 3]); % Filter out the noise by using median filter
    binFrameRed = im2bw(diffFrameRed, redThresh); % Convert the image into binary image with the red objects as white
    [centroidRed, bboxRed] = step(hblob, binFrameRed); % Get the centroids and bounding boxes of the red blobs
    
    if index < 0
        continue
    end
    
    l_cntr = 0;
    clear l dLoc
    for i=1:length(centroidRed)-1 % no repetition
        for j=i+1:length(centroidRed(:,1)) % no repetition
            l_cntr = l_cntr+1;
            l(l_cntr) = sqrt((centroidRed(i,1)-centroidRed(j,1))^2 + (centroidRed(i,2)-centroidRed(j,2))^2);
            dLoc(l_cntr,1) = abs(centroidRed(i,1)-centroidRed(j,1));
            dLoc(l_cntr,2) = abs(centroidRed(i,2)-centroidRed(j,2));
            xs(l_cntr,1) = centroidRed(i,1)-centroidRed(j,1);
            ys(l_cntr,1) = centroidRed(i,2)-centroidRed(j,2);
        end
    end

    bin_cntr = 0;
    l_sum = 0;
    x_dist = 0;
    y_dist = 0;
    
    x_dist = sum(xs);
    y_dist = sum(ys);
    x_store(nFrame) = x_dist;
    y_store(nFrame) = y_dist;
   
    ang1(nFrame) = atan2(x_dist,y_dist)*180/3.14;
    ang2(nFrame) = atan2(y_dist,x_dist)*180/3.14;
    for i=1:l_cntr
       if l(i) < l_mean*1.3 && l(i) > l_mean * 0.7
           bin_cntr = bin_cntr + 1;
           l_bin(bin_cntr) = i;
           l_sum = l_sum + l(i);          
       end
    end
    l_mean = l_sum/(bin_cntr);
    l_m(nFrame) = l_mean;
    qLocPsum(1:2) = [0,0];
    ePsum = 0;
    clear pRed pRed2 qLocP ex ey eP
    for i=1:length(centroidRed)
        pRed(i,1) = qLoc(1) + ((centroidRed(i,1) - x_pix)/l_mean)*cos(qLoc(4)) - ((centroidRed(i,2) - y_pix)/l_mean)*sin(qLoc(4));
        pRed(i,2) = qLoc(2) + ((centroidRed(i,2) - y_pix)/l_mean)*sin(qLoc(4)) + ((centroidRed(i,2) - y_pix)/l_mean)*cos(qLoc(4));
        pRed2(i,1) = round(pRed(i,1)); % correct marker locations
        pRed2(i,2) = round(pRed(i,2)); % correct marker locations
        rgbFrame = step(htextinsCent, rgbFrame, [uint16(pRed(i,1)) uint16(pRed(i,2))], [centroidRed(i,1) centroidRed(i,2)]);
        
        eP(i) = sqrt((pRed(i,1) - pRed2(i,1))^2+(pRed(i,1) - pRed2(i,1))^2); % error in marker locations
        qLocP(i,1) = pRed2(i,1) - ((centroidRed(i,1) - x_pix)/l_mean)*cos(qLoc(4)) + ((centroidRed(i,2) - y_pix)/l_mean)*sin(qLoc(4));
        qLocP(i,2) = pRed2(i,2) - ((centroidRed(i,2) - y_pix)/l_mean)*sin(qLoc(4)) - ((centroidRed(i,2) - y_pix)/l_mean)*cos(qLoc(4));        
        
        qLocPsum(1) = qLocPsum(1) + (1/eP(i)) * qLocP(i,1);
        qLocPsum(2) = qLocPsum(2) + (1/eP(i)) * qLocP(i,2);
        ePsum = ePsum + (1/eP(i));
    end
    e_m(nFrame) = mean(eP(:));
    
    flag = 1;
    clear angvar
    for i=1:length(pRed2)
        for j=1:length(pRed2)
            if pRed2(i,1) == pRed2(j,1) + 1 && pRed2(i,2) == pRed2(j,2) % is i 1 unit vertical above j
                dx = pRed(i,1) - pRed(j,1);
                dy = pRed(i,2) - pRed(j,2);
                dxx(flag) = dx;
                dyy(flag) = dy;
                inds_i(flag) = i;
                inds_j(flag) = j;
                angvar(flag) = atan2(-dy,dx)*180/pi
                flag = flag + 1;
            end
            if pRed2(i,2) == pRed2(j,2) + 1 && pRed2(i,1) == pRed2(j,1) % is i 1 unit vertical above j
                dx = centroidRed(i,1) - centroidRed(j,1);
                dy = centroidRed(i,2) - centroidRed(j,2);
                angvar(flag) = atan2(dx,dy)*180/pi;
                flag = flag + 1;
            end
            
        end
    end
   
    qLocP;
    eP';
    qLoc(1) = qLocPsum(1) / ePsum;
    qLoc(2) = qLocPsum(2) / ePsum;
    qLoc(3) = 322.5806*1/l_mean;
    qLoc(4) = median(angvar)*pi/180;
    qLoc'
    
    %if (nFrame == 23)
        %ardvark
    %end
    %rgbFrame = step(htextinsCent2, rgbFrame, [uint16(mean(qLocP(:,1))*100) uint16(mean(qLocP(:,2))*100)], [x_pix y_pix]);
    
    %% kalman filter
    % compute Kalman gain, figure 4.1
    ka = pm*h'/(h*pm*h'+r);
    % take measurement z(k)
    z = [mean(qLocP(:,1)) mean(qLocP(:,2)) 322.5806*1/l_mean qLoc(4)]';
    %z = [qLoc(1) qLoc(2) qLoc(3) qLoc(4)]';
    
    zL(nFrame,1:4) = z(1:4);
    % update estimate with measurement z(k), figure 4.1
    xh = xm+ka*(z-h*xm);
    % Compute error covariance for updated estimate, figure 4.1
    p = (eye(8)-ka*h)*pm;
    % Project ahead, figure 4.1
    xm = phi*xh;
    pm = phi*p*phi'+q;
    p_trace(nFrame) = trace(nFrame);
    % locate corners from current position
    rgbFrame = step(htextinsCent2, rgbFrame, [xh(1) xh(3)], [uint16(x_pix) uint16(y_pix)]);

    qLoc(1) = xh(1) + xh(2)*1/frame_rate;
    qLoc(2) = xh(3) + xh(4)*1/frame_rate;
    qLoc(3) = xh(5) + xh(6)*1/frame_rate;
    qLoc(4) = xh(7) + xh(8)*1/frame_rate;
   
    ang_kalman(nFrame) = qLoc(4);
    x(nFrame,1:3) = qLoc(1:3);
    v(nFrame,1) = xh(2);
    v(nFrame,2) = xh(4);
    v(nFrame,3) = xh(6);
    plot(x(nFrame,1),x(nFrame,2),'rx')
    grid on
    axis([0 20 0 20])
       
    step(videoFWriter, rgbFrame);
    step(hVideoIn, rgbFrame); % Output video stream
    nFrame = nFrame + 1;
    time(nFrame) = time(nFrame-1) + 1/frame_rate;
end
release(videoPlayer);
release(videoFReader);
release(videoFWriter);
% clear empty variables
time(1) = [];

figure(2)
title('Cam Position, XYZ')
subplot(3,1,1)
plot(time,x(:,1),'g')
ylabel('X-Posit Cam')
subplot(3,1,2)
plot(time,x(:,2),'g')
ylabel('Y-Posit Cam')
subplot(3,1,3)
plot(time,x(:,3),'g')
ylabel('Z-Posit Cam')
xlabel('Time')
grid on

figure(3)
subplot(3,1,1)
plot(time,v(:,1),'g')
ylabel('X-Vel Cam')
subplot(3,1,2)
plot(time,v(:,2),'g')
ylabel('Y-Vel Cam')
subplot(3,1,3)
plot(time,v(:,3),'g')
ylabel('Z-Vel Cam')
xlabel('Time')
grid on

figure(4)
plot(time,e_m,'r')
ylabel('Cam Marker Prediction Error')
xlabel('Time')

%% Analyzing Motion Capture Data for comparison
load('Trial06.mat');
output(:,2) = output(:,2)/1000;
output(:,3:4) = output(:,3:4)/1000;

output2 = output;
output2(:,2) = -output(:,3);
output2(:,3) = -output(:,2);

x(:,1) = x(:,1) - x(1,1);
x(:,2) = x(:,2) - x(1,2);

x2 = x.*0.3048;

output2(:,2) = output2(:,2) - output2(1,2);
output2(:,3) = output2(:,3) - output2(1,3);

qXYZ  = quaternion.eulerangles('XYZ',output2(:,5)*pi/180,output2(:,6)*pi/180,output2(:,7)*pi/180);
angles = EulerAngles(qXYZ,'ZYX')';
angles = angles*180/pi;

qXYZ2=SpinCalc('EA123toQ',output2(:,5:7)*pi/180,1e-5,0);
angles2 = SpinCalc('QtoEA321',qXYZ2, 1e-5, 0);
angles2 = angles*180/pi;

figure(5)
subplot(3,1,1)
plot(output2(:,1), output2(:,2))
ylabel('X-Pos MoCap')
subplot(3,1,2)
plot(output2(:,1), output2(:,3))
ylabel('Y-Pos MoCap')
subplot(3,1,3)
plot(output2(:,1), output2(:,4))
ylabel('Z-Pos MoCap')

[r, lag] = xcorr(x2(:,1),output2(:,2));

[~,I] = max(abs(r));
lagDiff = -lag(I);

%output2(1:lagDiff,:) = [];
%% for video 4 and workspace 6
output2(:,1) = output2(:,1) - output2(1,1)-1.69;

for k = 1:length(output2)
    if (output2(k,1) >= time(end))
        break;
    end
end
output2(k:end,:) = [];
angles(k:end,:) = [];
angles2(k:end,:) = [];

for k = 1:length(output2)
    if (output2(k,1) >= 0)
        break;
    end
end

output2(1:k,:) = [];
angles(1:k,:) = [];
angles2(1:k,:) = [];

ang3 = -atan2(-x_store, y_store)*180/pi;
ang4 = atan2(-y_store, x_store)*180/pi;

% figure(6)
% 
% subplot(3,1,1)
% set(gca,'fontsize',14)
% hold on
% plot(output2(:,1), output2(:,2), time, x2(:,1))
% xlabel('Time [sec]')
% ylabel('Distance [ft]')
% legend('Motion Capture','Cam Estimate')
% subplot(3,1,2)
% set(gca,'fontsize',14)
% hold on
% plot(output2(:,1), output2(:,3), time, x2(:,2))
% xlabel('Time [sec]')
% ylabel('Distance [ft]')
% subplot(3,1,3)
% plot(output2(:,1), angles(:,1)-angles(500,1), time,Theta-Theta(1), time, -(ang1-ang1(1,1)))%, time, ang2-ang2(1,1))%, time, ang4)
% 

figure(7)
subplot(3,1,1)
set(gca,'fontsize',14)
hold on
plot(output2(:,1), output2(:,2), time, x2(:,1))
xlabel('Time [sec]')
ylabel('Distance [ft]')
legend('Motion Capture','Cam Estimate')
subplot(3,1,2)
set(gca,'fontsize',14)
hold on
plot(output2(:,1), output2(:,3), time, x2(:,2)*480/640)
xlabel('Time [sec]')
ylabel('Distance [ft]')
subplot(3,1,3)
plot(output2(:,1), output2(:,4), time,x2(:,3), time, 322.5806./l_m)
xlabel('Time [sec]')
ylabel('Altitude [ft]')
legend('Motion Capture','Cam Estimate', '322.5806/l_m')

o2(:,1) = interp1(output2(:,1),output2(:,2),time);
o2(:,2) = interp1(output2(:,1),output2(:,3),time);
o2(:,3) = interp1(output2(:,1),output2(:,4),time);

E(:,1) = o2(1:end-1,1)-x2(1:end-1,1);
E(:,2) = o2(1:end-1,2)-x2(1:end-1,2);
E(:,3) = o2(1:end-1,3)-x2(1:end-1,3);

mE(1) = mean(E(:,1));
mE(2) = mean(E(:,2));
mE(3) = mean(E(:,3));

x2(:,1) = x2(:,1) + mE(1);
x2(:,2) = x2(:,2) + mE(2);
x2(:,3) = x2(:,3) + mE(3);

E(:,1) = o2(1:end-1,1)-x2(1:end-1,1);
E(:,2) = o2(1:end-1,2)-x2(1:end-1,2);
E(:,3) = o2(1:end-1,3)-x2(1:end-1,3);

mE(1) = mean(E(:,1));
mE(2) = mean(E(:,2));
mE(3) = mean(E(:,3));

sE(1) = std(E(:,1));
sE(2) = std(E(:,2));
sE(3) = std(E(:,3));

maE(1) = mean(abs(E(:,1)));
maE(2) = mean(abs(E(:,2)));
maE(3) = mean(abs(E(:,3)));

figure(8)
set(gca,'fontsize',16, 'fontname', 'Times')
plot(time,o2(:,1),'g',time,x2(:,1),'b',time(1:end-1),abs(E(:,1)),'r','LineWidth',2)
legend('Motion Capture','Cam Estimate','Residual')
grid on
xlabel('Time (s)')
ylabel('Position X (m)')
figure(9)
set(gca,'fontsize',16, 'fontname', 'Times')
plot(time,o2(:,2),'g',time,x2(:,2),'b',time(1:end-1),abs(E(:,2)),'r','LineWidth',2)
legend('Motion Capture','Cam Estimate','Residual')
grid on
xlabel('Time (s)')
ylabel('Position Y (m)')
figure(10)
set(gca,'fontsize',16, 'fontname', 'Times')
plot(time,o2(:,3),'g',time,x2(:,3),'b',time(1:end-1),abs(E(:,3)),'r','LineWidth',2)
legend('Motion Capture','Cam Estimate','Residual')
grid on
xlabel('Time (s)')
ylabel('Position Z (m)')