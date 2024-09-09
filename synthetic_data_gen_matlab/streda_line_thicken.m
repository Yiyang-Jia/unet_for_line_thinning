    % background:
% This is the Synthetic sample for the Streda.
% There are 3 classes of lines:
% 1. Integer quantum Hall effect (IQH) or Chern insulators (ChIs):
% they are: y=(k-i)*x or x=i; where i is integer
% 2. Fractional Chern insulators (FCIs, someone also call it fractional quantum Hall effect):
% they are: y=(k-f)*x; where f is fractional, usually i+1/3 or i+2/3;
% 3. charge density wave (CDW, someone also call it generalized Wigner crystal):
% they are vertical lines: x=f; where f is fractional.

%% class 1
class1x_s=[-4 -3 -2 -1 0 1 2 3 4];
class1x_e=[-4 -3 -2 -1 0 1 2 3 4];
class1y_s=[rand rand 0 0 0 0 0 rand rand];     % this is the x/y coordinate of start/end points of lines
class1y_e=[rand rand 1 1 1 1 1 rand rand];

for i=-4:4             % for each x-intercept
    for i_k=1:12      % for each slope
        % for positive slope
        if rand<(1-i_k/12-abs(i)/5)  % we can omit one line
            k=1/i_k;
            if i_k<=4
                x_s_add=i+0.05*rand;
                y_s_add=k*(x_s_add-i);

                x_e_add=i+1+0.5*(1-i_k/12)*rand;
                y_e_add=k*(x_e_add-i);
                class1x_s=[class1x_s,x_s_add];
                class1x_e=[class1x_e,x_e_add];
                class1y_s=[class1y_s,y_s_add];
                class1y_e=[class1y_e,y_e_add];
            elseif abs(i)<=2
                x_s_add=i+0.5*rand;
                y_s_add=k*(x_s_add-i);
                x_e_add=i+(1.5-abs(i)/6*(i<-1))*rand;
                y_e_add=k*(x_e_add-i);
                class1x_s=[class1x_s,x_s_add];
                class1x_e=[class1x_e,x_e_add];
                class1y_s=[class1y_s,y_s_add];
                class1y_e=[class1y_e,y_e_add];
            end
        end

        if rand<(1-i_k/12-abs(i)/5)  % we can omit one line
            k=-1/i_k;
            if i_k<=4
                x_e_add=i-0.5-(1-i_k/12)*rand;
                y_e_add=k*(x_e_add-i);
                class1x_s=[class1x_s,i];
                class1x_e=[class1x_e,x_e_add];
                class1y_s=[class1y_s,0];
                class1y_e=[class1y_e,y_e_add];
            elseif abs(i)<=2
                x_s_add=i-0.1*(1-i_k/12)*rand;
                y_s_add=k*(x_s_add-i);
                x_e_add=i-(1.5-abs(i)/6*(i>1))*rand;
                y_e_add=k*(x_e_add-i);
                class1x_s=[class1x_s,x_s_add];
                class1x_e=[class1x_e,x_e_add];
                class1y_s=[class1y_s,y_s_add];
                class1y_e=[class1y_e,y_e_add];
            end
        end

    end
end


%% class 2
class2x_s=[];
class2x_e=[];
class2y_s=[];     % this is the x/y coordinate of start/end points of lines
class2y_e=[];

for i=[-3 -2 -1 1 2 3]             % for each x-intercept
    for i_k=1:4      % for each slope
        % for positive slope
        if rand<(1-i_k/12) && i>0  % we can omit one line
            k=1/i_k;
            a=rand;
            f=1/3*(a<1/2)+2/3*(a>1/2);

            x_s_add=i+0.5*(rand-0.8)+(1-i_k/12)*rand;
            y_s_add=k*(x_s_add-i-f);
            x_e_add=i+(1.5-i_k/12)*rand;
            y_e_add=k*(x_e_add-i-f);
            class2x_s=[class2x_s,x_s_add];
            class2x_e=[class2x_e,x_e_add];
            class2y_s=[class2y_s,y_s_add];
            class2y_e=[class2y_e,y_e_add];

        end

        % for negative slope
        if rand<(1-i_k/12) && i<0  % we can omit one line
            k=-1/i_k;
            a=rand;
            f=1/3*(a<1/2)+2/3*(a>1/2);

            x_s_add=i-0.5*(rand-0.8)-(1-i_k/12)*rand;
            y_s_add=k*(x_s_add-i-f);
            x_e_add=i-(1.5-i_k/12)*rand;
            y_e_add=k*(x_e_add-i-f);
            class2x_s=[class2x_s,x_s_add];
            class2x_e=[class2x_e,x_e_add];
            class2y_s=[class2y_s,y_s_add];
            class2y_e=[class2y_e,y_e_add];

        end

    end
end


%% class 3
class3x_s=[];
class3x_e=[];
class3y_s=[];     % this is the x/y coordinate of start/end points of lines
class3y_e=[];

num=round(8*rand);
for i=1:num
    a=rand;
    x=2*(randi(2)-1.5)*(randi(4)+1/2*(a<1/3)+1/3*(a>1/3 & a<2/3)+2/3*(a>2/3));
    y_s_add=rand;
    y_e_add=rand;
    class3x_s=[class3x_s,x];
    class3x_e=[class3x_e,x];
    class3y_s=[class3y_s,y_s_add];
    class3y_e=[class3y_e,y_e_add];
end


%% generate a figure
% from line figure to 2D figure
img = zeros(200,400);
twoDimage;

for iii=1:length(class1x_s)
    x1=50*class1x_s(iii)+200;
    x2=50*class1x_e(iii)+200;
    y1=200*class1y_s(iii);
    y2=200*class1y_e(iii);
    numPoints = 1000;
    x = linspace(x1, x2, numPoints);
    y = linspace(y1, y2, numPoints);
    for i = 1:numPoints
        x_pos=min(400,max(1,round(x(i))));
        y_pos=min(200,max(1,round(y(i))));
        img(y_pos,x_pos) = 2*z_grid(y_pos,x_pos);
    end
end

for iii=1:length(class2x_s)
    x1=50*class2x_s(iii)+200;
    x2=50*class2x_e(iii)+200;
    y1=200*class2y_s(iii);
    y2=200*class2y_e(iii);
    numPoints = 1000;
    x = linspace(x1, x2, numPoints);
    y = linspace(y1, y2, numPoints);
    for i = 1:numPoints
        x_pos=min(400,max(1,round(x(i))));
        y_pos=min(200,max(1,round(y(i))));
        img(y_pos,x_pos) = 0.4*z_grid(y_pos,x_pos);
    end
end

for iii=1:length(class3x_s)
    x1=50*class3x_s(iii)+200;
    x2=50*class3x_e(iii)+200;
    y1=200*class3y_s(iii);
    y2=200*class3y_e(iii);
    numPoints = 1000;
    x = linspace(x1, x2, numPoints);
    y = linspace(y1, y2, numPoints);
    for i = 1:numPoints
        x_pos=min(400,max(1,round(x(i))));
        y_pos=min(200,max(1,round(y(i))));
        img(y_pos,x_pos) = 0.4*z_grid(y_pos,x_pos);
    end
end


%%plot the original figure
figure; hold on
for ii=1:length(class1y_e)
    plot([class1x_s(ii),class1x_e(ii)],[class1y_s(ii),class1y_e(ii)],'k')
end

for ii=1:length(class2y_e)
    plot([class2x_s(ii),class2x_e(ii)],[class2y_s(ii),class2y_e(ii)],'k')
end

for ii=1:length(class3y_e)
    plot([class3x_s(ii),class3x_e(ii)],[class3y_s(ii),class3y_e(ii)],'k')
end

%title('streda lines')
xlim([-4 4])
ylim([0 1])
axis off
box off

%original figure but with random line thickness and random gray scale
figure; hold on

% random gray noise as background
% background_size = [200, 400];
% random_noise = rand(background_size);
% kernel_size = 2;
% kernel = fspecial('gaussian', [kernel_size kernel_size], kernel_size/4);
% smoothed_noise = imfilter(random_noise, kernel, 'replicate');
% background_gray = 0.2 + 0.1 * smoothed_noise; % Scale to lighter grays
% imagesc([-4 4], [1 0], background_gray);
% colormap(gray);

%   % random gray uniform color as background
% background_gray = 0.7 + 0.3 * rand(); % Random light gray value between 0.7 and 1.0
% set(gca, 'Color', [background_gray background_gray background_gray]);

min_width = 1; % Minimum line width
max_width = 15;   % Maximum line width
min_gray =0.2;
max_gray =1;
% Plot class 1 lines (random grayscale)
for ii = 1:length(class1y_e)
    random_width = min_width + (max_width - min_width) * rand();
    random_gray = min_gray +(max_gray - min_gray) * rand(); % Random value between 0 (black) and 1 (white)
    plot([class1x_s(ii), class1x_e(ii)], [class1y_s(ii), class1y_e(ii)], 'Color', [random_gray random_gray random_gray], 'LineWidth', random_width)
end

% Plot class 2 lines (random grayscale)
for ii = 1:length(class2y_e)
    random_width = min_width + (max_width - min_width) * rand();
    random_gray = rand(); % Random value between 0 (black) and 1 (white)
    plot([class2x_s(ii), class2x_e(ii)], [class2y_s(ii), class2y_e(ii)], 'Color', [random_gray random_gray random_gray], 'LineWidth', random_width)
end

% Plot class 3 lines (random grayscale)
for ii = 1:length(class3y_e)
    random_width = min_width + (max_width - min_width) * rand();
    random_gray = rand(); % Random value between 0 (black) and 1 (white)
    plot([class3x_s(ii), class3x_e(ii)], [class3y_s(ii), class3y_e(ii)], 'Color', [random_gray random_gray random_gray], 'LineWidth', random_width)
end

axis off 
box off

xlim([-4 4])
ylim([0 1])

% figure('Position', [100, 100, 800, 400]); % Create a larger figure
% hold on
% 
% min_width = 5; % Minimum line width
% max_width = 15; % Maximum line width
% min_gray = 0.2; % Minimum grayscale value
% max_gray = 1; % Maximum grayscale value
% 
% % Plot lines with random width and grayscale
% for class_data = {[class1x_s; class1x_e; class1y_s; class1y_e], ...
%                   [class2x_s; class2x_e; class2y_s; class2y_e], ...
%                   [class3x_s; class3x_e; class3y_s; class3y_e]}
%     data = class_data{1};
%     for ii = 1:size(data, 2)
%         random_width = min_width + (max_width - min_width) * rand();
%         random_gray = min_gray + (max_gray - min_gray) * rand();
%         plot([data(1,ii), data(2,ii)], [data(3,ii), data(4,ii)], ...
%              'Color', [random_gray random_gray random_gray], 'LineWidth', random_width);
%     end
% end
% 
% axis off 
% xlim([-4 4])
% ylim([0 1])
% 
% % Capture the plot as an image
% frame = getframe(gcf);
% img = frame2im(frame);
% img = rgb2gray(img); % Convert to grayscale
% img = im2double(img); % Convert to double precision
% 
% % Add salt and pepper noise
% noise_density = 0.02; % Adjust this value to control the amount of salt and pepper noise
% img_noisy = imnoise(img, 'salt & pepper', noise_density);
% 
% % Add stripe noise using FFT
% F = fft2(img_noisy);
% [rows, cols] = size(F);
% 
% % Horizontal stripes
% F(floor(rows/2), :) = F(floor(rows/2), :) + ...
%     150 * (rand(1, cols) + 1i * rand(1, cols));
% 
% % Vertical stripes
% F(:, floor(cols/2)) = F(:, floor(cols/2)) + ...
%     300 * (rand(rows, 1) + 1i * rand(rows, 1));
% 
% img_stripe_noise = real(ifft2(F));
% 
% % Display the final noisy image
% figure('Position', [100, 100, 800, 400]);
% imshow(img_stripe_noise, []);
% colormap(gray);
% axis off;
% xlim([-4 4])
% ylim([0 1])
% title('Plot with Random Width, Grayscale, Salt & Pepper, and Stripe Noise');

% figure;hold on
% min_width = 5; % Minimum line width
% max_width = 15;   % Maximum line width
% 
% % Plot class 1 lines (black)
% for ii=1:length(class1y_e)
%     random_width = min_width + (max_width - min_width) * rand();
%     plot([class1x_s(ii),class1x_e(ii)],[class1y_s(ii),class1y_e(ii)],'k', 'LineWidth', random_width)
% end
% 
% % Plot class 2 lines (black)
% for ii=1:length(class2y_e)
%     random_width = min_width + (max_width - min_width) * rand();
%     plot([class2x_s(ii),class2x_e(ii)],[class2y_s(ii),class2y_e(ii)],'k', 'LineWidth', random_width)
% end
% 
% % Plot class 3 lines (black)
% for ii=1:length(class3y_e)
%     random_width = min_width + (max_width - min_width) * rand();
%     plot([class3x_s(ii),class3x_e(ii)],[class3y_s(ii),class3y_e(ii)],'k', 'LineWidth', random_width)
% end
% 
% %title('streda lines')
% axis off 
% xlim([-4 4])
% ylim([0 1])



% blur out the lines
% [x_k,y_k]=meshgrid(1:50,1:50);
% Lolentz_kernel = 1./((x_k-25).^2+(y_k-25).^2+20);
% imgBlurred=conv2(img,Lolentz_kernel,'same');
% 
% noise_level = 0.02*max(imgBlurred,[],2);
% imgNoise = imgBlurred+noise_level.*randn(size(imgBlurred));
% 
% % filterSize = 50;
% % sigma = 3;  %<---------- change this
% % h = fspecial('gaussian', filterSize, sigma);
% % imgBlurred = imfilter(img/max(img,[],'all'), h);
% 
% % % add some noise
% % imgNoise = imnoise(imgBlurred, 'gaussian', 0, 0.001); 
% % % add Gussian noise; the last value is the noise strength
% % imgNoise = imnoise(imgNoise, 'salt & pepper', 0.005);
% % agg salt&pepper noise; the last value is the noise strength
% F = fft2(imgNoise);
% 
% % add stripe noise by fft2
% F(size(F,1)/2, :) = F(size(F,1)/2,:) +...
%     150*(rand(1,size(F,2)) + 1i*rand(1,size(F,2)));
% F(:,size(F,2)/2) = F(:,size(F,2)/2) +...
%     300*(rand(size(F,1),1) + 1i*rand(size(F,1),1));
% 
% imgStripeNoise = ifft2(F);
% imgStripeNoise = abs(imgStripeNoise); % 
% imgStripeNoise(1,:)=0;
% imgStripeNoise(:,1)=0;
% 
% data_z=sqrt(abs(imgNoise));
% data_z(data_z > 0.7*max(data_z,[],'all') ) = 0.7*max(data_z,[],'all');
% 
% figure;
% imagesc(linspace(-4,4,400),linspace(1,0,200),data_z);
% axis tight;
% 
% colormap(gray);
% colorbar;
% xlabel('filling')
% ylabel('B (a.u.)')

