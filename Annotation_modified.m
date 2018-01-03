% %
% PH, PW: height and width respectively of the patch images 
% souce: source image
% specify the path to your images folder in "path"
% N: number of training images 
% create two folder "crack" and "no_crack" under the specified path. The
% training images will be saved in this directory
% click with mouse once done labeling the cracks in the image
% % 

PH = 127;
PW = 127;
path = 'D:/CMU/UAV/';
N = 100;
thr = 8; % how much crack is included in the image to be counted as crack image
names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
num_images = 2;
l=7;
%source = strcat(strcat('Crack_',int2str(l)),'.jpg');
source = strcat('Crack_20','.jpg');
I=imread(strcat(path,source));
[H,W,~] = size(I);
imshow(I);
h=imfreehand;
BW = createMask(h);

cumulative_Mask = zeros(size(BW)); 
while sum(BW(:)) > 5
      cumulative_Mask = cumulative_Mask | BW;
      h = imfreehand;
      BW = createMask( h );
end


% getting the mask in RGB form
RGB_Mask = uint8(zeros(size(I)));
for i = 1:H
    for j = 1:W        
        RGB_Mask(i,j,1) = cumulative_Mask(i,j);
        RGB_Mask(i,j,2) = cumulative_Mask(i,j);
        RGB_Mask(i,j,3) = cumulative_Mask(i,j);
    end 
end

res = immultiply(RGB_Mask,I);
imshow(res)

% cropping the image patches and saving them
Y = randi([1 H-PH],1,N);
X = randi([1 W-PW],1,N);


for j = 1:N
    patch = imcrop(res,[X(j),Y(j),PH,PW]);
    if sum(patch) < thr
        im_patch = imcrop(I,[X(j),Y(j),PH,PW]);
        patch_name = strcat(strcat(path, 'no_crack/im2'), int2str(j),names(l));
        imwrite(im_patch,strcat(patch_name,'.jpg'));
        %Constrast
        im_contrast = imadjust(im_patch,[.2 .25 0.1; .7 .75 0.8],[]);
        patch_name_contrast = strcat(strcat(path, 'no_crack/imcontr2'), int2str(j),names(l));
        imwrite(im_contrast,strcat(patch_name_contrast,'.jpg'));
    else 
        im_patch = imcrop(I,[X(j),Y(j),PH,PW]);
        patch_name = strcat(strcat(strcat(path, 'crack/im2'), int2str(j)),names(l));
        imwrite(im_patch,strcat(patch_name,'.jpg'));
        %rotate Image
        im_rot = imrotate(im_patch,90);%Rotate by 90 degrees
        patch_name_rot = strcat(strcat(strcat(path, 'crack/imrot2'), int2str(j)),names(l));
        imwrite(im_rot,strcat(patch_name_rot,'.jpg'));
        %Gaussian Filter
        im_gauss = imgaussfilt(im_patch,2);
        patch_name_gauss = strcat(strcat(strcat(path, 'crack/imguass2'), int2str(j)),names(l));
        imwrite(im_gauss,strcat(patch_name_gauss,'.jpg'));
        %Adjust contrast
        im_contrast = imadjust(im_patch,[.2 .25 0.1; .7 .75 0.8],[]);
        patch_name_contrast = strcat(strcat(strcat(path, 'crack/imcontr2'), int2str(j)),names(l));
        imwrite(im_contrast,strcat(patch_name_contrast,'.jpg'));
    end 
end