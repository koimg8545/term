% load image
image1 = imread("images/wall/img4.png");
image2 = imread("images/wall/img1.png");

image1 = cast(im2gray(image1),'double')/255;
image2 = cast(im2gray(image2),'double')/255;

G = fspecial('gaussian',3,1);
image1 = imfilter(image1,G);
image2 = imfilter(image2,G);

imshowpair(image1, image2,'montage')
%%
% operate SIFT with different size images
[key_features1, feature_descriptors1, image1_2] = sift(image1, 1);
[key_features1_2, feature_descriptors1_2, image1_3] = sift(image1_2, 2);
[key_features1_3, feature_descriptors1_3, image1_4] = sift(image1_3, 3);
[key_features1_4, feature_descriptors1_4, image1_5] = sift(image1_4, 4);

% concatenate keypoints information and feature descriptors from each image
key_features1 = [key_features1; key_features1_2; key_features1_3;key_features1_4];
feature_descriptors1 = [feature_descriptors1; feature_descriptors1_2; feature_descriptors1_3;feature_descriptors1_4 ];

% operate SIFT with different size images
[key_features2, feature_descriptors2, image2_2] = sift(image2, 1);
[key_features2_2, feature_descriptors2_2, image2_3] = sift(image2_2, 2);
[key_features2_3, feature_descriptors2_3, image2_4] = sift(image2_3, 3);
[key_features2_4, feature_descriptors2_4, image2_5] = sift(image2_4, 4);

% concatenate keypoints information and feature descriptors from each image
key_features2 = [key_features2; key_features2_2; key_features2_3;key_features2_4];
feature_descriptors2 = [feature_descriptors2; feature_descriptors2_2; feature_descriptors2_3; feature_descriptors2_4];

%%
row1 = key_features1(:,1);
col1 = key_features1(:,2);
row2 = key_features2(:,1);
col2 = key_features2(:,2);

% calculate ssd using feature diescriptor
ssd = zeros(size(feature_descriptors1,1),size(feature_descriptors2,1));
for i = 1:size(key_features1,1)
    for j = 1:size(key_features2,1)
        ssd(i,j) = sum(sum((feature_descriptors1(i,:) - feature_descriptors2(j,:)).^2));
    end
end


ssd_im = imresize(ssd, 2,'nearest');
imshow(rescale(ssd_im));

%%
% remain features that distance between second nearest
% neighborhood is low compare to first nearest neighbor hood
first_second_nearest = mink(ssd,2,1);
first_second_ratio = first_second_nearest(1,:) ./ first_second_nearest(2,:);

ratio_treshold = 0.8;
col_filt_ssd = ssd(:,first_second_ratio < ratio_treshold);
row2 = row2(first_second_ratio < ratio_treshold,:);
col2 = col2(first_second_ratio < ratio_treshold,:);
[feat1_index, feat2_index] = find(col_filt_ssd == min(col_filt_ssd,[],1));
filt_ssd = col_filt_ssd(feat1_index,:);
row1 = row1(feat1_index);
col1 = col1(feat1_index);
imshow(rescale(imresize(filt_ssd, 2,'nearest')));

%%
% RANSAC
[trans_mat_iter, match_count_iter, max_inlier, trans_mat, feat_index] = homography_ransac(row2, col2, row1, col1, 100000);

tform = projective2d(transpose(trans_mat));
homography_im = imwarp(image1, tform);
disp(max_inlier)
imshow(homography_im)

%%
% overlap images on the big image

% get matched features for translate image and overlapping
domain_feat_index = feat_index;
range_feat_index = feat_index;

% make big image
big_image = zeros(2000,5000,3);

% find a location of the feature in transformed image, for translating
feature_image1 = zeros(size(image1));
feature_image1(row1(range_feat_index), col1(range_feat_index)) = 1;
feature_image1 = imwarp(feature_image1, tform);

% find a location of the feature in original image
[feature_row1, feature_col1] = find(feature_image1);
[feature_row2, feature_col2] = deal(row2(range_feat_index), col2(range_feat_index));

% get displacement for overlapping
translate_row = feature_row1(1) - feature_row2;
translate_col = feature_col1(1) - feature_col2;

% overlap images using displacement
big_image(300+translate_row:299+translate_row+size(image2,1),...
                300+translate_col:299+translate_col+size(image2,2),1) = image2;
big_image(300:299+size(homography_im,1),300:299+size(homography_im,2),2) =...
    big_image(300:299+size(homography_im,1),300:299+size(homography_im,2),2) + homography_im;

imshow(big_image)

%%
% get inliers for drawing line between matched features
inliers = zeros(size(col2,1),1);
trans_vec = trans_mat * transpose([col1 row1 ones(size(row1, 1),1)]);
adjust_one = repmat(trans_vec(3,:),3,1);
trans_vec = trans_vec ./ adjust_one;
trans_vec = trans_vec(1:2,:);

[trans_row, trans_col] = deal(trans_vec(2,:), trans_vec(1,:));

% count number of inliers using error threshold
match_count=0;
for j = 1:size(trans_vec,2)
    error = sqrt((row2(j) - trans_row(j))^2 + (col2(j) - trans_col(j))^2);
    if error < 5
        match_count = match_count+1;
        inliers(j) = 1;
    end
end

% show matching lines
figure; ax = axes;
showMatchedFeatures(image1,image2,[col1(inliers>0) row1(inliers>0)],[col2(inliers>0) row2(inliers>0)],'montage','Parent',ax);


%%
function [key_features, feature_descriptor, half_im1] = sift(image1, octave)
    height = size(image1,1);
    width = size(image1,2);

    % implement scale space using Gaussian filter
    scaled_images = zeros(5,height,width);
    for i = (1:5)
        sigma_iter = (2^(i/2))/2;
        G = fspecial('gaussian',round(3*sigma_iter),sigma_iter);
        scaled_images(i,:,:) = imfilter(image1,G);
    end

    % get dog images
    dog = zeros(4,height, width);
    for i = (1:4)
        dog_image = squeeze(scaled_images(i+1,:,:)) - squeeze(scaled_images(i,:,:));
        dog(i,11:height-10,11:width-10) = dog_image(11:height-10,11:width-10);
    end

    % non‐maxima suppression
    feature1 = zeros(height, width);
    feature2 = zeros(height, width);
    index = 2;
    for i = 12:height-11
        for j = 12:width-11
            if dog(index,i,j) == max(max([dog(index-1,i-1:i+1,j-1:j+1) dog(index,i-1:i+1,j-1:j+1) dog(index+1,i-1:i+1,j-1:j+1)]))...
                    || dog(index,i,j) == min(min([dog(index-1,i-1:i+1,j-1:j+1) dog(index,i-1:i+1,j-1:j+1) dog(index+1,i-1:i+1,j-1:j+1)]));
                feature1(i,j) = dog(index,i,j);
            end
        end
    end

    % non‐maxima suppression
    index = 3;
    for i = 12:height-11
        for j = 12:width-11
            if dog(index,i,j) == max(max([dog(index-1,i-1:i+1,j-1:j+1) dog(index,i-1:i+1,j-1:j+1) dog(index+1,i-1:i+1,j-1:j+1)]))...
                   || dog(index,i,j) == min(min([dog(index-1,i-1:i+1,j-1:j+1) dog(index,i-1:i+1,j-1:j+1) dog(index+1,i-1:i+1,j-1:j+1)]));
                feature2(i,j) = dog(index,i,j);
            end
        end
    end

    % discard low contrast features
    thresh_feature1 = (feature1>0.03) + (feature1 < -0.03);
    thresh_feature2 = (feature2>0.03) + (feature2 < -0.03);

    dog_image1 = squeeze(dog(2,:,:));
    dog_image2 = squeeze(dog(3,:,:));
    
    % get Harris operatored image for filtering
    corners1 = harris(dog_image1);
    corners2 = harris(dog_image2);
    
    sigma =  (2^(3/2))/2;
    collect_size = 2*floor(sigma*1.5*3/2)+1;
    half_collect = fix(collect_size/2);
    
    % make padding for prevent out of index
    thresh_feature1(1:half_collect,:) = 0;
    thresh_feature1(end-half_collect:end,:) = 0;
    thresh_feature1(:,1:half_collect) = 0;
    thresh_feature1(:,end-half_collect:end) = 0;
    thresh_feature2(1:half_collect,:) = 0;
    thresh_feature2(end-half_collect:end,:) = 0;
    thresh_feature2(:,1:half_collect) = 0;
    thresh_feature2(:,end-half_collect:end) = 0;
    
    % discard features on edges using Harris operator
    filt_features1 = corners1 .* thresh_feature1;
    filt_features2 = corners2 .* thresh_feature2;
    
    % save location of remaining features
    [row1, col1] = find(filt_features1);
    [row2, col2] = find(filt_features2);
    
    % half size image to return
    half_im1 = imresize(squeeze(scaled_images(4,:,:)),0.5);
    
    % get feature descriptors
    [key_features1, feature_descriptor1] = get_rotation(scaled_images, 2, row1, col1, octave, dog_image1);
    [key_features2, feature_descriptor2] = get_rotation(scaled_images, 3, row2, col2, octave, dog_image2);
    key_features = [key_features1;key_features2];
    feature_descriptor = [feature_descriptor1;feature_descriptor2];
    
    key_features(:,1:2) = 2^(octave-1) * key_features(:,1:2);
end
%%
function [key_features_list, feature_descriptor_list] = get_rotation(scaled_images, scale_num, row, col, octave, dog_image)
    height = size(squeeze(scaled_images(scale_num,:,:)),1);
    width = size(squeeze(scaled_images(scale_num,:,:)),2);

    % calculate gradient magnitude and orientation of each pixel
    sigma = (2^(scale_num/2))/2;
    L = squeeze(scaled_images(scale_num,:,:));
    Lx_minus = [zeros(height, 1) L(:,1:end-1)];
    Lx_plus = [L(:,2:end) zeros(height, 1)];
    Ly_minus = [L(2:end,:); zeros(1, width)];
    Ly_plus = [zeros(1, width); L(1:end-1,:)];
    m = sqrt((Lx_plus - Lx_minus).^2 + (Ly_plus - Ly_minus).^2);
    theta = atan2((Ly_plus - Ly_minus),(Lx_plus - Lx_minus));
    theta(theta<0) = 2*pi + theta(theta<0);
    theta = theta*180/pi;

    % size of around area to collect gradients
    collect_size = 2*floor(sigma*1.5*3/2)+1;
    half_size = fix(collect_size/2);

    % key point factors to return
    key_features_list = [row col sigma*ones(size(row,1),1) octave*ones(size(row,1),1) zeros(size(row,1),1)];
    feature_descriptor_list = zeros(size(row,1),128);
    for i = (1:size(row))
        % collect gradient around the key point
        collect_gauss = fspecial('gaussian',collect_size,sigma*1.5);
        collect_m = m(row(i) - half_size:row(i) + half_size,col(i) - half_size:col(i) + half_size) .* collect_gauss;
        collect_theta = theta(row(i) - half_size:row(i) + half_size,col(i) - half_size:col(i) + half_size);
        
        % make 36 bins histogram
        hist_theta = ceil(collect_theta/10);
        key_hist = zeros(36,1);
        for j = 1:36
            key_hist(j) = key_hist(j) + sum(collect_m(hist_theta == j),'all');
        end

        % get main orientation
        [max_vec, max_orientation] = max(key_hist);
        
        % set 0 to bins around main orientation
        key_hist(mod((max_orientation-1-1),36)+1) = 0;
        key_hist(mod((max_orientation-1+1),36)+1) = 0;
        key_hist(mod((max_orientation-1-2),36)+1) = 0;
        key_hist(mod((max_orientation-1+2),36)+1) = 0;
        key_hist(mod((max_orientation-1-3),36)+1) = 0;
        key_hist(mod((max_orientation-1+3),36)+1) = 0;
        key_num = 1;
        key_orientations = max_orientation * 10-5;
        
        % find another orientation that value > 0.8 * max_bin
        if (sum(key_hist>0.8 * max_vec)) > 1
            key_num = 2;
            [o, max2] = maxk(key_hist,2);
            key_orientations = [max_orientation; max2(2)] * 10-5;
        end
        
        % get feature descriptor
        feature_descriptor = get_descriptor(row(i),col(i), key_orientations, key_num, m, theta, dog_image);
        
        % add orientation and descriptors to key point factors
        key_features_list(i,5) = key_orientations(1);
        feature_descriptor_list(i,:) = feature_descriptor(1,:);
        if key_num > 1
            key_features_list = [key_features_list; row(i) col(i) sigma octave key_orientations(2)];
            feature_descriptor_list = [feature_descriptor_list; feature_descriptor(2,:)];
        end
    end
end
% end
%%
function [feature_descriptor] = get_descriptor(row, col, key_orientations, key_num, m, theta, dog_image)
    feature_descriptor = zeros(key_num, 128);
    for l = (1:key_num)
        
        % rotate keypoint window with inverse of orientation and subtract
        % orientation from gradient angle
        theta_rot_window = imrotate((theta(row-11:row+11,col-11:col+11)-key_orientations(l)),-key_orientations(l));
        m_rot_window = imrotate((m(row-11:row+11,col-11:col+11)),-key_orientations(l));

        rot_height = size(theta_rot_window,1);
        rot_center = ceil(rot_height/2);

        % find center of window
        if mod(rot_height,2) ~= 1
            window_start = [rot_center-7 rot_center-7];
        else
            % find center of window using DOG value of neighbors
            L_rot = imrotate(dog_image(row-11:row+11,col-11:col+11),-key_orientations(l));
            neighbors = [L_rot(rot_center-1,rot_center-1) L_rot(rot_center-1,rot_center+1); L_rot(rot_center+1,rot_center-1) L_rot(rot_center+1,rot_center+1)];
            neighbors = abs(neighbors);
            [o, center] = max(neighbors);
            window_start = [(rot_center-9 + center(1)) (rot_center - 9 + center(2))];
        end
        
        % crop 16*16 window from rotated window from above process
        m_window =  m_rot_window(window_start(1):window_start(1) + 15, window_start(2):window_start(2) + 15) .* fspecial('gaussian',16,8);
        theta_window = theta_rot_window(window_start(1):window_start(1) + 15, window_start(2):window_start(2) + 15);
        theta_window(theta_window < 0) = 360 + theta_window(theta_window < 0);
        
        % make 128d feature descriptor
        theta_window = ceil(theta_window/45);
        feature_finger_print = zeros(128,1);
        for i = (1:4)
            for j = (1:4)
                sub_window_m = m_window(4*(i-1)+1:4*(i-1)+4,4*(j-1)+1:4*(j-1)+4);
                sub_window_theta = theta_window(4*(i-1)+1:4*(i-1)+4,4*(j-1)+1:4*(j-1)+4);
                for k = (1:8)
                    feature_finger_print(32*(i-1)+8*(j-1)+k) = feature_finger_print(32*(i-1)+8*(j-1)+k) + sum(sub_window_m(sub_window_theta == k),'all');
                end
            end
        end
        % normalize feature descriptor for illumination invarience
        first_normal = feature_finger_print ./ sqrt(sum(feature_finger_print.^2));
        
        % normalize again with threshold 0.2
        first_normal(first_normal > 0.2) = 0.2;
        feature_descriptor(l,:) = first_normal ./ sqrt(sum(first_normal.^2));
    end
end

%%
function strong_features = harris(image)
    filty = [1 2 1; 0 0 0; -1 -2 -1];
    filtx = filty';
     
    % get gradient
    grad_x = filter2(filtx, image);
    grad_y = filter2(filty, image);

    sumfilt = ones(2,2);

    grad_x_pow = grad_x.^2;
    grad_y_pow = grad_y.^2;
    grad_xy = grad_x.*grad_y;
    

    grad_x_pow_sum = filter2(sumfilt, grad_x_pow);
    grad_y_pow_sum = filter2(sumfilt, grad_y_pow);
    grad_xy_sum = filter2(sumfilt, grad_xy);

    det = grad_x_pow_sum .* grad_y_pow_sum - grad_xy_sum.^2;
    trace = grad_x_pow_sum + grad_y_pow_sum;
    
    % get edge filter using Harris operator
    r = 10;
    harris_threshold = ((r+1)^2)/r;
    R = (trace.^2) ./ det;
    R(1:10,:) = 100;
    R(end-10:end,:) = 100;
    R(:,1:10) = 100;
    R(:,end-10:end) = 100;
    strong_features = R<harris_threshold;
    
    disp(sum(sum(strong_features>0)));
end

%%
function [trans_mat_iter, match_count_iter, max_inner, homography_mat, feat_index] = homography_ransac(row1, col1, row2, col2, iter)
    % save transform matrices, matched features, 
    % index of matched feature in each iteration
    trans_mat_iter = [];
    match_count_iter = [];
    feat_index_iter = [];
    for i = 1:iter
        % randomly choose 4 matching features
        points_r = randperm(size(row1,1), 4);
        [x_r1, x_r2, x_r3, x_r4] = deal(col1(points_r(1)), col1(points_r(2)), col1(points_r(3)),col1(points_r(4)));
        [y_r1, y_r2, y_r3, y_r4] = deal(row1(points_r(1)), row1(points_r(2)), row1(points_r(3)), row1(points_r(4)));

        [x_d1, x_d2, x_d3, x_d4] = deal(col2(points_r(1)), col2(points_r(2)), col2(points_r(3)), col2(points_r(4)));
        [y_d1, y_d2, y_d3, y_d4] = deal(row2(points_r(1)), row2(points_r(2)), row2(points_r(3)), row2(points_r(4)));

        % formula for obtaining homography matrix
        domain = [x_d1 y_d1 1 0 0 0 -x_d1*x_r1 -y_d1*x_r1;...
                        x_d2 y_d2 1 0 0 0 -x_d2*x_r2 -y_d2*x_r2;...
                        x_d3 y_d3 1 0 0 0 -x_d3*x_r3 -y_d3*x_r3;...
                        x_d4 y_d4 1 0 0 0 -x_d4*x_r4 -y_d4*x_r4;...
                        0 0 0 x_d1 y_d1 1 -x_d1*y_r1 -y_d1*y_r1;...
                        0 0 0 x_d2 y_d2 1 -x_d2*y_r2 -y_d2*y_r2;...
                        0 0 0 x_d3 y_d3 1 -x_d3*y_r3 -y_d3*y_r3;...
                        0 0 0 x_d4 y_d4 1 -x_d4*y_r4 -y_d4*y_r4];
                        
        range = [x_r1; x_r2; x_r3; x_r4; y_r1; y_r2; y_r3; y_r4;];

        trans_var = domain\range;
        trans_mat = [trans_var(1) trans_var(2) trans_var(3);...
                            trans_var(4) trans_var(5) trans_var(6);...
                            trans_var(7) trans_var(8) 1];

        % transform locations of features for checking 
        % whether the point is matching and counting
        trans_vec = trans_mat * transpose([col2 row2 ones(size(row2, 1),1)]);
        adjust_one = repmat(trans_vec(3,:),3,1);
        trans_vec = trans_vec ./ adjust_one;
        trans_vec = trans_vec(1:2,:);
   
        [trans_row, trans_col] = deal(trans_vec(2,:), trans_vec(1,:));
        
        % count number of inliers using error threshold
        match_count=0;
        for j = 1:size(trans_vec,2)
            error = sqrt((row1(j) - trans_row(j))^2 + (col1(j) - trans_col(j))^2);
            if error < 5
                match_count = match_count+1;
                feat_index_iter(i) = points_r(1);
            end
        end
        % save matrix and inlier count
        trans_mat_iter(:,:,i) = trans_mat;
        match_count_iter(i) = match_count;
    end
    % get max inlier number, index, homography matrix, index of matched feature
    max_inner = max(match_count_iter);
    max_index = find(match_count_iter==max_inner);
    homography_mat = trans_mat_iter(:,:,max_index(1));
    feat_index = feat_index_iter(max_index(1));
end