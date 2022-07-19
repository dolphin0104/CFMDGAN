import os
import numpy as np
from PIL import Image
import math
import cv2
from collections import OrderedDict

#================================
# Set your paths
origin_dir = 'your/path/to/300VW Dataset'
extract_dir = 'your/path/to/300VW Dataset/extract frames'
sharp_dir = 'your/path/to/save/sharp ground-truth images/'
blur_dir = 'your/path/to/save/blur input images/'
#================================


def pil2np(img):
    return np.array(img)

def np2Pil(img):
    return Image.fromarray(img)

def make_dirs(apath):
    if not os.path.exists(apath):
        os.makedirs(apath)

def SquareResize(img):      
    w, h = img.size
    new_size = min(w, h)
    if (w == new_size and h == new_size):
        return img
    else:
        left = (w - new_size)/2
        top = (h - new_size)/2
        right = (w + new_size)/2
        bottom = (h + new_size)/2
        
        return img.crop((left, top, right, bottom))

def variance_of_laplacian(image):    
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def Resize(img, size=256):      
    size = (size, size)
    w, h = img.size
    if (w == h and w == size):
        return img
    else:        
        return img.resize(size, Image.BICUBIC)

def load_txt_file(file_path):
    '''
    load data or string from text file.
    '''
    with open(file_path, 'r') as cfile:
        content = cfile.readlines()
    cfile.close()
    content = [x.strip() for x in content]
    num_lines = len(content)
    return content, num_lines

def anno_parser(anno_path, num_pts):
    '''                        
    parse the annotation for 300W dataset, which has a fixed format for .pts file                                
    return:                    
    pts: 3 x num_pts (x, y, oculusion)                                
    '''                        
    data, num_lines = load_txt_file(anno_path)                          
    assert data[0].find('version: ') == 0, 'version is not correct'     
    assert data[1].find('n_points: ') == 0, 'number of points in second line is not correct'                     
    assert data[2] == '{' and data[-1] == '}', 'starting and end symbol is not correct'                          
                                
    assert data[0] == 'version: 1' or data[0] == 'version: 1.0', 'The version is wrong : {}'.format(data[0])
    n_points = int(data[1][len('n_points: '):])                         
                                
    assert num_lines == n_points + 4, 'number of lines is not correct'    # 4 lines for general information: version, n_points, start and end symbol      
    assert num_pts == n_points, 'number of points is not correct'
                                
    # read points coordinate   
    pts = np.zeros((3, n_points), dtype='float32')                      
    line_offset = 3    # first point starts at fourth line              
    point_set = set()
    for point_index in range(n_points):                                
        try:                     
            pts_list = data[point_index + line_offset].split(' ')       # x y format                                 
            if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates   
                pts_list = remove_item_from_list(pts_list, '')              
            pts[0, point_index] = float(pts_list[0])                        
            pts[1, point_index] = float(pts_list[1])                        
            pts[2, point_index] = float(1)      # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by 300-W   
            point_set.add( point_index )
        except ValueError:       
            print('error in loading points in %s' % anno_path)              
    return pts, point_set 

def PTSconvert2box(points, expand_ratio_w=0.5, expand_ratio_h=0.5):
    assert isinstance(points, np.ndarray) and len(points.shape) == 2, 'The points is not right : {}'.format(points)
    assert points.shape[0] == 2 or points.shape[0] == 3, 'The shape of points is not right : {}'.format(points.shape)
    if points.shape[0] == 3:
        points = points[:2, points[-1,:].astype('bool') ]
    elif points.shape[0] == 2:
        points = points[:2, :]
    else:
        raise Exception('The shape of points is not right : {}'.format(points.shape))
    assert points.shape[1] >= 2, 'To get the box of points, there should be at least 2 vs {}'.format(points.shape)
    box = np.array([ points[0,:].min(), points[1,:].min(), points[0,:].max(), points[1,:].max() ])
    W = box[2] - box[0]
    H = box[3] - box[1]
    assert W > 0 and H > 0, 'The size of box should be greater than 0 vs {}'.format(box)
    if expand_ratio_w is not None and expand_ratio_h is not None:
        box[0] = int( math.floor(box[0] - W * expand_ratio_w) )
        box[1] = int( math.floor(box[1] - H * expand_ratio_h) )
        box[2] = int( math.ceil(box[2] + W * expand_ratio_w) )
        box[3] = int( math.ceil(box[3] + H * expand_ratio_h) )
    return box
#================================


TRAIN_DIRS = [
    '001', '002', '003', '007', '013', '015', '016', '017', '018', '019',
    '020', '022', '025', '027', '028', '029', '031', '033', '034', '035', 
    '041', '043', '044', '046', '047', '048', '049', '057', '059', '112',
    '113', '114', '115', '119', '123', '125', '126', '138', '143', '144',
    '150', '160', '203', '208', '212', '213', '214', '218', '223', '225',
    '401', '402', '403', '404', '405', '408', '412', '505', '506', '507',
    '508', '510', '511', '514', '517', '519', '520', '521', '524', '525',
    '526', '528', '529', '530', '531', '540', '541', '546', '547', '548',
    '553', '559', '562'
    ]
TEST_DIRS = ['009', '010', '037', '039', '053', '158', '211', '406', '522']

N_FRAMES = [5, 7, 9, 11, 13]
THRESHOLD = 100

make_dirs(sharp_dir)
make_dirs(blur_dir)

count = 1


for ttt, TDIRS in enumerate([TRAIN_DIRS, TEST_DIRS]):
    if ttt == 0:
        txtfile = open('D:/002.Projects/01.Dataset/300VW_Dataset_2015_12_14/300VW_Dataset_LTB/train_filelist_210826.txt', 'w')            
    else:
        txtfile = open('D:/002.Projects/01.Dataset/300VW_Dataset_2015_12_14/300VW_Dataset_LTB/test_filelist_210826.txt', 'w')
    for sub_dirs in TDIRS:
        extracts = os.path.join(extract_dir, sub_dirs)
        
        sharps = os.path.join(sharp_dir, sub_dirs)
        # make_dirs(sharps)

        blurs = os.path.join(blur_dir, sub_dirs)
        # make_dirs(blurs)

        annots = os.path.join(origin_dir, sub_dirs, 'annot')        
        extract_frames = sorted(os.listdir(extracts))

        for nf in N_FRAMES:
            blur_frames = None
            blur_sub_dir = os.path.join(blurs, 'blur{:03d}'.format(nf)) 
            # make_dirs(blur_sub_dir)

            sharp_sub_dir = os.path.join(sharps, 'blur{:03d}'.format(nf)) 
            # make_dirs(sharp_sub_dir)

            sampled_frames = [extract_frames[i:i+nf] for i in range(0, len(extract_frames), 5)]

            for rf in sampled_frames:
                if len(rf)==nf:
                    rf = sorted(rf)
                    
                    top_x = 0.
                    top_y = 0.
                    box_w = 0.
                    box_h = 0.                   

                    annot_dict = OrderedDict()

                    for i, im in enumerate(rf): 
                        extracted_img_path = os.path.join(extracts, im)
                        filename = os.path.splitext(os.path.split(extracted_img_path)[-1])[0]
                        correct_filename = '0'+filename
                        annot_file = os.path.join(annots, correct_filename + '.pts')
                        points, _ = anno_parser(annot_file, 68)
                                        
                        box = PTSconvert2box(points)

                        left_x, left_y = int(points[0][36]), int(points[1][36])
                        annot_dict['{}'.format(correct_filename)] = (left_x, left_y)
                                            
                        top_x += box[0]
                        top_y += box[1]
                        box_w += box[2]
                        box_h += box[3]
                                
                    top_x /= len(rf)
                    top_y /= len(rf)
                    box_w /= len(rf)
                    box_h /= len(rf)
                    box = (top_x, top_y, box_w, box_h)
                    check_fm = 1
                    for i, im in enumerate(rf):                 
                        extracted_img_path = os.path.join(extracts, im)
                        filename = os.path.splitext(os.path.split(extracted_img_path)[-1])[0]                
                        img = Image.open(extracted_img_path)
                        crop_img = img.crop(box)
                        crop_img = SquareResize(crop_img)
                        crop_img = Resize(crop_img)
                        
                        # use numpy to convert the pil_image into a numpy array
                        numpy_image=np.array(crop_img)  
                        # convert to a openCV2 image and convert from RGB to BGR format
                        opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                        fm = variance_of_laplacian(gray)
                        if fm >= THRESHOLD:
                            check_fm *= 1
                        else:
                            check_fm *= 0
                    
                    if check_fm:
                        make_dirs(sharp_sub_dir)
                        make_dirs(sharp_sub_dir)

                        mid_frame = (nf-1)//2
                        mid_filename = os.path.splitext(os.path.split(rf[mid_frame])[-1])[0]
                        mid_frame_filename = '0'+ mid_filename

                        sharp_sub_sub_dir = os.path.join(sharp_sub_dir, mid_frame_filename)
                        make_dirs(sharp_sub_sub_dir)
                        
                        for i, im in enumerate(rf):                 
                            extracted_img_path = os.path.join(extracts, im)
                            filename = os.path.splitext(os.path.split(extracted_img_path)[-1])[0]                
                            img = Image.open(extracted_img_path)
                            crop_img = img.crop(box)
                            crop_img = SquareResize(crop_img)
                            crop_img = Resize(crop_img)
                            correct_filename = '0'+filename                                                
                            crop_img.save(os.path.join(sharp_sub_sub_dir, correct_filename +'.png'))                         
                            
                            img_np = pil2np(crop_img)    
                            if i == 0:
                                blur = np.zeros(img_np.shape).astype(np.float32)
                            blur += img_np 
                                        
                        blur = (blur/len(rf)).clip(0, 255).astype(np.uint8) 
                        b_img = np2Pil(blur)
                        
                        savename = os.path.join(blur_sub_dir, mid_frame_filename+'.png')
                        # print(savename)
                        b_img.save(savename)

                        sorted_dict = OrderedDict(sorted(annot_dict.items(), key=lambda item: item[1]))
                        assert len(sorted_dict) == nf
                        itemlist = list(sorted_dict.keys())
                        log = " ".join(str(item) for item in itemlist)
                        logg = '{} blur{:03d} {} {}\n'.format(
                            sub_dirs, nf, mid_frame_filename, log)
                        count = count + 1
                        print(count, logg)
                        txtfile.write(logg)
    txtfile.close()

       