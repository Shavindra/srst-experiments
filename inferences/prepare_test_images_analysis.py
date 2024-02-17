# %%
import os
import sys
sys.path.append('./utils')  # Adds the parent directory to the system path
sys.path.append('./models')  # Adds the parent directory to the system path


# %%
import dataloader as dl

# %%
CLASS_LIST = [
    #'background', 
    #'asphalt', 
    #'clinkers', 'grass', 
    #ß'mozaik', 'bike-asphalt', 'cars', 
    'tiles'
]

# CLASS_NAME = 'asphalt'

for CLASS_NAME in CLASS_LIST:
    print(f'Processing {CLASS_NAME}')
    # %%
    ANALYSIS_DIR_TEST_IMGS = f'/home/sfonseka/dev/SRST/srst-analysis/test_images/{CLASS_NAME}'
    IMG_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/images/512/{CLASS_NAME}'


    # %%
    os.makedirs(ANALYSIS_DIR_TEST_IMGS, exist_ok=True)

    # %%
    TEST_DIR = f'/projects/0/gusr51794/srst_scratch_drive/binary_training/test/512/{CLASS_NAME}'
    test_dataloader = dl.SRST_DataloaderGray(mask_dir=TEST_DIR, image_dir=IMG_DIR, mask_count=99999999999)
    test_dataset = test_dataloader.dataset

    with open(f'{ANALYSIS_DIR_TEST_IMGS}/test_list.txt', 'w') as f:
        # Write the file paths to the file
        for file_path in test_dataset.masks:
            f.write(file_path + '\n')
            print(file_path)

    # %%
    #Load the text file
    with open(f'{ANALYSIS_DIR_TEST_IMGS}/test_list.txt', 'r') as f:
        # Read the lines of the file
        test_files = f.readlines()

    # %%
    print(len(test_files))

    # %%
    import os, cv2
    from torchvision import transforms


    # %%
    import pickle
    to_pickle = []

    for file in test_files:
        file = file.strip()
        path = os.path.join('/projects/0/gusr51794/srst_scratch_drive/binary_training/baseline_test_images/512/', file.strip())

        print(path)

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(), # convert the image to a tensor
        ])
        img = img_transforms(img)
        img = img.unsqueeze(0) # add batch dimension to the image tensor
        img_np = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))

        
        to_pickle.append({
            'img_np': img_np,
            'img': img,
            'path': path
        })

    # %%
    print(len(to_pickle))

    # %%
    # Save pickle

    with open(f'{ANALYSIS_DIR_TEST_IMGS}/{CLASS_NAME}_test_images.pickle', 'wb') as f:
        pickle.dump(to_pickle, f)


