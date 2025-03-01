import os
import shutil
from sklearn.model_selection import train_test_split

# Define the root directories where data exists, including KAGGLE/AUDIO
data_folders = [
    'for-2sec/for-2seconds/',
    'for-norm/for-norm/',
    'for-original/for-original/',
    'for-rerec/for-rerecorded/',
    'KAGGLE/AUDIO/'
]

# Create directories for unified train/test/validation structure
base_dir = 'cleaned_data/'
train_real_dir = os.path.join(base_dir, 'train/real/')
train_fake_dir = os.path.join(base_dir, 'train/fake/')
test_real_dir = os.path.join(base_dir, 'test/real/')
test_fake_dir = os.path.join(base_dir, 'test/fake/')
val_real_dir = os.path.join(base_dir, 'validation/real/')
val_fake_dir = os.path.join(base_dir, 'validation/fake/')

os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_fake_dir, exist_ok=True)
os.makedirs(test_real_dir, exist_ok=True)
os.makedirs(test_fake_dir, exist_ok=True)
os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_fake_dir, exist_ok=True)

# Function to move files to the correct train/test/validation directories
def organize_files(source_dir):
    # Define subfolders for training, testing, and validation
    for split in ['training', 'testing', 'validation']:
        real_dir = os.path.join(source_dir, split, 'real')
        fake_dir = os.path.join(source_dir, split, 'fake')

        if os.path.exists(real_dir):
            real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
        else:
            real_files = []

        if os.path.exists(fake_dir):
            fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
        else:
            fake_files = []

        # Move files to the appropriate unified folder
        if split == 'training':
            for f in real_files:
                shutil.move(f, train_real_dir)
            for f in fake_files:
                shutil.move(f, train_fake_dir)
        elif split == 'testing':
            for f in real_files:
                shutil.move(f, test_real_dir)
            for f in fake_files:
                shutil.move(f, test_fake_dir)
        elif split == 'validation':
            for f in real_files:
                shutil.move(f, val_real_dir)
            for f in fake_files:
                shutil.move(f, val_fake_dir)

# Organize files from each dataset folder
for folder in data_folders[:-1]:  # Exclude KAGGLE/AUDIO for now
    organize_files(folder)


print("Data organized into unified train/test/validation splits successfully!")
