#print("temp")


## Steps in making this:
# 1. Data preparation / data creation
# 2. train the model 
# 3. test model on performace


# we do not do this becuase we use a external dataset

# import os

# import cv2


# DATA_DIR = './data'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# number_of_classes = 3
# dataset_size = 100

# cap = cv2.VideoCapture(0) # 0 for my computer (2 for macOS maybe)

# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))

#     print('Collecting data for class {}'.format(j))

#     done = False
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
#                     cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('q'):
#             break

#     counter = 0
#     while counter < dataset_size:
#         ret, frame = cap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

#         counter += 1

# cap.release()
# cv2.destroyAllWindows()



# ## This is to download dataset instead of making our own (try it out)
# # Install dependencies as needed:
# # pip install kagglehub[pandas-datasets]
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# # Set the path to the file you'd like to load
# file_path = "./data"

# # Load the latest version
# df = kagglehub.load_dataset(
#   KaggleDatasetAdapter.PANDAS,
#   "datamunge/sign-language-mnist",
#   file_path,
#   # Provide any additional arguments like 
#   # sql_query or pandas_kwargs. See the 
#   # documenation for more information:
#   # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
# )

# print("First 5 records:", df.head())



import kagglehub
import os
import shutil

# Get folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create a local 'data' folder inside the script directory
local_folder = os.path.join(script_dir, "data")
os.makedirs(local_folder, exist_ok=True)
print("Local folder path:", local_folder)

# Download the ASL Alphabet dataset (latest version)
cache_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset downloaded to cache:", cache_path)

# Copy all files/folders from cache to local folder
for item in os.listdir(cache_path):
    src = os.path.join(cache_path, item)
    dst = os.path.join(local_folder, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset copied to local folder:", local_folder)
