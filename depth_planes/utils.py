import os
import requests
import mimetypes
import cv2
from params import *

def download_image(image_url: str,
  file_dir: str,
  name: str):
  """
  Save loccaly an image from url.

  Args:
      image_url (str): _description_
      file_dir (str): _description_
      name (str): _description_
  """

  response = requests.get(image_url)
  extension = mimetypes.guess_extension(response.headers.get('content-type', '').split(';')[0])

  if response.status_code == 200:
      directory = os.path.dirname(file_dir)
      if not os.path.exists(directory):
          os.makedirs(directory)

      local_path = "{}/{}{}".format(file_dir,name,extension)

      with open(local_path, "wb") as fp:
          fp.write(response.content)

      return local_path, extension

      print(f"Image downloaded successfully at {file_dir}.")
  else:
      print(f"Failed to download the image. Status code: {response.status_code}")


def get_image_size(fname):
    img = cv2.imread(fname)
    return (img.shape[0], img.shape[1])

if __name__ == '__main__':
  download_image('https://blog.apify.com/content/images/2023/12/logo-dark.svg', LOCAL_DATA_PATH, 'test')
