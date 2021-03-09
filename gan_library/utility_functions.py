import os
#simple utility function which attempts to make a directory given a filepath
#space efficient way to prevent getting error messages if directory already exists
def build_directory(filepath):
    try:
        os.mkdir(f'{filepath}')
    except:
        print('already exists')
