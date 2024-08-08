import os
for file in os.listdir():
    if file.endswith('.ui'):
        os.system('pyuic5 {} > ui_{}.py'.format(file,file[:-3]))
        os.system('uic -o ui_{}.h {}'.format(file[:-3],file))