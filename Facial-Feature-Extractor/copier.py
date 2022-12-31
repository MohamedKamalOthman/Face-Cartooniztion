import shutil

# copy files in subdirectories to a single parent directory
if __name__ == "__main__":
    k = 0
    for i in range(1679):
        for j in range(20):
            try:
                shutil.copy2(f'./{i}/{j}.jpg', f'./output/{k}.jpg')
                k += 1
            except:
                continue
