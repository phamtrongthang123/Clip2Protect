from deepface import DeepFace

def get_gender(img_path, shorten=True):
    result = DeepFace.analyze(img_path = img_path,
        actions = ['gender']
    )
    if not shorten:
        return result
    else:
        return result[0]['dominant_gender']

if __name__ == '__main__':
    print(get_gender('047073.jpg'))