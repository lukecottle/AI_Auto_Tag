import os
import io
from google.cloud import vision
import pandas as pd
import glob


csv_path = f'AI/alltags.csv'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'W:/Python/Keys/modelthumb-1a950608e4ed.json'
client = vision.ImageAnnotatorClient()

file_name = r'01-hermit-crabs-minden_00512417.jpg'
#image_path = f'AI/{file_name}'

for image_path in glob.glob('files/*/*.*.jpeg'):
	with io.open(image_path, 'rb') as image_file:
		content = image_file.read()

		image = vision.types.Image(content=content)
		response = client.label_detection(image=image)
		labels = response.label_annotations

		df = pd.DataFrame(columns=['description', 'score'])
		for label in labels:
			df = df.append(
				dict(
					description=label.description,
					score=label.score,
				), ignore_index=True
			)
		print(df)
		f = open(csv_path, 'a') # Open file as append mode
		df.to_csv(f, index=False, header = False)
		f.close()






