import os
import csv
import json
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_array_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    return obj


def analyze_emotions(folder_path):
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_plus_base"
    )

    results = []
    official_emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
    emotion_mapping = {
        '生气': 'angry', '愤怒': 'angry',
        '厌恶': 'disgusted',
        '害怕': 'fearful', '恐惧': 'fearful',
        '开心': 'happy', '高兴': 'happy',
        '中立': 'neutral',
        '其他': 'other',
        '难过': 'sad', '悲伤': 'sad',
        '惊讶': 'surprised',
        '<unk>': 'unknown', '未知': 'unknown'
    }
    detected_emotions = set()

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            try:
                print(f"Processing: {filename}")
                rec_result = inference_pipeline(file_path, output_dir="./outputs")

                serializable_result = numpy_array_to_list(rec_result)

                print(f"Raw result for {filename}:")
                print(json.dumps(serializable_result, indent=2))

                if isinstance(serializable_result, list) and len(serializable_result) > 0:
                    first_result = serializable_result[0]
                    if 'labels' in first_result and 'scores' in first_result:
                        labels = first_result['labels']
                        scores = first_result['scores']

                        emotion_scores = {emotion: 0 for emotion in official_emotions}

                        for label, score in zip(labels, scores):
                            emotion = label.split('/')[-1].lower()
                            mapped_emotion = emotion_mapping.get(emotion, emotion)
                            if mapped_emotion in official_emotions:
                                emotion_scores[mapped_emotion] = score
                                detected_emotions.add(mapped_emotion)

                        predicted_emotion = max(emotion_scores, key=emotion_scores.get)

                        results.append((filename, emotion_scores, predicted_emotion))
                        print(f"Processed: {filename} - Predicted emotion: {predicted_emotion}")
                    else:
                        print(f"Warning: Unexpected result format for {filename}")
                else:
                    print(f"Warning: Unexpected result format for {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results.append((filename, {emotion: 0 for emotion in official_emotions}, "Error"))

    print(f"Emotions actually detected by the model: {detected_emotions}")
    return results, official_emotions


def save_to_csv(results, emotions, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Audio File Name'] + emotions + ['Predicted Emotion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for filename, emotion_scores, predicted_emotion in results:
            row = {'Audio File Name': filename, 'Predicted Emotion': predicted_emotion}
            row.update(emotion_scores)
            writer.writerow(row)
    print(f"CSV file saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    folder_path = r"/Users/zhoutieyu/Desktop/testing"  # 换成你自己的
    output_csv = "emotion_analysis_results.csv"

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
    else:
        results, emotions = analyze_emotions(folder_path)
        if results:
            save_to_csv(results, emotions, output_csv)
            print(f"Analysis complete. Results saved to {output_csv}")
        else:
            print("No results to save. Check if there are MP3 files in the specified folder.")