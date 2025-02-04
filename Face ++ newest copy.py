import os
import csv
import time
import pandas as pd
import numpy as np
from base64 import b64encode
from collections import Counter
from tqdm import tqdm
import requests
from moviepy.editor import VideoFileClip
from pathlib import Path
from PIL import Image
import io
from datetime import datetime

# 基础日志设置
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_analysis_debug.log'),
        logging.StreamHandler()
    ]
)

# Face++ API settings
API_KEY = "///"
API_SECRET = "////"
API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# 定义分析字段
FRAME_ANALYSIS_FIELDS = [
    # 基础信息
    'Video_id', 'Frame_id', 'Frame_time', 'Video_duration', 'total_faces',

    # 情绪数据
    'sadness', 'neutral', 'disgust', 'anger', 'surprise', 'fear', 'happiness',
    'sadness_score', 'neutral_score', 'disgust_score', 'anger_score',
    'surprise_score', 'fear_score', 'happiness_score',

    # 人脸质量和模糊度
    'blur_value', 'blur_threshold',
    'face_quality', 'face_quality_threshold',

    # 时间戳
    'Timestamp'
]

# 定义汇总字段
SUMMARY_ANALYSIS_FIELDS = [
    # 基础信息
    'Video_id', 'Video_duration', 'total_frames', 'frames_with_faces',
    'total_faces', 'average_faces_per_frame',

    # 情绪统计
    'emotion_counts_sadness', 'emotion_counts_neutral', 'emotion_counts_disgust',
    'emotion_counts_anger', 'emotion_counts_surprise', 'emotion_counts_fear',
    'emotion_counts_happiness',

    # 情绪平均分数
    'emotion_avg_sadness', 'emotion_avg_neutral', 'emotion_avg_disgust',
    'emotion_avg_anger', 'emotion_avg_surprise', 'emotion_avg_fear',
    'emotion_avg_happiness',

    # 情绪最高分数
    'emotion_max_sadness', 'emotion_max_neutral', 'emotion_max_disgust',
    'emotion_max_anger', 'emotion_max_surprise', 'emotion_max_fear',
    'emotion_max_happiness',

    # 人脸质量和模糊度统计
    'avg_blur_value', 'min_blur_value', 'max_blur_value',
    'avg_face_quality', 'min_face_quality', 'max_face_quality',

    # 主导情绪
    'dominant_emotion',

    # 处理信息
    'process_time',
    'Timestamp'
]


def compress_image(image_path, max_size_mb=1.5):
    """压缩图像到指定大小以下"""
    try:
        with Image.open(image_path) as img:
            quality = 95
            output = io.BytesIO()

            while quality > 5:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality)
                if len(output.getvalue()) / (1024 * 1024) <= max_size_mb:
                    break
                quality -= 5

            with open(image_path, 'wb') as f:
                f.write(output.getvalue())
            return True

    except Exception as e:
        logging.error(f"Error compressing image {image_path}: {e}")
        return False


def extract_frames(video_path, output_folder, frame_interval=5):
    """提取视频的每一帧"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with VideoFileClip(video_path) as video:
            duration = video.duration
            total_frames = int(duration // frame_interval)

            for i in range(total_frames):
                current_time = i * frame_interval
                output_file = os.path.join(output_folder, f"frame_{i:04d}.jpg")
                video.save_frame(output_file, t=current_time)
                compress_image(output_file)

            return total_frames, duration
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {str(e)}")
        raise


def analyze_image(image_path, return_attributes, max_retries=5):
    """分析单个图像，增加重试机制和详细日志"""
    retries = 0
    while retries < max_retries:
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = b64encode(image_file.read()).decode('utf-8')

            data = {
                "api_key": API_KEY,
                "api_secret": API_SECRET,
                "image_base64": image_base64,
                "return_attributes": return_attributes
            }

            logging.info(f"Sending API request for {image_path} with attributes: {return_attributes}")
            response = requests.post(API_URL, data=data)

            if response.status_code == 200:
                result = response.json()
                logging.info(f"API response successful: {result}")
                return result
            elif response.status_code == 403 and "CONCURRENCY_LIMIT_EXCEEDED" in response.text:
                wait_time = (2 ** retries) * 1.0
                logging.warning(f"API limit exceeded, waiting {wait_time}s before retry {retries + 1}/{max_retries}")
                time.sleep(wait_time)
                retries += 1
            else:
                logging.error(f"API Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logging.error(f"Error analyzing {image_path}: {str(e)}")
            return None

    logging.error(f"Max retries exceeded for image {image_path}")
    return None


def process_frame_result(result, video_id, frame_id, video_duration, frame_time, analyze_emotion):
    """处理单帧分析结果，添加详细日志"""
    logging.info(f"Processing frame {frame_id} with analyze_emotion={analyze_emotion}")
    logging.info(f"Raw API result: {result}")

    row = {
        'Video_id': video_id,
        'Frame_id': frame_id,
        'Frame_time': frame_time,
        'Video_duration': video_duration,
        'total_faces': 0,
        'sadness': 0, 'neutral': 0, 'disgust': 0, 'anger': 0,
        'surprise': 0, 'fear': 0, 'happiness': 0,
        'sadness_score': 0.0, 'neutral_score': 0.0, 'disgust_score': 0.0,
        'anger_score': 0.0, 'surprise_score': 0.0, 'fear_score': 0.0,
        'happiness_score': 0.0,
        'blur_value': 0.0, 'blur_threshold': 0.0,
        'face_quality': 0.0, 'face_quality_threshold': 0.0,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if not result or 'faces' not in result or not result['faces']:
        logging.info(f"No valid faces in result for frame {frame_id}")
        return row

    faces = result['faces']
    total_faces = len(faces)
    total_blur_value = 0.0
    total_blur_threshold = 0.0
    total_face_quality = 0.0
    total_face_quality_threshold = 0.0

    emotion_counter = Counter()
    emotion_scores = Counter()

    logging.info(f"Processing {total_faces} faces in frame {frame_id}")

    for face_idx, face in enumerate(faces):
        if 'attributes' in face:
            attrs = face['attributes']
            logging.info(f"Processing face {face_idx} attributes: {attrs}")

            # 处理模糊度
            if 'blur' in attrs:
                blurness = attrs['blur'].get('blurness', {})
                blur_value = blurness.get('value', 0)
                blur_threshold = blurness.get('threshold', 0)
                total_blur_value += blur_value
                total_blur_threshold += blur_threshold
                logging.info(f"Face {face_idx} blur: value={blur_value}, threshold={blur_threshold}")

            # 处理人脸质量
            if 'facequality' in attrs:
                quality = attrs['facequality']
                quality_value = quality.get('value', 0)
                quality_threshold = quality.get('threshold', 0)
                total_face_quality += quality_value
                total_face_quality_threshold += quality_threshold
                logging.info(f"Face {face_idx} quality: value={quality_value}, threshold={quality_threshold}")

            # 处理情感
            if 'emotion' in attrs:
                emotions = attrs['emotion']
                logging.info(f"Face {face_idx} emotions: {emotions}")
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                emotion_counter[dominant_emotion] += 1
                for emotion, score in emotions.items():
                    emotion_scores[emotion + '_score'] += score

    # 计算平均值
    avg_blur_value = round(total_blur_value / total_faces, 2) if total_faces > 0 else 0.0
    avg_face_quality = round(total_face_quality / total_faces, 2) if total_faces > 0 else 0.0

    # 更新行数据
    row['total_faces'] = total_faces
    row['blur_value'] = avg_blur_value
    row['face_quality'] = avg_face_quality

    # 更新情绪数据
    emotions_list = ['sadness', 'neutral', 'disgust', 'anger', 'surprise', 'fear', 'happiness']
    for emotion in emotions_list:
        row[emotion] = emotion_counter[emotion]
        row[f'{emotion}_score'] = round(emotion_scores[f'{emotion}_score'] / total_faces, 2) if total_faces > 0 else 0.0

    logging.info(f"Frame {frame_id} final stats:")
    logging.info(f"- Faces: {total_faces}")
    logging.info(f"- Quality: {avg_face_quality}")
    logging.info(f"- Blur: {avg_blur_value}")
    logging.info(f"- Emotion counts: {dict(emotion_counter)}")
    logging.info(f"- Emotion scores: {dict((k, row[f'{k}_score']) for k in emotions_list)}")

    return row


def process_video(video_path, output_folder):
    """处理单个视频"""
    video_id = Path(video_path).stem
    frames_folder = Path(output_folder) / video_id

    logging.info(f"Processing video: {video_id}")

    frames_data = []
    has_faces = False
    total_faces = 0
    total_quality = 0
    total_blur = 0
    frames_with_faces = 0
    first_pass_data = []  # 存储第一遍的所有数据

    try:
        # 第一遍处理：评估整个视频的质量
        total_frames, video_duration = extract_frames(video_path, frames_folder)
        logging.info(f"Extracted {total_frames} frames from video {video_id}")

        image_files = sorted(frames_folder.glob('*.jpg'))
        qualified_frames = {}  # 存储合格帧的信息

        # 第一遍：收集视频整体统计数据
        logging.info("First pass: collecting video statistics...")
        for image_file in tqdm(image_files, desc=f"First pass for {video_id}", leave=False):
            frame_id = image_file.stem
            frame_time = int(frame_id.split('_')[1])

            result = analyze_image(str(image_file), "blur,facequality")
            row = process_frame_result(result, video_id, frame_id, video_duration, frame_time, analyze_emotion=False)
            first_pass_data.append(row)  # 保存第一遍的所有数据

            if result and 'faces' in result and result['faces']:
                has_faces = True
                faces_count = len(result['faces'])
                total_faces += faces_count
                if faces_count > 0:
                    frames_with_faces += 1
                    total_quality += row['face_quality']
                    total_blur += row['blur_value']

                    # 记录符合条件的帧
                    if row['face_quality'] >= 80 and row['blur_value'] <= 20:
                        qualified_frames[frame_id] = {
                            'path': str(image_file),
                            'frame_time': frame_time,
                            'first_pass_row': row  # 保存第一遍的完整数据
                        }

            time.sleep(0.1)

        # 计算视频整体指标
        if frames_with_faces > 0:
            avg_quality = total_quality / frames_with_faces
            avg_blur = total_blur / frames_with_faces

            logging.info(f"Video statistics for {video_id}:")
            logging.info(f"- Total faces: {total_faces}")
            logging.info(f"- Average quality: {avg_quality:.2f}")
            logging.info(f"- Average blur: {avg_blur:.2f}")
            logging.info(f"- Qualified frames: {len(qualified_frames)}")

            # 判断视频是否满足整体条件
            if avg_quality >= 80 and avg_blur <= 20 and total_faces >= 1:
                logging.info(f"Video {video_id} meets criteria for emotion analysis")
                frames_data = []

                # 处理每一帧
                for row in first_pass_data:
                    frame_id = row['Frame_id']
                    if frame_id in qualified_frames:
                        # 对合格帧进行情感分析
                        logging.info(f"Analyzing emotions for qualified frame {frame_id}")
                        result_emotion = analyze_image(qualified_frames[frame_id]['path'], "emotion")

                        if result_emotion and 'faces' in result_emotion and result_emotion['faces']:
                            # 创建新的row，合并第一遍的基础数据和第二遍的情感数据
                            emotion_row = process_frame_result(result_emotion, video_id, frame_id,
                                                               video_duration, qualified_frames[frame_id]['frame_time'],
                                                               analyze_emotion=True)

                            # 保留第一遍的所有非情感数据
                            for key in row.keys():
                                if key not in ['sadness', 'neutral', 'disgust', 'anger', 'surprise',
                                               'fear', 'happiness', 'sadness_score', 'neutral_score',
                                               'disgust_score', 'anger_score', 'surprise_score',
                                               'fear_score', 'happiness_score']:
                                    emotion_row[key] = row[key]

                            frames_data.append(emotion_row)
                        else:
                            frames_data.append(row)  # 如果情感分析失败，使用第一遍数据

                        time.sleep(0.1)
                    else:
                        # 对于不合格的帧，直接使用第一遍数据
                        frames_data.append(row)
            else:
                logging.info(f"Video {video_id} does not meet criteria for emotion analysis:")
                logging.info(f"- Average quality: {avg_quality:.2f} (needs >= 50)")
                logging.info(f"- Average blur: {avg_blur:.2f} (needs <= 20)")
                logging.info(f"- Total faces: {total_faces} (needs >= 5)")
                frames_data = first_pass_data  # 使用第一遍的数据

        return has_faces, frames_data

    except Exception as e:
        logging.error(f"Error processing video {video_id}: {str(e)}")
        return False, []


def calculate_video_summary(frames_data):
    """计算视频汇总统计数据"""
    if not frames_data:
        return None

    video_id = frames_data[0]['Video_id']
    logging.info(f"Calculating summary for video {video_id}")

    summary = {
        'Video_id': video_id,
        'Video_duration': frames_data[0]['Video_duration'],
        'total_frames': len(frames_data),
        'frames_with_faces': 0,
        'total_faces': 0,
        'average_faces_per_frame': 0.0,
        'emotion_counts_sadness': 0, 'emotion_counts_neutral': 0, 'emotion_counts_disgust': 0,
        'emotion_counts_anger': 0, 'emotion_counts_surprise': 0, 'emotion_counts_fear': 0,
        'emotion_counts_happiness': 0,
        'emotion_avg_sadness': 0.0, 'emotion_avg_neutral': 0.0, 'emotion_avg_disgust': 0.0,
        'emotion_avg_anger': 0.0, 'emotion_avg_surprise': 0.0, 'emotion_avg_fear': 0.0,
        'emotion_avg_happiness': 0.0,
        'emotion_max_sadness': 0.0, 'emotion_max_neutral': 0.0, 'emotion_max_disgust': 0.0,
        'emotion_max_anger': 0.0, 'emotion_max_surprise': 0.0, 'emotion_max_fear': 0.0,
        'emotion_max_happiness': 0.0,
        'avg_blur_value': 0.0, 'min_blur_value': 0.0, 'max_blur_value': 0.0,
        'avg_face_quality': 0.0, 'min_face_quality': 0.0, 'max_face_quality': 0.0,
        'dominant_emotion': 'none',
        'process_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 基于第一遍数据的统计（所有帧）
    frames_with_faces = [frame for frame in frames_data if frame['total_faces'] > 0]
    summary['frames_with_faces'] = len(frames_with_faces)
    summary['total_faces'] = sum(frame['total_faces'] for frame in frames_data)
    summary['average_faces_per_frame'] = round(summary['total_faces'] / len(frames_data), 2) if frames_data else 0.0

    logging.info(f"Basic statistics for video {video_id}:")
    logging.info(f"- Total frames: {summary['total_frames']}")
    logging.info(f"- Frames with faces: {summary['frames_with_faces']}")
    logging.info(f"- Total faces: {summary['total_faces']}")
    logging.info(f"- Average faces per frame: {summary['average_faces_per_frame']}")

    # 基于第一遍数据的质量统计（所有有人脸的帧）
    blur_values = [frame['blur_value'] for frame in frames_with_faces]
    face_quality_values = [frame['face_quality'] for frame in frames_with_faces]

    if blur_values:
        summary['avg_blur_value'] = round(np.mean(blur_values), 2)
        summary['min_blur_value'] = round(np.min(blur_values), 2)
        summary['max_blur_value'] = round(np.max(blur_values), 2)

    if face_quality_values:
        summary['avg_face_quality'] = round(np.mean(face_quality_values), 2)
        summary['min_face_quality'] = round(np.min(face_quality_values), 2)
        summary['max_face_quality'] = round(np.max(face_quality_values), 2)

    logging.info(f"Quality metrics for video {video_id}:")
    logging.info(
        f"- Blur (avg/min/max): {summary['avg_blur_value']}/{summary['min_blur_value']}/{summary['max_blur_value']}")
    logging.info(
        f"- Quality (avg/min/max): {summary['avg_face_quality']}/{summary['min_face_quality']}/{summary['max_face_quality']}")

    # 基于第二遍数据的情感统计（只统计进行了情感分析的帧）
    emotions = ['sadness', 'neutral', 'disgust', 'anger', 'surprise', 'fear', 'happiness']
    emotion_counts = Counter()
    emotion_score_totals = {emotion: 0.0 for emotion in emotions}
    emotion_score_max = {emotion: 0.0 for emotion in emotions}

    # 找出进行了情感分析的帧（至少有一个情感分数大于0）
    frames_with_emotion = [
        frame for frame in frames_data
        if any(frame[f'{emotion}_score'] > 0 for emotion in emotions)
    ]

    for frame in frames_with_emotion:
        # 确定帧的主导情绪
        emotion_scores = {emotion: frame[f'{emotion}_score'] for emotion in emotions}
        max_score = max(emotion_scores.values())
        if max_score > 0:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant_emotion] += 1

        # 更新情绪统计
        for emotion in emotions:
            score = frame[f'{emotion}_score']
            emotion_score_totals[emotion] += score
            if score > emotion_score_max[emotion]:
                emotion_score_max[emotion] = score

    # 更新情绪统计数据
    num_emotion_frames = len(frames_with_emotion) if frames_with_emotion else 1
    for emotion in emotions:
        summary[f'emotion_counts_{emotion}'] = emotion_counts[emotion]
        # 情感平均分数只考虑进行了情感分析的帧
        summary[f'emotion_avg_{emotion}'] = round(emotion_score_totals[emotion] / num_emotion_frames, 2)
        summary[f'emotion_max_{emotion}'] = round(emotion_score_max[emotion], 2)

    # 确定主导情绪
    if emotion_counts:
        summary['dominant_emotion'] = emotion_counts.most_common(1)[0][0]
    else:
        summary['dominant_emotion'] = 'none'

    logging.info(f"Emotion statistics for video {video_id}:")
    logging.info(f"- Frames with emotion analysis: {len(frames_with_emotion)}")
    logging.info(f"- Emotion counts: {dict(emotion_counts)}")
    logging.info(f"- Dominant emotion: {summary['dominant_emotion']}")
    for emotion in emotions:
        logging.info(f"- {emotion}: avg={summary[f'emotion_avg_{emotion}']} max={summary[f'emotion_max_{emotion}']}")

    return summary

def main():
    # 基础路径设置
    base_path = Path("////")
    output_folder = base_path / "face ++ output"
    output_folder.mkdir(exist_ok=True)

    # 设置CSV文件路径
    frame_csv_path = output_folder / "frame_analysis.csv"
    summary_csv_path = output_folder / "video_summary.csv"

    # 创建CSV文件并写入表头（如果不存在）
    if not frame_csv_path.exists():
        with open(frame_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_ANALYSIS_FIELDS)
            writer.writeheader()
    if not summary_csv_path.exists():
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_ANALYSIS_FIELDS)
            writer.writeheader()

    # 读取已处理的视频ID列表
    processed_videos = set()
    if summary_csv_path.exists():
        df_summary = pd.read_csv(summary_csv_path)
        processed_videos.update(df_summary['Video_id'].astype(str).tolist())

    # 收集视频文件
    video_files = list(base_path.glob('**/*.mp4'))
    total_videos = len(video_files)
    logging.info(f"Found {total_videos} videos.")

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_id = video_file.stem

        # 检查视频是否已经处理过
        if video_id in processed_videos:
            logging.info(f"Video {video_id} already processed. Skipping.")
            continue

        try:
            logging.info(f"\nProcessing video: {video_id}")
            # 处理视频
            _, frames_data = process_video(str(video_file), str(output_folder))

            # 只要有frames_data就写入
            if frames_data:
                # 写入帧数据
                with open(frame_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=FRAME_ANALYSIS_FIELDS)
                    for frame_data in frames_data:
                        writer.writerow(frame_data)

                # 计算并写入汇总数据
                summary_data = calculate_video_summary(frames_data)
                if summary_data:
                    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=SUMMARY_ANALYSIS_FIELDS)
                        writer.writerow(summary_data)

                logging.info(f"Successfully processed video {video_id}")
                processed_videos.add(video_id)
            else:
                logging.error(f"No frames data generated for video {video_id}")

        except Exception as e:
            logging.error(f"Error processing video {video_id}: {str(e)}")
            continue

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()