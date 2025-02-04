import csv
import json
import time
from alibabacloud_alinlp20200629.client import Client as AliyunClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_alinlp20200629 import models as alinlp_models


def create_client(access_key_id: str, access_key_secret: str) -> AliyunClient:
    config = open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret
    )
    config.endpoint = 'alinlp.cn-hangzhou.aliyuncs.com'
    return AliyunClient(config)


def sentiment_analysis(client: AliyunClient, text: str):
    request = alinlp_models.GetSaChGeneralRequest(
        service_code='alinlp',
        text=text
    )
    try:
        response = client.get_sa_ch_general(request)
        return response.body.to_map()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None


def process_csv(input_file: str, output_file: str, client: AliyunClient):
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['sentiment', 'positive_prob', 'negative_prob']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = row['Transcription']
            if len(text) > 1000:
                text = text[:1000]  # 限制文本长度不超过1000字
            result = sentiment_analysis(client, text)

            if result and 'Data' in result:
                data = json.loads(result['Data'])
                row['sentiment'] = data['result']['sentiment']
                row['positive_prob'] = data['result']['positive_prob']
                row['negative_prob'] = data['result']['negative_prob']
            else:
                row['sentiment'] = 'unknown'
                row['positive_prob'] = 0.0
                row['negative_prob'] = 0.0

            writer.writerow(row)
            time.sleep(1)  # 为了避免超过API调用限制，在每次请求后暂停1秒

    print(f"分析完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    access_key_id = ""
    access_key_secret = ""

    client = create_client(access_key_id, access_key_secret)

    input_file = "demo.csv"  # 请替换为您的输入文件名
    output_file = "text_detection.csv"  # 输出文件名


    process_csv(input_file, output_file, client)
