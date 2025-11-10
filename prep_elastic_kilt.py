import argparse
import glob
import time
import json
from tqdm import tqdm
from src.retrieve.beir.beir.retrieval.search.lexical.elastic_search import ElasticSearch


def build_elasticsearch(
    kilt_file_pattern: str,
    index_name: str,
):
    kilt_files = glob.glob(kilt_file_pattern)
    print(f'#files {len(kilt_files)}')

    config = {
        'hostname': 'http://localhost:9200',
        'index_name': index_name,
        'keys': {'title': 'title', 'body': 'txt'},
        'timeout': 100,
        'retry_on_timeout': True,
        'maxsize': 24,
        'number_of_shards': 'default',
        'language': 'english',
    }
    es = ElasticSearch(config)

    # create index
    print(f'create index {index_name}')
    es.delete_index()
    time.sleep(5)
    es.create_index()

    # generator
    def generate_actions():
        for kilt_file in kilt_files:
            with open(kilt_file, 'r') as fin:
                for line in fin:
                    try:
                        obj = json.loads(line.strip())
                        _id = obj["global_id"]
                        title = obj.get("wikipedia_title", "")
                        text = obj.get("text", "")
                        es_doc = {
                            "_id": _id,
                            "_op_type": "index",
                            "refresh": "wait_for",
                            config["keys"]["title"]: title,
                            config["keys"]["body"]: text,
                        }
                        yield es_doc
                    except Exception as e:
                        print(f"Error parsing line: {e}")
                        continue

    # index
    progress = tqdm(unit='docs')
    es.bulk_add_to_index(
        generate_actions=generate_actions(),
        progress=progress,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument("--index_name", type=str, default="kilt", help="index name")
    args = parser.parse_args()
    build_elasticsearch(args.data_path, index_name=args.index_name)
