## EVALUATION

import sklearn
import re
sklearn.metrics.mutual_info_score(
    news_topics['topic_num'].astype(int),
    news_topics['topic_1_name'].apply(lambda x: int(re.findall("\d+", x)[0])).tolist()
)

