# https://docs.voyageai.com/docs/introduction
# Voyage AI provides cutting-edge embedding and rerankers.

import voyageai
import settings

vo = voyageai.Client(api_key = settings.VOYAGE_API_KEY)

result = vo.embed(["hello world"], model="voyage-3")

