import logging

import numpy as np
from pandas import DataFrame

log = logging.getLogger(__name__)


# Extra metadata field names for custom ticket dataset
EXTRA_METADATA_FIELD_NAMES = [
    "_tenant", "account_id", "workspace_id", "ticket_id",
    "ticket_type", "ticket_status", "catalog_item_ids", "created_at"
]


def get_data(data_df: DataFrame, normalize: bool) -> tuple[list[list[float]], list[str], dict[str, list] | None]:
    all_metadata = data_df["id"].tolist()
    emb_np = np.stack(data_df["emb"])
    if normalize:
        log.debug("normalize the 100k train data")
        all_embeddings = (emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis]).tolist()
    else:
        all_embeddings = emb_np.tolist()

    # Extract extra metadata fields if they exist in the dataframe
    extra_metadata_fields = None
    available_extra_fields = [f for f in EXTRA_METADATA_FIELD_NAMES if f in data_df.columns]
    if available_extra_fields:
        extra_metadata_fields = {
            field: data_df[field].tolist()
            for field in available_extra_fields
        }
        log.debug(f"Extracted {len(available_extra_fields)} extra metadata fields: {available_extra_fields}")

    return all_embeddings, all_metadata, extra_metadata_fields
